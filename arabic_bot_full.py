# ---------------------------
# arabic_bot_full.py - Part 1
# ---------------------------

import os
import sqlite3
import asyncio
import random
from datetime import datetime, timedelta, date, time as dtime
from zoneinfo import ZoneInfo
from typing import Optional, List, Tuple

import discord
from discord import app_commands, ui
from discord.ext import tasks, commands

# Embedding / ML
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
# ---------------- GLOBALS & CONFIG HELPERS ----------------
# Challenge behavior
CHALLENGE_TIMEOUT_SECONDS = 90
CHALLENGE_SIM_THRESHOLD = 0.50
POINTS_PER_CORRECT = 5

# Admin helpers: set OWNER_IDS env var to a comma-separated list of Discord IDs,
# or the code will fall back to username checks below.
OWNER_IDS = [848935997358211122]
_owner_env = os.getenv("OWNER_IDS", "")
if _owner_env:
    for part in _owner_env.split(","):
        try:
            OWNER_IDS.add(int(part.strip()))
        except:
            pass

# A small set of username-based admin fallbacks (lowercase)
ADMIN_USERNAMES = {"baja1121", "baja", "justin", "justin baja"}

# In-memory caches to make scoring fast and non-blocking
# struggle_embedding_cache: struggle_id -> numpy array (float32)
struggle_embedding_cache: dict = {}
# also cache struggle metadata (optional): struggle_id -> (guild_id, user_id, word, definition)
struggle_meta_cache: dict = {}
# active_challenges: maps user_id -> dict with keys: struggle_id, definition, word, expires_at (datetime)
active_challenges = {}
# small in-memory lock to avoid race conditions
_active_challenge_lock = asyncio.Lock()

# Helper: check admin membership for interaction/member
def is_member_admin(member: discord.Member):
    if not member:
        return False
    if member.id in OWNER_IDS:
        return True
    uname = str(member).split("#")[0].lower()
    if uname in ADMIN_USERNAMES:
        return True
    # check guild perms as fallback
    try:
        return member.guild_permissions.administrator
    except:
        return False

TOKEN = os.getenv("DISCORD_TOKEN")
TIMEZONE = ZoneInfo("America/Chicago")
REMINDER_HOUR = 12  # 12:00 PM CST
REMINDER_MINUTE = 0
GUILD_ID = 1438553047344353293  # set to your guild id or None to run everywhere
# ----------------------------------------

# ---------- Intents and Bot ----------
intents = discord.Intents.default()
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix="/", intents=intents)
tree = bot.tree

# ---------- Database connection (persistent migration for Railway) ----------
import os, shutil

ORIGINAL_DB = "arabic_bot.db"
PERSISTENT_DB = "/data/arabic_bot.db"

# If persistent DB doesn't exist yet but original does, migrate it
if os.path.exists(ORIGINAL_DB) and not os.path.exists(PERSISTENT_DB):
    print("Migrating existing database to Railway persistent storage...")
    shutil.copy(ORIGINAL_DB, PERSISTENT_DB)

# Always connect to persistent DB from now on
DB = PERSISTENT_DB
conn = sqlite3.connect(DB, check_same_thread=False)
c = conn.cursor()


# --------------------
# TABLE CREATION
# --------------------
# Existing core tables
c.execute("""
CREATE TABLE IF NOT EXISTS completions (
    id INTEGER PRIMARY KEY,
    guild_id INTEGER,
    user_id INTEGER,
    username TEXT,
    date TEXT,
    time TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS streaks (
    guild_id INTEGER,
    user_id INTEGER,
    username TEXT,
    streak INTEGER DEFAULT 0,
    last_done_date TEXT,
    total_completions INTEGER DEFAULT 0,
    PRIMARY KEY (guild_id, user_id)
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS settings (
    guild_id INTEGER PRIMARY KEY,
    channel_id INTEGER,
    ping_role_id INTEGER,
    ping_enabled INTEGER DEFAULT 0,
    last_reminder_date TEXT
)
""")

# New gamification tables
c.execute("""
CREATE TABLE IF NOT EXISTS struggle_words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER,
    user_id INTEGER,
    word TEXT,
    definition TEXT,
    UNIQUE(guild_id, user_id, word)
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS struggle_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    struggle_id INTEGER,
    embedding BLOB
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS points (
    guild_id INTEGER,
    user_id INTEGER,
    points INTEGER DEFAULT 0,
    PRIMARY KEY (guild_id, user_id)
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS titles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    price INTEGER
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS user_titles (
    guild_id INTEGER,
    user_id INTEGER,
    title_id INTEGER,
    PRIMARY KEY (guild_id, user_id)
)
""")

# --------------------
# INVENTORY & FREEZE TABLES (NEW)
# --------------------

# Multi-owner titles (allow many titles per user)
c.execute("""
CREATE TABLE IF NOT EXISTS user_titles_multi (
    guild_id INTEGER,
    user_id INTEGER,
    title_id INTEGER,
    PRIMARY KEY (guild_id, user_id, title_id)
)
""")

# Equipped title (which title the user currently wears — at most one)
c.execute("""
CREATE TABLE IF NOT EXISTS user_equipped_titles (
    guild_id INTEGER,
    user_id INTEGER,
    title_id INTEGER,
    PRIMARY KEY (guild_id, user_id)
)
""")

# Streak freeze counts
c.execute("""
CREATE TABLE IF NOT EXISTS streak_freezes (
    guild_id INTEGER,
    user_id INTEGER,
    freezes INTEGER DEFAULT 0,
    PRIMARY KEY (guild_id, user_id)
)
""")

# Migrate any existing single-title rows from old user_titles to user_titles_multi (safe no-op if user_titles empty)
try:
    c.execute("""
        INSERT OR IGNORE INTO user_titles_multi (guild_id, user_id, title_id)
        SELECT guild_id, user_id, title_id FROM user_titles
    """)
    conn.commit()
except Exception:
    # migration best-effort; ignore if structure differs
    pass

# --- PATCH: Ensure last_summary_date column exists ---
try:
    c.execute("ALTER TABLE settings ADD COLUMN last_summary_date TEXT")
    conn.commit()
except:
    pass


# --------------------
# EMBEDDING MODEL LOAD
# --------------------
# Load once at module import (small model)
# Note: this may increase cold-start time but avoids repeated loads.
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --------------------
# BASIC HELPERS
# --------------------
def now_cst():
    return datetime.now(TIMEZONE)

def today_cst_str():
    return now_cst().date().isoformat()

def yesterday_cst_str():
    return (now_cst().date() - timedelta(days=1)).isoformat()

# ---- Settings helpers (unchanged semantics) ----
def get_settings(guild_id: int):
    c.execute("SELECT channel_id, ping_role_id, ping_enabled, last_reminder_date FROM settings WHERE guild_id = ?", (guild_id,))
    row = c.fetchone()
    if row:
        return {"channel_id": row[0], "ping_role_id": row[1], "ping_enabled": bool(row[2]), "last_reminder_date": row[3]}
    return {"channel_id": None, "ping_role_id": None, "ping_enabled": False, "last_reminder_date": None}

def upsert_settings(guild_id: int, **kwargs):
    existing = get_settings(guild_id)
    if existing["channel_id"] is None and "channel_id" not in kwargs and "ping_role_id" not in kwargs and "ping_enabled" not in kwargs:
        c.execute("INSERT OR IGNORE INTO settings (guild_id, channel_id, ping_role_id, ping_enabled, last_reminder_date) VALUES (?, ?, ?, ?, ?)",
                  (guild_id, None, None, 0, None))
        conn.commit()
        existing = get_settings(guild_id)
    if "channel_id" in kwargs:
        c.execute("UPDATE settings SET channel_id = ? WHERE guild_id = ?", (kwargs["channel_id"], guild_id))
    if "ping_role_id" in kwargs:
        c.execute("UPDATE settings SET ping_role_id = ? WHERE guild_id = ?", (kwargs["ping_role_id"], guild_id))
    if "ping_enabled" in kwargs:
        c.execute("UPDATE settings SET ping_enabled = ? WHERE guild_id = ?", (1 if kwargs["ping_enabled"] else 0, guild_id))
    if "last_reminder_date" in kwargs:
        c.execute("UPDATE settings SET last_reminder_date = ? WHERE guild_id = ?", (kwargs["last_reminder_date"], guild_id))
    conn.commit()

# ---- Completions & streak helpers (kept from original) ----
def record_completion(guild_id: int, user: discord.Member, date_str: str, time_str: str):
    c.execute("SELECT 1 FROM completions WHERE guild_id = ? AND user_id = ? AND date = ?", (guild_id, user.id, date_str))
    if c.fetchone():
        return False

    c.execute("INSERT INTO completions (guild_id, user_id, username, date, time) VALUES (?, ?, ?, ?, ?)",
              (guild_id, user.id, str(user), date_str, time_str))
    c.execute("SELECT streak, last_done_date, total_completions FROM streaks WHERE guild_id = ? AND user_id = ?", (guild_id, user.id))
    row = c.fetchone()
    if row:
        streak_val, last_done_date, total = row
        if last_done_date == date_str:
            pass
        else:
            if last_done_date == (date.fromisoformat(date_str) - timedelta(days=1)).isoformat():
                streak_val += 1
            else:
                streak_val = 1
            total = (total or 0) + 1
            c.execute("UPDATE streaks SET streak = ?, last_done_date = ?, username = ?, total_completions = ? WHERE guild_id = ? AND user_id = ?",
                      (streak_val, date_str, str(user), total, guild_id, user.id))
    else:
        streak_val = 1
        total = 1
        c.execute("INSERT INTO streaks (guild_id, user_id, username, streak, last_done_date, total_completions) VALUES (?, ?, ?, ?, ?, ?)",
                  (guild_id, user.id, str(user), streak_val, date_str, total))
    conn.commit()
    return True

def get_today_completions(guild_id: int):
    date_str = today_cst_str()
    c.execute("SELECT username, time FROM completions WHERE guild_id = ? AND date = ?", (guild_id, date_str))
    return c.fetchall()

def get_user_streak(guild_id: int, user_id: int):
    c.execute("SELECT streak, last_done_date, total_completions FROM streaks WHERE guild_id = ? AND user_id = ?", (guild_id, user_id))
    row = c.fetchone()
    if row:
        return {"streak": row[0], "last_done_date": row[1], "total": row[2]}
    return {"streak": 0, "last_done_date": None, "total": 0}

def get_leaderboard_streaks(guild_id: int, limit=10):
    c.execute("""
        SELECT username, streak, total_completions FROM streaks
        WHERE guild_id = ?
        ORDER BY streak DESC, total_completions DESC
        LIMIT ?
    """, (guild_id, limit))
    return c.fetchall()

def reset_streaks_for_missed_yesterday(guild: discord.Guild):
    y = yesterday_cst_str()
    c.execute("SELECT DISTINCT user_id FROM completions WHERE guild_id = ? AND date = ?", (guild.id, y))
    done_ids = {row[0] for row in c.fetchall()}

    settings = get_settings(guild.id)
    tracked = []
    if settings.get("ping_role_id"):
        role = guild.get_role(settings.get("ping_role_id"))
        if role:
            tracked = [m for m in role.members if not m.bot]
    else:
        tracked = [m for m in guild.members if not m.bot]

    missed = []
    for m in tracked:
        if m.id not in done_ids:
            # If user has a freeze, consume it and keep streak intact
            if consume_freeze_if_available(guild.id, m.id):
                # consume_freeze_if_available already decremented DB; leave streak as-is
                continue

            # otherwise reset streak
            c.execute("INSERT OR IGNORE INTO streaks (guild_id, user_id, username, streak, last_done_date, total_completions) VALUES (?, ?, ?, ?, ?, ?)",
                      (guild.id, m.id, str(m), 0, None, 0))
            c.execute("UPDATE streaks SET streak = 0 WHERE guild_id = ? AND user_id = ?", (guild.id, m.id))
            missed.append(m)
    conn.commit()
    return missed


# --------------------
# END OF QUADRANT 1
# --------------------


# ---------------------------
# arabic_bot_full.py - Part 2
# ---------------------------

# ----------------------------------------
# STRUGGLE WORD HELPERS
# ----------------------------------------

def embed_text(text: str) -> bytes:
    """Generate embedding bytes from text.

    Note: keep behavior same (returns bytes) — caller must store in DB.
    """
    vec = embedding_model.encode(text)
    arr = np.asarray(vec, dtype=np.float32)
    return arr.tobytes()

def get_embedding_array(blob: bytes) -> np.ndarray:
    """Convert stored blob back to array."""
    return np.frombuffer(blob, dtype=np.float32)

def add_struggle_word(guild_id: int, user_id: int, word: str, definition: str):
    """Insert a struggle word + its embedding and cache it for faster scoring."""
    word_l = word.lower()
    def_l = definition.lower()
    try:
        c.execute("""
            INSERT INTO struggle_words (guild_id, user_id, word, definition)
            VALUES (?, ?, ?, ?)
        """, (guild_id, user_id, word_l, def_l))
        conn.commit()
    except sqlite3.IntegrityError:
        return False

    # Insert embedding
    struggle_id = c.lastrowid
    emb_bytes = embed_text(definition)
    c.execute("""
        INSERT INTO struggle_embeddings (struggle_id, embedding)
        VALUES (?, ?)
    """, (struggle_id, emb_bytes))
    conn.commit()

    # Cache embedding and meta
    try:
        arr = np.frombuffer(emb_bytes, dtype=np.float32)
        struggle_embedding_cache[int(struggle_id)] = arr
        struggle_meta_cache[int(struggle_id)] = (int(guild_id), int(user_id), word_l, def_l)
    except Exception:
        pass

    return True


def remove_struggle_word(guild_id: int, user_id: int, word: str):
    """Remove struggle word."""
    c.execute("""
        SELECT id FROM struggle_words
        WHERE guild_id = ? AND user_id = ? AND word = ?
    """, (guild_id, user_id, word.lower()))
    row = c.fetchone()
    if not row:
        return False

    struggle_id = row[0]
    c.execute("DELETE FROM struggle_embeddings WHERE struggle_id = ?", (struggle_id,))
    c.execute("DELETE FROM struggle_words WHERE id = ?", (struggle_id,))
    conn.commit()
    return True

def get_user_struggle_words(guild_id: int, user_id: int):
    c.execute("""
        SELECT id, word, definition FROM struggle_words
        WHERE guild_id = ? AND user_id = ?
    """, (guild_id, user_id))
    return c.fetchall()

# Keep a tiny per-user last-served memory to avoid immediate repeats
_user_last_word = {}

def get_random_struggle_word(guild_id: int, user_id: int):
    rows = get_user_struggle_words(guild_id, user_id)
    if not rows:
        return None
    # rows is list of (id, word, definition)
    if len(rows) == 1:
        _user_last_word[user_id] = rows[0][0]
        return rows[0]
    # Prefer a different word than last served
    last_id = _user_last_word.get(user_id)
    choices = [r for r in rows if r[0] != last_id]
    choice = random.choice(choices) if choices else random.choice(rows)
    _user_last_word[user_id] = choice[0]
    return choice


def evaluate_answer(user_answer: str, correct_definition: str, struggle_id: int):
    """
    Compare embeddings of the correct definition vs. user's answer.
    Returns similarity (0 to ~1).
    This is CPU-bound; call via asyncio.to_thread() for non-blocking behavior.
    """
    # Try cache first
    stored_vec = struggle_embedding_cache.get(int(struggle_id))
    if stored_vec is None:
        # Fallback to DB read and then cache
        c.execute("SELECT embedding FROM struggle_embeddings WHERE struggle_id = ?", (struggle_id,))
        row = c.fetchone()
        if not row:
            return 0.0
        try:
            stored_vec = get_embedding_array(row[0])
            struggle_embedding_cache[int(struggle_id)] = stored_vec
        except Exception:
            return 0.0

    # Embed user's answer (this is the expensive step)
    ans_vec = embedding_model.encode(user_answer)
    ans_vec = np.asarray(ans_vec, dtype=np.float32)

    # cosine_similarity expects 2D arrays
    sim = cosine_similarity([stored_vec], [ans_vec])[0][0]
    # clamp to [0,1] for safety
    if sim != sim:  # NaN guard
        return 0.0
    return float(max(0.0, min(1.0, sim)))


async def send_reminder_message(guild: discord.Guild):
    settings = get_settings(guild.id)
    channel_id = settings.get("channel_id")
    ping_role_id = settings.get("ping_role_id")
    ping_enabled = settings.get("ping_enabled")

    if not channel_id:
        return

    channel = guild.get_channel(channel_id)
    if not channel:
        return

    role_txt = f"<@&{ping_role_id}>" if (ping_enabled and ping_role_id) else ""

    embed = discord.Embed(
        title="📘 Daily Arabic Reminder",
        description="Don't forget to do your Arabic today!",
        color=0x00B2FF
    )

    await channel.send(content=role_txt, embed=embed)


async def send_summary_embed(guild: discord.Guild):
    rows = get_today_completions(guild.id)

    embed = discord.Embed(
        title="🌙 Daily Summary",
        description="Here’s who completed their Arabic today.",
        color=0xFFD700
    )

    if rows:
        for username, t in rows:

            # Try to get the actual member to display title
            member = guild.get_member_named(username)
            if member:
                name = get_display_name_with_title(guild.id, member)
            else:
                name = username  # fallback

            embed.add_field(
                name=name,
                value=f"Completed at {t}",
                inline=False
            )
    else:
        embed.add_field(
            name="Nobody completed today 😭",
            value="Try again tomorrow!",
            inline=False
        )

    # Get summary channel
    settings = get_settings(guild.id)
    channel_id = settings.get("channel_id")
    if not channel_id:
        return

    channel = guild.get_channel(channel_id)
    if not channel:
        return

    await channel.send(embed=embed)

        # ----------------------------------------
# POINTS SYSTEM
# ----------------------------------------

def add_points(guild_id: int, user_id: int, amount: int):
    c.execute("""
        INSERT INTO points (guild_id, user_id, points)
        VALUES (?, ?, ?)
        ON CONFLICT(guild_id, user_id)
        DO UPDATE SET points = points + EXCLUDED.points
    """, (guild_id, user_id, amount))
    conn.commit()


def get_points(guild_id: int, user_id: int):
    c.execute(
        "SELECT points FROM points WHERE guild_id = ? AND user_id = ?",
        (guild_id, user_id)
    )
    row = c.fetchone()
    return row[0] if row else 0


def get_points(guild_id: int, user_id: int):
    c.execute("SELECT points FROM points WHERE guild_id = ? AND user_id = ?", (guild_id, user_id))
    row = c.fetchone()
    return row[0] if row else 0


# ----------------------------------------
# TITLES SYSTEM
# ----------------------------------------

def create_title(name: str, price: int):
    try:
        c.execute("""
            INSERT INTO titles (name, price)
            VALUES (?, ?)
        """, (name, price))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def list_titles():
    c.execute("SELECT id, name, price FROM titles ORDER BY price ASC")
    return c.fetchall()


# --------------------
# INVENTORY / EQUIP / FREEZE HELPERS
# --------------------

def add_title_to_user(guild_id: int, user_id: int, title_id: int):
    """Give title ownership to user (doesn't equip)."""
    c.execute("""
        INSERT OR IGNORE INTO user_titles_multi (guild_id, user_id, title_id)
        VALUES (?, ?, ?)
    """, (guild_id, user_id, title_id))
    conn.commit()

def user_owns_title(guild_id: int, user_id: int, title_id: int) -> bool:
    c.execute("""
        SELECT 1 FROM user_titles_multi WHERE guild_id = ? AND user_id = ? AND title_id = ?
    """, (guild_id, user_id, title_id))
    return c.fetchone() is not None

def get_user_titles(guild_id: int, user_id: int):
    """Return list of (title_id, title_name, price)."""
    c.execute("""
        SELECT t.id, t.name, t.price
        FROM user_titles_multi ut
        JOIN titles t ON ut.title_id = t.id
        WHERE ut.guild_id = ? AND ut.user_id = ?
        ORDER BY t.price ASC
    """, (guild_id, user_id))
    return c.fetchall()

def equip_title_db(guild_id: int, user_id: int, title_id: int):
    """Equip a title for user (one equipped per user)."""
    c.execute("""
        INSERT OR REPLACE INTO user_equipped_titles (guild_id, user_id, title_id)
        VALUES (?, ?, ?)
    """, (guild_id, user_id, title_id))
    conn.commit()

def get_equipped_title(guild_id: int, user_id: int):
    c.execute("""
        SELECT t.name FROM user_equipped_titles ue
        JOIN titles t ON ue.title_id = t.id
        WHERE ue.guild_id = ? AND ue.user_id = ?
    """, (guild_id, user_id))
    row = c.fetchone()
    return row[0] if row else None

# STREAK FREEZE HELPERS
def get_freeze_count(guild_id: int, user_id: int) -> int:
    c.execute("SELECT freezes FROM streak_freezes WHERE guild_id = ? AND user_id = ?", (guild_id, user_id))
    row = c.fetchone()
    return int(row[0]) if row else 0

def add_freezes(guild_id: int, user_id: int, amount: int):
    c.execute("""
        INSERT INTO streak_freezes (guild_id, user_id, freezes)
        VALUES (?, ?, ?)
        ON CONFLICT(guild_id, user_id)
        DO UPDATE SET freezes = freezes + excluded.freezes
    """, (guild_id, user_id, amount))
    conn.commit()

def set_freezes(guild_id: int, user_id: int, amount: int):
    c.execute("""
        INSERT INTO streak_freezes (guild_id, user_id, freezes)
        VALUES (?, ?, ?)
        ON CONFLICT(guild_id, user_id)
        DO UPDATE SET freezes = excluded.freezes
    """, (guild_id, user_id, amount))
    conn.commit()

def consume_freeze_if_available(guild_id: int, user_id: int) -> bool:
    """If freeze >0, decrement and return True, else False."""
    cnt = get_freeze_count(guild_id, user_id)
    if cnt and cnt > 0:
        c.execute("UPDATE streak_freezes SET freezes = freezes - 1 WHERE guild_id = ? AND user_id = ?", (guild_id, user_id))
        conn.commit()
        return True
    return False

def purchase_title(guild_id: int, user_id: int, title_id: int):
    # Get title price
    c.execute("SELECT price FROM titles WHERE id = ?", (title_id,))
    row = c.fetchone()
    if not row:
        return "NO_TITLE"

    price = row[0]
    user_pts = get_points(guild_id, user_id)

    if user_pts < price:
        return "NO_POINTS"

    # Deduct points
    c.execute("UPDATE points SET points = points - ? WHERE guild_id = ? AND user_id = ?",
              (price, guild_id, user_id))

    # Add to inventory (multi ownership)
    add_title_to_user(guild_id, user_id, title_id)

    conn.commit()
    return "OK"


# ----------------------------------------
# AUTO COLOR ROLE SYSTEM
# ----------------------------------------



# --------------------
# END OF QUADRANT 2
# --------------------
# ---------------------------
# arabic_bot_full.py - Part 3
# ---------------------------

# ============================================================
#                  SLASH COMMANDS START
# ============================================================


# ------------------------------------------------------------
# /struggle add  |  /struggle remove  |  /struggle list
# ------------------------------------------------------------

@tree.command(name="struggle_add", description="Add a struggle word.")
@app_commands.describe(word="The Arabic word", definition="The meaning/definition")
async def struggle_add(interaction: discord.Interaction, word: str, definition: str):
    guild_id = interaction.guild_id
    user_id = interaction.user.id

    ok = add_struggle_word(guild_id, user_id, word, definition)
    if not ok:
        await interaction.response.send_message(
            f"❌ You already added **{word}**.",
            ephemeral=True
        )
        return

    await interaction.response.send_message(
        f"✅ Added **{word}** → `{definition}` to your struggle list.",
        ephemeral=True
    )


@tree.command(name="struggle_remove", description="Remove a struggle word.")
@app_commands.describe(word="Word to remove")
async def struggle_remove(interaction: discord.Interaction, word: str):
    guild_id = interaction.guild_id
    user_id = interaction.user.id

    ok = remove_struggle_word(guild_id, user_id, word)
    if not ok:
        await interaction.response.send_message(
            f"❌ You don’t have **{word}** in your struggle words.",
            ephemeral=True
        )
        return

    await interaction.response.send_message(
        f"🗑️ Removed **{word}**.",
        ephemeral=True
    )


@tree.command(name="struggle_list", description="Show your struggle words.")
async def struggle_list(interaction: discord.Interaction):
    guild_id = interaction.guild_id
    user_id = interaction.user.id

    rows = get_user_struggle_words(guild_id, user_id)
    if not rows:
        await interaction.response.send_message(
            "You have **no** struggle words.",
            ephemeral=True
        )
        return

    # PATCH 9: Embed to avoid message length errors
    embed = discord.Embed(
        title="📘 Your Struggle Words",
        description="Here are your saved words and definitions:",
        color=discord.Color.blue()
    )

    for _, word, definition in rows:
        embed.add_field(
            name=word,
            value=definition if definition else "*No definition*",
            inline=False
        )

    await interaction.response.send_message(embed=embed, ephemeral=True)


# ------------------------------------------------------------
# /challenge — gives a random struggle word & waits for answer
# ------------------------------------------------------------

# active_challenges defined earlier at top-level (reused)
@tree.command(name="challenge", description="Get quizzed on one of your struggle words.")
async def challenge(interaction: discord.Interaction):
    guild_id = interaction.guild_id
    user = interaction.user
    user_id = user.id

    # Prevent multiple active challenges for same user
    async with _active_challenge_lock:
        if user_id in active_challenges:
            await interaction.response.send_message(
                "⏳ You already have an active challenge. Answer it or wait for it to expire.",
                ephemeral=True
            )
            return

        row = get_random_struggle_word(guild_id, user_id)
        if not row:
            await interaction.response.send_message(
                "❌ You have no struggle words. Add some first.",
                ephemeral=True
            )
            return

        struggle_id, word, definition = row
        expires = now_cst() + timedelta(seconds=CHALLENGE_TIMEOUT_SECONDS)
        active_challenges[user_id] = {
            "struggle_id": int(struggle_id),
            "definition": definition,
            "word": word,
            "expires_at": expires
        }

    # Immediate response so Discord doesn't mark "application did not respond"
    await interaction.response.send_message(
        f"🧠 **Challenge Time!**\n\nWhat is the definition of:\n\n👉 **{word}** ?\n\nType your answer in chat. You have {CHALLENGE_TIMEOUT_SECONDS} seconds.",
        ephemeral=False
    )




# ------------------------------------------------------------
# /points — see your points (PATCH 10)
# ------------------------------------------------------------

@tree.command(name="points", description="Check your points.")
async def points_cmd(interaction: discord.Interaction):
    guild_id = interaction.guild_id
    user_id = interaction.user.id

    # PATCH 10: thread-safe DB fetch to avoid race conditions
    pts = await asyncio.to_thread(get_points, guild_id, user_id)

    embed = discord.Embed(
        title="⭐ Your Points",
        description=f"You currently have **{pts}** points.",
        color=discord.Color.gold()
    )

    await interaction.response.send_message(embed=embed, ephemeral=True)



# ------------------------------------------------------------
# /shop — list titles
# ------------------------------------------------------------

@tree.command(name="shop", description="View the title shop.")
async def shop_cmd(interaction: discord.Interaction):
    rows = list_titles()
    if not rows:
        await interaction.response.send_message(
            "Shop is empty. Admins can add titles with `/add_title`.",
            ephemeral=True
        )
        return

    embed = discord.Embed(
        title="🏪 Title Shop",
        description="Available titles",
        color=0x00B2FF
    )

    for tid, name, price in rows:
        embed.add_field(
            name=f"[{name}] — {price} pts",
            value=f"id: `{tid}`",
            inline=False
        )

    await interaction.response.send_message(
        embed=embed,
        ephemeral=True
    )




# ------------------------------------------------------------
# ADMIN COMMAND — Add a title to the shop
# ------------------------------------------------------------
@tree.command(name="add_title", description="Admin: create a new title for the shop.")
@app_commands.describe(name="Title name", price="Point cost")
async def add_title_cmd(interaction: discord.Interaction, name: str, price: int):

    if interaction.user.id not in OWNER_IDS:
        await interaction.response.send_message("❌ No permission.", ephemeral=True)
        return

    ok = create_title(name, price)

    if not ok:
        await interaction.response.send_message("❌ Title already exists.", ephemeral=True)
        return

    await interaction.response.send_message(
        f"🏷️ **Title created!**\nName: **{name}**\nPrice: **{price} pts**",
        ephemeral=False
    )


# ------------------------------------------------------------
# ADMIN COMMAND — Set someone’s points
# ------------------------------------------------------------
@tree.command(name="set_points", description="Admin: set a user's points.")
@app_commands.describe(user="User to modify", amount="Point amount to set")
async def set_points_cmd(interaction: discord.Interaction, user: discord.Member, amount: int):

    if interaction.user.id not in OWNER_IDS:
        await interaction.response.send_message("❌ No permission.", ephemeral=True)
        return

    c.execute("""
        INSERT INTO points (guild_id, user_id, points)
        VALUES (?, ?, ?)
        ON CONFLICT(guild_id, user_id)
        DO UPDATE SET points = excluded.points
    """, (interaction.guild_id, user.id, amount))
    conn.commit()

    await interaction.response.send_message(
        f"✨ Set **{user.display_name}**'s points to **{amount}**.",
        ephemeral=False
    )


@tree.command(name="remove_title", description="Admin: remove a title from the shop.")
@app_commands.describe(title_id="The ID of the title to remove")
async def remove_title_cmd(interaction: discord.Interaction, title_id: int):

    if interaction.user.id not in OWNER_IDS:
        await interaction.response.send_message("❌ No permission.", ephemeral=True)
        return

    # Check existence
    c.execute("SELECT name FROM titles WHERE id = ?", (title_id,))
    row = c.fetchone()

    if not row:
        await interaction.response.send_message("❌ Title not found.", ephemeral=True)
        return

    # Delete title and all references
    c.execute("DELETE FROM titles WHERE id = ?", (title_id,))
    c.execute("DELETE FROM user_titles_multi WHERE title_id = ?", (title_id,))
    c.execute("DELETE FROM user_equipped_titles WHERE title_id = ?", (title_id,))
    conn.commit()

    await interaction.response.send_message(
        f"🗑️ Removed title **{row[0]}** (ID {title_id}) from the shop.",
        ephemeral=False
    )



# ------------------------------------------------------------
# ADMIN COMMAND — Set someone’s streak
# ------------------------------------------------------------
@tree.command(name="set_streak", description="Admin: set a user's streak.")
@app_commands.describe(user="User to modify", amount="New streak number")
async def set_streak_cmd(interaction: discord.Interaction, user: discord.Member, amount: int):

    if interaction.user.id not in OWNER_IDS:
        await interaction.response.send_message("❌ No permission.", ephemeral=True)
        return

    c.execute("""
        INSERT INTO streaks (guild_id, user_id, username, streak, last_done_date, total_completions)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(guild_id, user_id)
        DO UPDATE SET streak = excluded.streak
    """, (
        interaction.guild_id,
        user.id,
        str(user),
        amount,
        None,
        0
    ))
    conn.commit()

    await interaction.response.send_message(
        f"🔥 Set **{user.display_name}**'s streak to **{amount}**.",
        ephemeral=False
    )



# ------------------------------------------------------------
# /buy_title — purchase a title by ID
# ------------------------------------------------------------


@tree.command(name="buy_title", description="Purchase a title by ID.")
@app_commands.describe(title_id="ID from /shop")
async def buy_title(interaction: discord.Interaction, title_id: int):
    guild_id = interaction.guild_id
    user = interaction.user

    result = purchase_title(guild_id, user.id, title_id)

    if result == "NO_TITLE":
        await interaction.response.send_message("❌ Invalid title ID.", ephemeral=True)
        return

    if result == "NO_POINTS":
        await interaction.response.send_message("❌ You do not have enough points.", ephemeral=True)
        return

      # Success → add to inventory (no role changes)
    c.execute("SELECT name FROM titles WHERE id = ?", (title_id,))
    row = c.fetchone()
    if row:
        name = row[0]
    else:
        await interaction.response.send_message("⚠️ Title purchased but title record missing.", ephemeral=True)
        return

    # Add to user's inventory (purchase_title already did this, but ensure idempotent)
    add_title_to_user(guild_id, user.id, title_id)

    await interaction.response.send_message(
        f"🎉 You bought the title **[{name}]**! Use `/equiptitle {title_id}` to wear it.",
        ephemeral=False
    )

def get_display_name_with_title(guild_id: int, user: discord.Member):
    title_name = get_equipped_title(guild_id, user.id)
    if title_name:
        return f"[{title_name}] {user.display_name}"
    return user.display_name


# ------------------------------------------------------------
# /inventory — show owned titles & freezes (user optional)
# ------------------------------------------------------------
@tree.command(name="inventory", description="Show your inventory or another user's inventory.")
@app_commands.describe(user="Optional user to view")
async def inventory_cmd(interaction: discord.Interaction, user: Optional[discord.Member] = None):
    target = user or interaction.user
    guild_id = interaction.guild_id

    titles = get_user_titles(guild_id, target.id)
    equipped = get_equipped_title(guild_id, target.id)
    freezes = get_freeze_count(guild_id, target.id)

    embed = discord.Embed(
        title=f"📦 Inventory — {target.display_name}",
        color=0x00B2FF
    )

    embed.add_field(name="🧊 Streak Freezes", value=str(freezes), inline=False)

    if titles:
        lines = []
        for tid, name, price in titles:
            mark = " (equipped)" if equipped and name == equipped else ""
            lines.append(f"`{tid}` — **{name}** — {price} pts{mark}")
        embed.add_field(name="🏷️ Titles", value="\n".join(lines), inline=False)
    else:
        embed.add_field(name="🏷️ Titles", value="None", inline=False)

    await interaction.response.send_message(embed=embed, ephemeral=True)


# ------------------------------------------------------------
# /equiptitle — equip one of your owned titles
# ------------------------------------------------------------
@tree.command(name="equiptitle", description="Equip a title you own by ID.")
@app_commands.describe(title_id="ID from /shop or /inventory")
async def equip_title_cmd(interaction: discord.Interaction, title_id: int):
    guild_id = interaction.guild_id
    user = interaction.user

    # Check ownership
    if not user_owns_title(guild_id, user.id, title_id):
        await interaction.response.send_message("❌ You do not own that title.", ephemeral=True)
        return

    equip_title_db(guild_id, user.id, title_id)
    await interaction.response.send_message(f"✅ Equipped title id `{title_id}`.", ephemeral=True)


# ------------------------------------------------------------
# /buy_freeze — buy a streak freeze (cost 100 pts)
# ------------------------------------------------------------
@tree.command(name="buy_freeze", description="Buy a streak freeze (prevents a missed day from breaking streak).")
async def buy_freeze_cmd(interaction: discord.Interaction):
    guild_id = interaction.guild_id
    user = interaction.user

    cost = 100  # as requested

    pts = get_points(guild_id, user.id)
    if pts < cost:
        await interaction.response.send_message("❌ Not enough points to buy a streak freeze.", ephemeral=True)
        return

    # Deduct and add freeze
    add_points(guild_id, user.id, -cost)  # uses add_points which increments (we pass negative)
    add_freezes(guild_id, user.id, 1)

    await interaction.response.send_message(f"🧊 You bought a Streak Freeze for {cost} pts! Use is automatic if you miss a day.", ephemeral=True)


# ------------------------------------------------------------
# ADMIN: give_freeze & set_freeze
# ------------------------------------------------------------
@tree.command(name="give_freeze", description="Admin: give a streak freeze to a user.")
@app_commands.describe(user="User to modify", amount="How many to give")
async def give_freeze_cmd(interaction: discord.Interaction, user: discord.Member, amount: int):
    if interaction.user.id not in OWNER_IDS:
        await interaction.response.send_message("❌ No permission.", ephemeral=True)
        return

    add_freezes(interaction.guild_id, user.id, amount)
    await interaction.response.send_message(f"✅ Gave {amount} streak freeze(s) to {user.display_name}.", ephemeral=True)


@tree.command(name="set_freeze", description="Admin: set a user's streak freeze count.")
@app_commands.describe(user="User to modify", amount="New freeze amount")
async def set_freeze_cmd(interaction: discord.Interaction, user: discord.Member, amount: int):
    if interaction.user.id not in OWNER_IDS:
        await interaction.response.send_message("❌ No permission.", ephemeral=True)
        return

    set_freezes(interaction.guild_id, user.id, amount)
    await interaction.response.send_message(f"✅ Set {user.display_name}'s streak freezes to {amount}.", ephemeral=True)


# ------------------------------------------------------------
# /stats — Show your daily progress and title
# ------------------------------------------------------------
@tree.command(name="stats", description="View your streak, points, title, and freezes.")
@app_commands.describe(user="Optional user to view")
async def stats_cmd(interaction: discord.Interaction, user: Optional[discord.Member] = None):

    target = user or interaction.user
    guild_id = interaction.guild_id

    streak_info = get_user_streak(guild_id, target.id)
    points = get_points(guild_id, target.id)
    title = get_equipped_title(guild_id, target.id)
    freezes = get_freeze_count(guild_id, target.id)

    embed = discord.Embed(
        title=f"📊 Stats for {target.display_name}",
        color=0x00B2FF
    )

    embed.add_field(name="🔥 Streak", value=f"{streak_info['streak']} days", inline=True)
    embed.add_field(name="⭐ Points", value=f"{points}", inline=True)
    embed.add_field(name="📘 Total Completions", value=f"{streak_info['total']}", inline=True)

    if title:
        embed.add_field(name="🏅 Title", value=f"[{title}]", inline=False)
    else:
        embed.add_field(name="🏅 Title", value="None", inline=False)

    embed.add_field(name="🧊 Streak Freezes", value=str(freezes), inline=False)

    embed.set_footer(text="Keep studying! You got this 💪")

    await interaction.response.send_message(embed=embed, ephemeral=True)

# ------------------------------------------------------------
# /summary — Show who completed today (embed)
# ------------------------------------------------------------
@tree.command(name="summary", description="Show today's completions.")
async def summary_cmd(interaction: discord.Interaction):

    guild = interaction.guild
    rows = get_today_completions(guild.id)

    embed = discord.Embed(
        title="🌙 Daily Summary",
        description="Here’s who completed today:",
        color=0xFFD700
    )

    if rows:
        for username, t in rows:
            # Try to get live member to include title
            member = guild.get_member_named(username)
            if member:
                display = get_display_name_with_title(guild.id, member)
            else:
                display = username  # fallback: user left guild

            embed.add_field(
                name=display,
                value=f"Completed at {t}",
                inline=False
            )

    else:
        embed.add_field(
            name="Nobody completed today 😭",
            value="Try again tomorrow!",
            inline=False
        )

    await interaction.response.send_message(
        embed=embed,
        ephemeral=False
    )



# ------------------------------------------------------------
# /leaderboard — Streak ranking (Top 10)
# ------------------------------------------------------------
@tree.command(name="leaderboard", description="View the top streaks.")
async def leaderboard_cmd(interaction: discord.Interaction):

    guild = interaction.guild
    rows = get_leaderboard_streaks(interaction.guild_id, limit=10)

    embed = discord.Embed(
        title="🏆 Streak Leaderboard",
        color=0xFFA500
    )

    if not rows:
        embed.add_field(
            name="Empty",
            value="Nobody has streaks yet.",
            inline=False
        )
    else:
        rank = 1
        for username, streak, total in rows:

            # Try to resolve member by name
            member = guild.get_member_named(username)

            # If the user still exists in the guild → show their title
            if member:
                display = get_display_name_with_title(interaction.guild_id, member)
            else:
                # Fallback: username only (user left server)
                display = username

            embed.add_field(
                name=f"#{rank} — {display}",
                value=f"🔥 **{streak}** day streak\n📘 {total} total completions",
                inline=False
            )

            rank += 1

    await interaction.response.send_message(embed=embed, ephemeral=False)



# ------------------------------------------------------------
# /done — the main daily check-in
# ------------------------------------------------------------

@tree.command(name="done", description="Mark today as completed.")
async def done_cmd(interaction: discord.Interaction):
    guild = interaction.guild
    user = interaction.user

    date_str = today_cst_str()
    time_str = now_cst().strftime("%I:%M %p")

    # record_completion returns False if already done today
    success = record_completion(guild.id, user, date_str, time_str)

    name = get_display_name_with_title(interaction.guild_id, user)

    if not success:
        await interaction.response.send_message(
            f"❌ **{name}**, you already marked today as done.",
            ephemeral=True
        )
        return

    # Award daily points (ONLY on first completion of the day)
    add_points(guild.id, user.id, 10)

    await interaction.response.send_message(
        f"🔥 **{name}** marked today as DONE and earned **10 points!**",
        ephemeral=False
    )


# ------------------------------------------------------------
# /setchannel — bot reminders channel
# ------------------------------------------------------------

@tree.command(name="setchannel", description="Set the daily reminder channel.")
@app_commands.describe(channel="Channel to send reminders to")
async def setchannel(interaction: discord.Interaction, channel: discord.TextChannel):
    upsert_settings(interaction.guild_id, channel_id=channel.id)
    await interaction.response.send_message(
        f"📌 Reminder channel set to {channel.mention}.",
        ephemeral=True
    )


# ------------------------------------------------------------
# /setrole — set the role to ping
# ------------------------------------------------------------

@tree.command(name="setrole", description="Set the role that receives reminders.")
@app_commands.describe(role="Role to ping")
async def setrole(interaction: discord.Interaction, role: discord.Role):
    upsert_settings(interaction.guild_id, ping_role_id=role.id)
    await interaction.response.send_message(
        f"🔔 Ping role set to **{role.name}**.",
        ephemeral=True
    )


# ------------------------------------------------------------
# /toggleping — on/off for role pinging
# ------------------------------------------------------------

@tree.command(name="toggleping", description="Enable/disable role pinging.")
async def toggleping(interaction: discord.Interaction):
    s = get_settings(interaction.guild_id)
    new_val = not s["ping_enabled"]
    upsert_settings(interaction.guild_id, ping_enabled=new_val)

    await interaction.response.send_message(
        f"🔄 Ping role is now **{'ENABLED' if new_val else 'DISABLED'}**.",
        ephemeral=True
    )


# ============================================================
#             END OF QUADRANT 3 (SLASH COMMANDS)
# ============================================================
# ---------------------------
# arabic_bot_full.py - Part 4
# ---------------------------

# ============================================================
#                MESSAGE LISTENER FOR CHALLENGES
# ============================================================

@bot.event
async def on_message(message: discord.Message):
    # Ignore bot messages
    if message.author.bot:
        return

    if not message.guild:
        return

    user_id = message.author.id
    guild_id = message.guild.id

    chal = active_challenges.get(user_id)

    # Only respond if user has an active challenge
    if chal:

        # ------------------------------
        # Expired challenge
        # ------------------------------
        if now_cst() > chal["expires_at"]:
            async with _active_challenge_lock:
                active_challenges.pop(user_id, None)

            await message.channel.send(
                "⌛ Your challenge expired. Run `/challenge` to get a new one."
            )

        else:
            # ------------------------------
            # Evaluate user's answer
            # ------------------------------
            user_answer = message.content.strip()
            struggle_id = int(chal["struggle_id"])
            correct_definition = chal["definition"]

            try:
                score = await asyncio.to_thread(
                    evaluate_answer,
                    user_answer,
                    correct_definition,
                    struggle_id
                )
            except Exception:
                async with _active_challenge_lock:
                    active_challenges.pop(user_id, None)

                await message.channel.send(
                    "⚠️ Error evaluating your answer. Try `/challenge` again."
                )
                await bot.process_commands(message)
                return

            # Title display
            name = get_display_name_with_title(guild_id, message.author)

            # ------------------------------
            # Correct answer
            # ------------------------------
            if score >= CHALLENGE_SIM_THRESHOLD:

                add_points(guild_id, user_id, POINTS_PER_CORRECT)

                async with _active_challenge_lock:
                    active_challenges.pop(user_id, None)

                await message.channel.send(
                    f"🔥 **{name} — Correct!** You earned **5 points!**\n"
                    f"Similarity score: `{score:.2f}`"
                )

            # ------------------------------
            # Incorrect answer
            # ------------------------------
            else:
                # Safely pull expected definition
                async with _active_challenge_lock:
                    data = active_challenges.pop(user_id, None)

                if data:
                    expected = data["definition"]
                    word_txt = data.get("word", "")
                else:
                    expected = correct_definition
                    word_txt = chal.get("word", "")

                await message.channel.send(
                    f"❌ **{name}**, not quite!\n"
                    f"Similarity score: `{score:.2f}`\n"
                    f"Run `/challenge` to try another."
                )

    # Continue processing other commands
    await bot.process_commands(message)

# ============================================================
#                 DAILY REMINDER TASK (12 PM)
# ============================================================

@tasks.loop(seconds=30)
async def daily_reminder_task():
    now = datetime.now(cst)
    today = now.strftime("%Y-%m-%d")

    if now.hour == 12 and now.minute == 0:
        for guild in bot.guilds:
            settings = get_settings(guild.id)

            if settings.get("last_reminder_date") == today:
                continue

            await send_reminder_message(guild)
            upsert_settings(guild.id, last_reminder_date=today)


# ============================================================
#                  MIDNIGHT SUMMARY TASK (12 AM)
# ============================================================

@tasks.loop(seconds=30)
async def midnight_task():
    now = datetime.now(cst)
    today = now.strftime("%Y-%m-%d")

    if now.hour == 0 and now.minute == 0:
        for guild in bot.guilds:
            settings = get_settings(guild.id)

            if settings.get("last_summary_date") == today:
                continue

            await send_summary_embed(guild)
            reset_streaks_for_missed_yesterday(guild)

            upsert_settings(guild.id, last_summary_date=today)

# ============================================================
#                CHALLENGE SWEEPER (unchanged)
# ============================================================

@tasks.loop(seconds=10)
async def _challenge_sweeper():
    """Remove expired challenges from memory."""
    now = now_cst()
    to_remove = []

    async with _active_challenge_lock:
        for uid, chal in list(active_challenges.items()):
            if now > chal["expires_at"]:
                to_remove.append(uid)

        for uid in to_remove:
            active_challenges.pop(uid, None)



# ============================================================
#                          ON READY (FULLY PATCHED)
# ============================================================

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

    # PATCH 1: Pre-sync to avoid "Application did not respond"
    try:
        cmds = await tree.sync()
        print(f"Pre-synced {len(cmds)} commands successfully.")
    except Exception as e:
        print("Pre-sync error:", e)

    # Normal sync
    try:
        await tree.sync()
        print("Slash commands synced.")
    except Exception as e:
        print("Sync error:", e)

    # Start tasks
    try:
        if not daily_reminder_task.is_running():
            daily_reminder_task.start()

        if not midnight_task.is_running():
            midnight_task.start()

        if not _challenge_sweeper.is_running():
            _challenge_sweeper.start()

    except Exception as e:
        print("Task start error:", e)

    print("Bot is fully ready.")





# ============================================================
#                        BOOTSTRAP
# ============================================================

async def main():
    await bot.start(TOKEN)


if __name__ == "__main__":
    asyncio.run(main())























