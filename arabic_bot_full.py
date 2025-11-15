# ---------------------------
# arabic_bot_full.py - QUADRANT 1 (PART 1)
# ---------------------------

import os
import sqlite3
import asyncio
import random
from datetime import datetime, timedelta, date, time as dtime
from zoneinfo import ZoneInfo
from typing import Optional, List, Tuple, Dict, Any

import discord
from discord import app_commands, ui
from discord.ext import tasks, commands

# Embedding / ML - will be lazy-loaded to avoid blocking startup
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
TOKEN = os.getenv("DISCORD_TOKEN")
TIMEZONE = ZoneInfo("America/Chicago")
REMINDER_HOUR = 12  # 12:00 PM CST
REMINDER_MINUTE = 0

# Guild-specific behavior: keep as None for multi-guild operation
GUILD_ID = None  # 1438553047344353293  # optional

# Admin configuration:
# Prefer explicit OWNER_ID in env for secure admin actions. Fallback to usernames.
OWNER_ID = int(os.getenv("OWNER_ID")) if os.getenv("OWNER_ID") else None
FALLBACK_ADMIN_USERNAMES = ["baja1121", "justin", "baja"]  # case-insensitive match

# Scoring threshold (similarity) to consider correct:
CHALLENGE_SIMILARITY_THRESHOLD = 0.50

# Points awarded for correct answer
POINTS_FOR_CORRECT = 5

# Database file
DB = os.getenv("ARABIC_BOT_DB", "arabic_bot.db")
# ----------------------------------------

# ---------- Intents and Bot ----------
intents = discord.Intents.default()
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix="/", intents=intents)
tree = bot.tree

# ---------- Database connection ----------
conn = sqlite3.connect(DB, check_same_thread=False)
c = conn.cursor()

# --------------------
# TABLE CREATION
# --------------------
# Core tables
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

# Gamification / struggle tables
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
    color TEXT,
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

# New support tables to prevent repeated/recycled challenge behavior
c.execute("""
CREATE TABLE IF NOT EXISTS challenge_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER,
    user_id INTEGER,
    struggle_id INTEGER,
    attempted_at TEXT
)
""")

# Optionally: table for admin overrides / audit (simple)
c.execute("""
CREATE TABLE IF NOT EXISTS admin_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER,
    admin_user_id INTEGER,
    action TEXT,
    target_user_id INTEGER,
    details TEXT,
    performed_at TEXT
)
""")

conn.commit()

# --------------------
# EMBEDDING MODEL (LAZY LOAD)
# --------------------
_embedding_model = None

def get_embedding_model():
    """Lazy load the embedding model. Keep as a separate function so it can be loaded
    in a thread if needed."""
    global _embedding_model
    if _embedding_model is None:
        # Load small model once
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model

def embed_text_sync(text: str) -> np.ndarray:
    """Synchronous embedding generator — call in executor if used inside async code."""
    model = get_embedding_model()
    vec = model.encode(text)
    return np.asarray(vec, dtype=np.float32)

def embed_text_bytes_sync(text: str) -> bytes:
    arr = embed_text_sync(text)
    return arr.tobytes()

def get_embedding_array(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)

# --------------------
# BASIC HELPERS
# --------------------
def now_cst() -> datetime:
    return datetime.now(TIMEZONE)

def today_cst_str() -> str:
    return now_cst().date().isoformat()

def yesterday_cst_str() -> str:
    return (now_cst().date() - timedelta(days=1)).isoformat()

# ---- Settings helpers ----
def get_settings(guild_id: int):
    c.execute("SELECT channel_id, ping_role_id, ping_enabled, last_reminder_date FROM settings WHERE guild_id = ?", (guild_id,))
    row = c.fetchone()
    if row:
        return {"channel_id": row[0], "ping_role_id": row[1], "ping_enabled": bool(row[2]), "last_reminder_date": row[3]}
    return {"channel_id": None, "ping_role_id": None, "ping_enabled": False, "last_reminder_date": None}

def upsert_settings(guild_id: int, **kwargs):
    # ensure a settings row exists
    c.execute("INSERT OR IGNORE INTO settings (guild_id, channel_id, ping_role_id, ping_enabled, last_reminder_date) VALUES (?, ?, ?, ?, ?)",
              (guild_id, None, None, 0, None))
    if "channel_id" in kwargs:
        c.execute("UPDATE settings SET channel_id = ? WHERE guild_id = ?", (kwargs["channel_id"], guild_id))
    if "ping_role_id" in kwargs:
        c.execute("UPDATE settings SET ping_role_id = ? WHERE guild_id = ?", (kwargs["ping_role_id"], guild_id))
    if "ping_enabled" in kwargs:
        c.execute("UPDATE settings SET ping_enabled = ? WHERE guild_id = ?", (1 if kwargs["ping_enabled"] else 0, guild_id))
    if "last_reminder_date" in kwargs:
        c.execute("UPDATE settings SET last_reminder_date = ? WHERE guild_id = ?", (kwargs["last_reminder_date"], guild_id))
    conn.commit()

# ---- Completions & streak helpers ----
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
            # last_done_date stored as ISO date string or None
            try:
                last_iso = date.fromisoformat(last_done_date) if last_done_date else None
            except Exception:
                last_iso = None

            if last_iso and last_iso == (date.fromisoformat(date_str) - timedelta(days=1)):
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
            c.execute("INSERT OR IGNORE INTO streaks (guild_id, user_id, username, streak, last_done_date, total_completions) VALUES (?, ?, ?, ?, ?, ?)",
                      (guild.id, m.id, str(m), 0, None, 0))
            c.execute("UPDATE streaks SET streak = 0 WHERE guild_id = ? AND user_id = ?", (guild.id, m.id))
            missed.append(m)
    conn.commit()
    return missed

# --------------------
# ADMIN CHECK HELPERS
# --------------------
def is_admin_user(member: discord.Member) -> bool:
    # 1) explicit owner ID from environment
    if OWNER_ID and member.id == OWNER_ID:
        return True

    # 2) check display name / username fallback (case-insensitive)
    uname = str(member).split('#')[0].lower()
    if any(p.lower() == uname for p in FALLBACK_ADMIN_USERNAMES):
        return True

    # 3) check for role named 'ArabAdmin' or 'Admin' (common)
    for role in member.roles:
        if role.name.lower() in ("arabadmin", "arabicadmin", "admin", "moderator", "mod"):
            return True

    # 4) guild owner
    if hasattr(member, "guild") and member.guild is not None and member.guild.owner_id == member.id:
        return True

    return False

# --------------------
# POINTS & TITLES HELPERS (declarations only — implementations in Quadrant 2)
# --------------------
def add_points(guild_id: int, user_id: int, amount: int):
    c.execute("""
        INSERT INTO points (guild_id, user_id, points)
        VALUES (?, ?, ?)
        ON CONFLICT(guild_id, user_id)
        DO UPDATE SET points = points + EXCLUDED.points
    """, (guild_id, user_id, amount))
    conn.commit()

def get_points(guild_id: int, user_id: int) -> int:
    c.execute("SELECT points FROM points WHERE guild_id = ? AND user_id = ?", (guild_id, user_id))
    row = c.fetchone()
    return row[0] if row else 0

# --------------------
# STRUGGLE / CHALLENGE DB HELPERS (declarations only — implementations in Quadrant 2)
# --------------------
def add_struggle_word_db(guild_id: int, user_id: int, word: str, definition: str) -> bool:
    """Add to DB and returns True if inserted, False if already exists."""
    try:
        c.execute("""
            INSERT INTO struggle_words (guild_id, user_id, word, definition)
            VALUES (?, ?, ?, ?)
        """, (guild_id, user_id, word.lower(), definition.lower()))
        conn.commit()
        struggle_id = c.lastrowid
        # store embedding synchronized helper here to avoid duplication (will be called from Quadrant 2 via async thread)
        return True
    except sqlite3.IntegrityError:
        return False

def remove_struggle_word_db(guild_id: int, user_id: int, word: str) -> bool:
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

def get_user_struggle_words_db(guild_id: int, user_id: int):
    c.execute("""
        SELECT id, word, definition FROM struggle_words
        WHERE guild_id = ? AND user_id = ?
    """, (guild_id, user_id))
    return c.fetchall()

def get_random_struggle_word_db(guild_id: int, user_id: int):
    rows = get_user_struggle_words_db(guild_id, user_id)
    return random.choice(rows) if rows else None

# --------------------
# END OF QUADRANT 1
# --------------------
# ---------------------------
# arabic_bot_full.py - QUADRANT 2 (PART 2)
# ---------------------------

import math
from typing import Optional

# NOTE: this file continues from Quadrant 1 where DB connection and
# helper quick-declarations were created.

# --------------------
# EMBEDDING HELPERS (async-safe)
# --------------------

async def embed_text_async(text: str) -> np.ndarray:
    """Return embedding vector (numpy array) without blocking the event loop."""
    loop = asyncio.get_running_loop()
    arr = await loop.run_in_executor(None, embed_text_sync, text)
    return arr

async def embed_text_bytes_async(text: str) -> bytes:
    loop = asyncio.get_running_loop()
    b = await loop.run_in_executor(None, embed_text_bytes_sync, text)
    return b

# --------------------
# STRUGGLE-WORD OPERATIONS (full implementations)
# --------------------

async def add_struggle_word(guild_id: int, user_id: int, word: str, definition: str) -> bool:
    """
    Add a struggle word and store its embedding. Returns True if added, False if already exists.
    This function runs the embedding generation in a thread to avoid blocking.
    """
    inserted = add_struggle_word_db(guild_id, user_id, word, definition)
    if not inserted:
        return False

    # fetch the inserted row id for the struggle (the DB helper returned True but didn't expose id)
    # we'll find it now
    c.execute("SELECT id FROM struggle_words WHERE guild_id = ? AND user_id = ? AND word = ?", (guild_id, user_id, word.lower()))
    row = c.fetchone()
    if not row:
        return False
    struggle_id = row[0]

    # generate embedding off-thread
    emb_bytes = await embed_text_bytes_async(definition.lower())

    # store embedding blob
    c.execute("INSERT INTO struggle_embeddings (struggle_id, embedding) VALUES (?, ?)", (struggle_id, emb_bytes))
    conn.commit()
    return True

async def remove_struggle_word(guild_id: int, user_id: int, word: str) -> bool:
    """Remove a struggle word and its embedding."""
    return remove_struggle_word_db(guild_id, user_id, word)

def get_user_struggle_words(guild_id: int, user_id: int):
    return get_user_struggle_words_db(guild_id, user_id)

def get_random_struggle_word(guild_id: int, user_id: int):
    # Improved selection: avoid recently-used struggle IDs for this user
    rows = get_user_struggle_words_db(guild_id, user_id)
    if not rows:
        return None

    # load last 5 attempted struggle_ids for this user
    c.execute("""
        SELECT struggle_id FROM challenge_history
        WHERE guild_id = ? AND user_id = ?
        ORDER BY attempted_at DESC
        LIMIT 5
    """, (guild_id, user_id))
    recent = {r[0] for r in c.fetchall()}

    candidates = [r for r in rows if r[0] not in recent]
    if not candidates:
        # all are recent — fallback to all rows but shuffle
        candidates = rows

    return random.choice(candidates)

async def evaluate_answer_async(user_answer: str, struggle_id: int) -> float:
    """
    Compute similarity between stored embedding for struggle_id and user's answer.
    Returns similarity float in [0,1] (or possibly small negative rounding noise).
    Runs embedding generation off-thread.
    """
    # fetch stored embedding blob
    c.execute("SELECT embedding FROM struggle_embeddings WHERE struggle_id = ?", (struggle_id,))
    row = c.fetchone()
    if not row or not row[0]:
        return 0.0

    stored_blob = row[0]
    stored_vec = get_embedding_array(stored_blob)

    # compute user vector (off-thread)
    loop = asyncio.get_running_loop()
    user_vec = await loop.run_in_executor(None, embed_text_sync, user_answer.lower())
    user_vec = np.asarray(user_vec, dtype=np.float32)

    # compute cosine similarity (safe)
    # handle zero vectors defensively
    try:
        sim = cosine_similarity([stored_vec], [user_vec])[0][0]
        if math.isnan(sim):
            return 0.0
        return float(sim)
    except Exception:
        # fallback: manual dot / norm
        denom = (np.linalg.norm(stored_vec) * np.linalg.norm(user_vec))
        if denom == 0:
            return 0.0
        return float(np.dot(stored_vec, user_vec) / denom)

# --------------------
# CHALLENGE HISTORY HELPERS
# --------------------

def log_challenge_attempt(guild_id: int, user_id: int, struggle_id: int):
    """Record that a user attempted a given struggle word (for repetition avoidance)."""
    c.execute("INSERT INTO challenge_history (guild_id, user_id, struggle_id, attempted_at) VALUES (?, ?, ?, ?)",
              (guild_id, user_id, struggle_id, datetime.utcnow().isoformat()))
    conn.commit()

def get_recent_attempts_count(guild_id: int, user_id: int, struggle_id: int, days: int = 7) -> int:
    """Count attempts for a given struggle word in last `days` days."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    c.execute("""
        SELECT COUNT(*) FROM challenge_history
        WHERE guild_id = ? AND user_id = ? AND struggle_id = ? AND attempted_at >= ?
    """, (guild_id, user_id, struggle_id, cutoff))
    row = c.fetchone()
    return row[0] if row else 0

# --------------------
# POINTS & TITLES IMPLEMENTATIONS
# --------------------

def create_title(name: str, color: str, price: int) -> bool:
    """Create a title in the shop. Returns False if already exists."""
    try:
        c.execute("INSERT INTO titles (name, color, price) VALUES (?, ?, ?)", (name, color, price))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def list_titles():
    c.execute("SELECT id, name, color, price FROM titles ORDER BY price ASC")
    return c.fetchall()

def get_user_title(guild_id: int, user_id: int):
    c.execute("""
        SELECT titles.name, titles.color FROM user_titles
        JOIN titles ON user_titles.title_id = titles.id
        WHERE guild_id = ? AND user_id = ?
    """, (guild_id, user_id))
    return c.fetchone()

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

    # Deduct points (ensure row exists)
    c.execute("INSERT OR IGNORE INTO points (guild_id, user_id, points) VALUES (?, ?, ?)", (guild_id, user_id, user_pts))
    c.execute("UPDATE points SET points = points - ? WHERE guild_id = ? AND user_id = ?", (price, guild_id, user_id))

    # Assign title
    c.execute("""
        INSERT OR REPLACE INTO user_titles (guild_id, user_id, title_id)
        VALUES (?, ?, ?)
    """, (guild_id, user_id, title_id))

    conn.commit()
    return "OK"

# --------------------
# APPLY TITLE COLOR (async-safe)
# --------------------

async def apply_title_color(member: discord.Member, color_hex: str, title_name: str):
    """
    Creates/gets a role with the correct title & color and applies it to the user.
    This is async because it calls Discord API.
    """
    guild = member.guild
    if guild is None:
        return

    # sanitize color hex
    try:
        color_int = int(color_hex.lstrip("#"), 16)
    except Exception:
        color_int = 0x000000

    # role name uses brackets to identify title roles
    role_name = f"[{title_name}]"

    # attempt to find existing role (case-sensitive)
    role = discord.utils.get(guild.roles, name=role_name)
    if role is None:
        # create role
        try:
            role = await guild.create_role(
                name=role_name,
                colour=discord.Colour(color_int),
                reason="Title purchase/assignment"
            )
        except Exception:
            # creation failed (permissions?) — try to find role by similar name
            role = discord.utils.get(guild.roles, name=role_name)

    # Remove old title roles prefixed with '[' and suffixed with ']'
    to_remove = [r for r in member.roles if r.name.startswith("[") and r.name.endswith("]")]
    try:
        if to_remove:
            await member.remove_roles(*to_remove, reason="Replacing title role")
    except Exception:
        # ignore failures to remove due to permissions
        pass

    # Add new title role
    try:
        await member.add_roles(role, reason="Assigning title role")
    except Exception:
        # ignore permission failures
        pass

# --------------------
# UTILITY: SEED SOME DEFAULT TITLES (callable by admin)
# --------------------

def seed_default_titles():
    defaults = [
        ("Beginner", "#9AA0A6", 10),
        ("Learner", "#6BBF59", 25),
        ("Scholar", "#3B82F6", 75),
        ("Master", "#8B5CF6", 200),
    ]
    for name, color, price in defaults:
        try:
            c.execute("INSERT OR IGNORE INTO titles (name, color, price) VALUES (?, ?, ?)", (name, color, price))
        except Exception:
            pass
    conn.commit()

# --------------------
# END OF QUADRANT 2
# --------------------
# ---------------------------
# arabic_bot_full.py - QUADRANT 3 (PART 3)
# ---------------------------

from discord import app_commands
from discord.app_commands import Choice
from typing import Dict
import time as _time

# In-memory active challenges
# key: user_id -> value: dict with keys: struggle_id, word, definition, issued_at
active_challenges: Dict[int, Dict[str, Any]] = {}

# Time (seconds) after which a challenge auto-expires (optional safety)
CHALLENGE_TTL_SECONDS = 300  # 5 minutes

# ------------------------------------------------------------
# /struggle_add, /struggle_remove, /struggle_list, /struggle_clear
# ------------------------------------------------------------

@tree.command(name="struggle_add", description="Add a struggle word.")
@app_commands.describe(word="The Arabic word", definition="The meaning/definition")
async def struggle_add(interaction: discord.Interaction, word: str, definition: str):
    # embed generation may take time — defer ephemeral response quickly
    await interaction.response.defer(ephemeral=True, thinking=False)
    guild_id = interaction.guild_id
    user_id = interaction.user.id

    try:
        ok = await add_struggle_word(guild_id, user_id, word, definition)
    except Exception as e:
        # ensure we send a response if something unexpected happens
        await interaction.followup.send(f"❌ Failed to add word: `{e}`", ephemeral=True)
        return

    if not ok:
        await interaction.followup.send(f"❌ You already added **{word}**.", ephemeral=True)
        return

    await interaction.followup.send(f"✅ Added **{word}** → `{definition}` to your struggle list.", ephemeral=True)


@tree.command(name="struggle_remove", description="Remove a struggle word.")
@app_commands.describe(word="Word to remove")
async def struggle_remove(interaction: discord.Interaction, word: str):
    await interaction.response.defer(ephemeral=True, thinking=False)
    guild_id = interaction.guild_id
    user_id = interaction.user.id

    ok = await remove_struggle_word(guild_id, user_id, word)
    if not ok:
        await interaction.followup.send(f"❌ You don’t have **{word}** in your struggle words.", ephemeral=True)
        return

    await interaction.followup.send(f"🗑️ Removed **{word}**.", ephemeral=True)


@tree.command(name="struggle_list", description="Show your struggle words.")
async def struggle_list(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=False)
    guild_id = interaction.guild_id
    user_id = interaction.user.id

    rows = get_user_struggle_words(guild_id, user_id)
    if not rows:
        await interaction.followup.send("You have **no** struggle words.", ephemeral=True)
        return

    msg = "**Your Struggle Words:**\n\n"
    for _, word, definition in rows:
        msg += f"• **{word}** → `{definition}`\n"

    await interaction.followup.send(msg, ephemeral=True)


@tree.command(name="struggle_clear", description="Clear all your struggle words.")
async def struggle_clear(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=False)
    guild_id = interaction.guild_id
    user_id = interaction.user.id

    rows = get_user_struggle_words_db(guild_id, user_id)
    if not rows:
        await interaction.followup.send("You have **no** struggle words to clear.", ephemeral=True)
        return

    # Remove each
    removed = 0
    for r in rows:
        wid = r[0]
        word = r[1]
        if remove_struggle_word_db(guild_id, user_id, word):
            removed += 1

    await interaction.followup.send(f"🧹 Cleared **{removed}** struggle words.", ephemeral=True)


# ------------------------------------------------------------
# /challenge — improved behavior:
#  - Only accepts answers from challenger
#  - When wrong once: close challenge and show answer
#  - When correct: award points and close
#  - Avoids recently-served words
# ------------------------------------------------------------

@tree.command(name="challenge", description="Get quizzed on one of your struggle words.")
async def challenge(interaction: discord.Interaction):
    # quick defer so Discord sees response started
    await interaction.response.defer(ephemeral=False, thinking=False)
    guild_id = interaction.guild_id
    user = interaction.user
    user_id = user.id

    # expire old active challenge for user if any (safety)
    if user_id in active_challenges:
        entry = active_challenges[user_id]
        issued = entry.get("issued_at", 0)
        if _time.time() - issued < CHALLENGE_TTL_SECONDS:
            await interaction.followup.send("You already have an active challenge. Answer that one or wait for it to expire.", ephemeral=False)
            return
        else:
            del active_challenges[user_id]

    # fetch a random struggle word (already avoids recently used ones)
    row = get_random_struggle_word_db(guild_id, user_id)
    if not row:
        await interaction.followup.send("❌ You have no struggle words. Add some first using /struggle_add.", ephemeral=True)
        return

    struggle_id, word, definition = row

    # record that we served this struggle (so we avoid repeating too much)
    try:
        log_challenge_attempt(guild_id, user_id, struggle_id)
    except Exception:
        pass

    # create active challenge entry
    active_challenges[user_id] = {
        "struggle_id": struggle_id,
        "word": word,
        "definition": definition,
        "issued_at": _time.time()
    }

    # Public prompt (non-ephemeral) so user can answer in channel — only their messages count
    await interaction.followup.send(
        f"🧠 **Challenge Time!**\n\nWhat is the definition of:\n\n👉 **{word}** ?\n\n*(Only {user.display_name}'s answers will be accepted for this challenge — you have one attempt.)*",
        ephemeral=False
    )


# ------------------------------------------------------------
# /points — check your points
# ------------------------------------------------------------

@tree.command(name="points", description="Check your points.")
async def points_cmd(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=False)
    guild_id = interaction.guild_id
    user_id = interaction.user.id

    pts = get_points(guild_id, user_id)
    await interaction.followup.send(f"⭐ You have **{pts}** points.", ephemeral=True)


# ------------------------------------------------------------
# /score — view someone else's points
# ------------------------------------------------------------

@tree.command(name="score", description="See someone's points. If omitted, shows your points.")
@app_commands.describe(member="Member to view (optional)")
async def score(interaction: discord.Interaction, member: Optional[discord.Member] = None):
    await interaction.response.defer(ephemeral=True, thinking=False)
    guild_id = interaction.guild_id
    if member is None:
        member = interaction.user

    pts = get_points(guild_id, member.id)
    await interaction.followup.send(f"⭐ **{member.display_name}** has **{pts}** points.", ephemeral=True)


# ------------------------------------------------------------
# /shop and /buy_title
# ------------------------------------------------------------

@tree.command(name="shop", description="View the title shop.")
async def shop_cmd(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=False)
    rows = list_titles()
    if not rows:
        await interaction.followup.send("Shop is empty.", ephemeral=True)
        return

    msg = "**🏪 TITLE SHOP:**\n\n"
    for tid, name, color, price in rows:
        msg += f"• **[{name}]** — `{color}` — **{price} pts** (id: `{tid}`)\n"

    await interaction.followup.send(msg, ephemeral=True)


@tree.command(name="buy_title", description="Purchase a title by ID.")
@app_commands.describe(title_id="ID from /shop")
async def buy_title(interaction: discord.Interaction, title_id: int):
    await interaction.response.defer(ephemeral=False, thinking=False)
    guild_id = interaction.guild_id
    user = interaction.user

    result = purchase_title(guild_id, user.id, title_id)

    if result == "NO_TITLE":
        await interaction.followup.send("❌ Invalid title ID.", ephemeral=True)
        return

    if result == "NO_POINTS":
        await interaction.followup.send("❌ You do not have enough points.", ephemeral=True)
        return

    # Success → apply role
    c.execute("SELECT name, color FROM titles WHERE id = ?", (title_id,))
    row = c.fetchone()
    if not row:
        await interaction.followup.send("⚠️ Title purchased but record missing.", ephemeral=True)
        return

    name, color = row
    member = interaction.guild.get_member(user.id)
    try:
        await apply_title_color(member, color, name)
    except Exception:
        await interaction.followup.send("⚠️ Title purchased but color assignment failed (permissions).", ephemeral=True)
        return

    await interaction.followup.send(f"🎉 You bought the title **[{name}]**!", ephemeral=False)


# ------------------------------------------------------------
# ADMIN COMMANDS: setpoints, assign_title, seed_titles
# ------------------------------------------------------------

def require_admin(member: discord.Member):
    if not is_admin_user(member):
        raise app_commands.AppCommandError("Not permitted")

@tree.command(name="setpoints", description="(Admin) Set a member's points to a value.")
@app_commands.describe(member="Member to modify", points="New points value")
async def setpoints(interaction: discord.Interaction, member: discord.Member, points: int):
    # require admin early
    try:
        require_admin(interaction.user)
    except app_commands.AppCommandError:
        await interaction.response.send_message("❌ You are not authorized to use this command.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True, thinking=False)
    guild_id = interaction.guild_id
    # Ensure points row exists, then set
    c.execute("INSERT OR REPLACE INTO points (guild_id, user_id, points) VALUES (?, ?, ?)", (guild_id, member.id, points))
    conn.commit()

    # log admin action
    c.execute("INSERT INTO admin_actions (guild_id, admin_user_id, action, target_user_id, details, performed_at) VALUES (?, ?, ?, ?, ?, ?)",
              (guild_id, interaction.user.id, "setpoints", member.id, f"points={points}", datetime.utcnow().isoformat()))
    conn.commit()

    await interaction.followup.send(f"✅ Set **{member.display_name}**'s points to **{points}**.", ephemeral=True)


@tree.command(name="assign_title", description="(Admin) Assign a title to a member by title ID.")
@app_commands.describe(member="Member", title_id="Title ID")
async def assign_title(interaction: discord.Interaction, member: discord.Member, title_id: int):
    try:
        require_admin(interaction.user)
    except app_commands.AppCommandError:
        await interaction.response.send_message("❌ You are not authorized to use this command.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True, thinking=False)
    guild_id = interaction.guild_id

    # Check title exists
    c.execute("SELECT id, name, color FROM titles WHERE id = ?", (title_id,))
    row = c.fetchone()
    if not row:
        await interaction.followup.send("❌ Invalid title ID.", ephemeral=True)
        return

    # Assign in DB
    c.execute("INSERT OR REPLACE INTO user_titles (guild_id, user_id, title_id) VALUES (?, ?, ?)", (guild_id, member.id, title_id))
    conn.commit()

    # apply role asynchronously
    _, name, color = row
    try:
        await apply_title_color(member, color, name)
    except Exception:
        pass

    # log admin action
    c.execute("INSERT INTO admin_actions (guild_id, admin_user_id, action, target_user_id, details, performed_at) VALUES (?, ?, ?, ?, ?, ?)",
              (guild_id, interaction.user.id, "assign_title", member.id, f"title_id={title_id}", datetime.utcnow().isoformat()))
    conn.commit()

    await interaction.followup.send(f"✅ Assigned **[{name}]** to **{member.display_name}**.", ephemeral=True)


@tree.command(name="seed_titles", description="(Admin) Seed default titles into the shop.")
async def cmd_seed_titles(interaction: discord.Interaction):
    try:
        require_admin(interaction.user)
    except app_commands.AppCommandError:
        await interaction.response.send_message("❌ You are not authorized to use this command.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True, thinking=False)
    seed_default_titles()
    await interaction.followup.send("✅ Default titles seeded.", ephemeral=True)


# ------------------------------------------------------------
# Helpful guard: sync commands when ready (on_ready in Quadrant 4 will call tree.sync())
# ------------------------------------------------------------

# --------------------
# END OF QUADRANT 3
# --------------------
# allow command processing first for slash commands etc.
try:
    user_id = message.author.id
    guild_id = message.guild.id if message.guild else None
except Exception:
    await bot.process_commands(message)
    return

# If there is an active challenge for this user, only accept their message
if user_id in active_challenges:
    chal = active_challenges.get(user_id)
    struggle_id = chal.get("struggle_id")
    correct_definition = chal.get("definition")
    word = chal.get("word")

    # We'll only accept the first non-empty message as the one attempt
    user_answer = message.content.strip()
    if not user_answer:
        # ignore empty content messages
        await bot.process_commands(message)
        return

    # Compute similarity off the event loop
    try:
        sim = await evaluate_answer_async(user_answer, struggle_id)
    except Exception as e:
        # In case of embedding errors, inform user and close the challenge to avoid stuck state
        try:
            await message.channel.send("⚠️ An error occurred while scoring your answer. Please try /challenge again.")
        except Exception:
            pass
        # remove active challenge
        try:
            del active_challenges[user_id]
        except KeyError:
            pass
        await bot.process_commands(message)
        return

    # Mark attempt in history (already logged when issued, but log the attempt too)
    try:
        log_challenge_attempt(guild_id, user_id, struggle_id)
    except Exception:
        pass

    # Check threshold
    if sim >= CHALLENGE_SIMILARITY_THRESHOLD:
        # correct
        try:
            add_points(guild_id, user_id, POINTS_FOR_CORRECT)
        except Exception:
            pass

        # remove active challenge
        try:
            del active_challenges[user_id]
        except KeyError:
            pass

        # Inform user
        try:
            await message.channel.send(
                f"🔥 **Correct!** You earned **{POINTS_FOR_CORRECT} points** for **{word}**.\n"
                f"Similarity score: `{sim:.2f}`"
            )
        except Exception:
            pass
    else:
        # wrong — reveal and close challenge immediately
        try:
            await message.channel.send(
                f"❌ Not quite. The correct definition was:\n\n`{correct_definition}`\n\n"
                f"Similarity score: `{sim:.2f}`\n\n"
                f"To try again, run `/challenge` (you will get a fresh attempt)."
            )
        except Exception:
            pass

        # remove active challenge
        try:
            del active_challenges[user_id]
        except KeyError:
            pass

    # Do not process this message as a command further (we handled it)
    return

# If no active challenge or message isn't relevant, allow commands to be processed
await bot.process_commands(message)

