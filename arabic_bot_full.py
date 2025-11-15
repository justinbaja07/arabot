# ---------------------------
# arabic_bot_full.py - QUADRANT 1
# (Imports, config, DB setup, model load, core helpers)
# ---------------------------

import os
import sqlite3
import asyncio
import random
from datetime import datetime, timedelta, date, time as dtime
from zoneinfo import ZoneInfo
from typing import Optional, List, Tuple, Dict

import discord
from discord import app_commands, ui
from discord.ext import tasks, commands

# Embedding / ML
# sentence-transformers requires huggingface_hub pinned in requirements.txt (we handled that earlier)
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
TOKEN = os.getenv("DISCORD_TOKEN")
TIMEZONE = ZoneInfo("America/Chicago")
REMINDER_HOUR = 12  # 12:00 PM CST
REMINDER_MINUTE = 0
# set to your guild id or None to run across all guilds
GUILD_ID = 1438553047344353293

# List of admin usernames (exact Discord usernames or IDs you want to treat as bot admins)
# You can add strings like "baja1121" or numeric user IDs. We'll accept either.
BOT_ADMINS = {"baja1121", "justin"}  # modify as needed; membership/ID checks handled later

# ----------------------------------------

# ---------- Intents and Bot ----------
intents = discord.Intents.default()
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix="/", intents=intents)
tree = bot.tree

# ---------- Database connection ----------
DB = "arabic_bot.db"
conn = sqlite3.connect(DB, check_same_thread=False)
c = conn.cursor()

# --------------------
# TABLE CREATION
# --------------------
# Core tables (existing behavior preserved)
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

# Gamification tables (expanded/fixed)
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

conn.commit()

# --------------------
# PRE-SEED DEFAULT TITLES (only inserts if not present)
# --------------------
def _ensure_default_titles():
    default_titles = [
        ("THE GOAT", "#FFD700", 200),
        ("THE ARAB", "#FF4500", 300),
        ("DESERT DEMON", "#FF8C00", 250),
        ("WORD WARRIOR", "#1E90FF", 150),
        ("ARABIC OVERLORD", "#8A2BE2", 500),
    ]
    for name, color, price in default_titles:
        c.execute("INSERT OR IGNORE INTO titles (name, color, price) VALUES (?, ?, ?)", (name, color, price))
    conn.commit()

_ensure_default_titles()

# --------------------
# EMBEDDING MODEL LOAD & IN-MEM CACHE
# --------------------
# Load the sentence-transformers model once. This increases cold-start but makes scoring fast.
# We also keep an in-memory cache of embeddings for struggle words to avoid repeated decode/encode.
try:
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    # If for some reason model fails at import time, we still want the bot to start (degraded).
    embedding_model = None
    print("Warning: embedding model failed to load at import time:", e)

# In-memory cache: struggle_id -> numpy array (float32)
EMBED_CACHE: Dict[int, np.ndarray] = {}

# --------------------
# BASIC HELPERS
# --------------------
def now_cst() -> datetime:
    return datetime.now(TIMEZONE)

def today_cst_str() -> str:
    return now_cst().date().isoformat()

def yesterday_cst_str() -> str:
    return (now_cst().date() - timedelta(days=1)).isoformat()

def is_bot_admin(member: discord.Member) -> bool:
    """
    Check if a member is considered a bot admin.
    Allows exact username matches from BOT_ADMINS or Discord perms (administrator).
    """
    if member is None:
        return False
    # direct ID match if someone put numeric IDs in BOT_ADMINS
    try:
        if str(member.id) in BOT_ADMINS:
            return True
    except Exception:
        pass
    # username (without discriminator) match
    if getattr(member, "name", "").lower() in {s.lower() for s in BOT_ADMINS}:
        return True
    # server administrator privilege fallback
    try:
        return member.guild_permissions.administrator
    except Exception:
        return False

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
# arabic_bot_full.py - QUADRANT 2
# (Struggle helpers, embeddings cache, points, titles, role color helper)
# ---------------------------

# --------------------
# EMBEDDING UTILITIES & CACHE MANAGEMENT
# --------------------
def _vec_to_blob(vec: np.ndarray) -> bytes:
    """Convert float32 numpy vector to bytes for storage."""
    return vec.astype(np.float32).tobytes()

def _blob_to_vec(blob: bytes) -> np.ndarray:
    """Convert stored bytes back to numpy float32 vector."""
    return np.frombuffer(blob, dtype=np.float32)

def embed_text_to_array(text: str) -> np.ndarray:
    """Return a float32 numpy array embedding for text. If model missing, returns None."""
    if embedding_model is None:
        return None
    vec = embedding_model.encode(text, convert_to_numpy=True)
    return np.asarray(vec, dtype=np.float32)

def preload_embeddings_into_cache():
    """
    Load all stored struggle_embeddings into the in-memory cache EMBED_CACHE.
    Called at startup from on_ready to avoid DB round-trips during scoring.
    """
    EMBED_CACHE.clear()
    c.execute("SELECT struggle_id, embedding FROM struggle_embeddings")
    for sid, blob in c.fetchall():
        try:
            EMBED_CACHE[sid] = _blob_to_vec(blob)
        except Exception:
            # skip malformed rows
            continue

# --------------------
# STRUGGLE WORD CRUD (DB + CACHE)
# --------------------
# Tracks last chosen struggle id per (guild, user) to avoid repeating
LAST_CHOSEN: Dict[Tuple[int, int], int] = {}

def add_struggle_word(guild_id: int, user_id: int, word: str, definition: str) -> bool:
    """
    Adds a struggle word and stores its embedding (if model available).
    Returns True on success, False if duplicate.
    """
    word_l = word.strip().lower()
    def_l = definition.strip()
    try:
        c.execute("""
            INSERT INTO struggle_words (guild_id, user_id, word, definition)
            VALUES (?, ?, ?, ?)
        """, (guild_id, user_id, word_l, def_l))
        conn.commit()
    except sqlite3.IntegrityError:
        return False

    struggle_id = c.lastrowid

    # compute & store embedding if model available
    vec = embed_text_to_array(def_l)
    if vec is not None:
        try:
            c.execute("INSERT INTO struggle_embeddings (struggle_id, embedding) VALUES (?, ?)",
                      (struggle_id, _vec_to_blob(vec)))
            conn.commit()
            EMBED_CACHE[struggle_id] = vec
        except Exception:
            # continue even if embedding save fails
            pass

    return True

def remove_struggle_word(guild_id: int, user_id: int, word: str) -> bool:
    word_l = word.strip().lower()
    c.execute("SELECT id FROM struggle_words WHERE guild_id = ? AND user_id = ? AND word = ?",
              (guild_id, user_id, word_l))
    row = c.fetchone()
    if not row:
        return False
    sid = row[0]
    c.execute("DELETE FROM struggle_embeddings WHERE struggle_id = ?", (sid,))
    c.execute("DELETE FROM struggle_words WHERE id = ?", (sid,))
    conn.commit()
    if sid in EMBED_CACHE:
        EMBED_CACHE.pop(sid, None)
    # also clear any LAST_CHOSEN referencing this
    for k, v in list(LAST_CHOSEN.items()):
        if v == sid:
            LAST_CHOSEN.pop(k, None)
    return True

def clear_user_struggle_words(guild_id: int, user_id: int) -> int:
    """
    Remove all struggle words for a user in a guild. Returns number removed.
    """
    c.execute("SELECT id FROM struggle_words WHERE guild_id = ? AND user_id = ?", (guild_id, user_id))
    ids = [r[0] for r in c.fetchall()]
    if not ids:
        return 0
    for sid in ids:
        c.execute("DELETE FROM struggle_embeddings WHERE struggle_id = ?", (sid,))
        EMBED_CACHE.pop(sid, None)
    c.execute("DELETE FROM struggle_words WHERE guild_id = ? AND user_id = ?", (guild_id, user_id))
    conn.commit()
    # clear LAST_CHOSEN for this pair
    LAST_CHOSEN.pop((guild_id, user_id), None)
    return len(ids)

def get_user_struggle_words(guild_id: int, user_id: int) -> List[Tuple[int, str, str]]:
    """
    Returns list of (id, word, definition).
    """
    c.execute("SELECT id, word, definition FROM struggle_words WHERE guild_id = ? AND user_id = ?", (guild_id, user_id))
    return c.fetchall()

def get_random_struggle_word_for_user(guild_id: int, user_id: int):
    """
    Choose a random struggle word for a user, trying to avoid repeating the last chosen word.
    Returns (id, word, definition) or None.
    """
    rows = get_user_struggle_words(guild_id, user_id)
    if not rows:
        return None

    # If only one row, just return it
    if len(rows) == 1:
        sid, w, d = rows[0]
        LAST_CHOSEN[(guild_id, user_id)] = sid
        return (sid, w, d)

    last_sid = LAST_CHOSEN.get((guild_id, user_id))
    # Try up to N times to pick a different one
    tries = 0
    chosen = None
    while tries < 10:
        sid, w, d = random.choice(rows)
        if sid != last_sid:
            chosen = (sid, w, d)
            break
        tries += 1

    if chosen is None:
        # fallback: pick any (maybe same as last)
        sid, w, d = random.choice(rows)
        chosen = (sid, w, d)

    LAST_CHOSEN[(guild_id, user_id)] = chosen[0]
    return chosen

# --------------------
# EVALUATION / SCORING
# --------------------
def evaluate_answer_for_struggle(struggle_id: int, user_answer: str) -> float:
    """
    Return cosine similarity between stored definition embedding and the user's answer.
    If no embedding or model, returns 0.0.
    """
    # fast path: use cache
    stored_vec = EMBED_CACHE.get(struggle_id)
    if stored_vec is None:
        # try to fetch from DB and place into cache if present
        c.execute("SELECT embedding FROM struggle_embeddings WHERE struggle_id = ?", (struggle_id,))
        row = c.fetchone()
        if row and row[0]:
            try:
                stored_vec = _blob_to_vec(row[0])
                EMBED_CACHE[struggle_id] = stored_vec
            except Exception:
                stored_vec = None

    if stored_vec is None or embedding_model is None:
        return 0.0

    try:
        ans_vec = embedding_model.encode(user_answer, convert_to_numpy=True)
        ans_vec = np.asarray(ans_vec, dtype=np.float32)
        sim = float(cosine_similarity([stored_vec], [ans_vec])[0][0])
        # clip to [0,1]
        if sim != sim:  # NaN check
            return 0.0
        return max(0.0, min(1.0, sim))
    except Exception:
        return 0.0

# --------------------
# POINTS SYSTEM
# --------------------
def add_points(guild_id: int, user_id: int, amount: int):
    """
    Atomically add points (works on INSERT or UPDATE).
    """
    # Ensure row exists
    c.execute("INSERT OR IGNORE INTO points (guild_id, user_id, points) VALUES (?, ?, 0)", (guild_id, user_id))
    c.execute("UPDATE points SET points = points + ? WHERE guild_id = ? AND user_id = ?", (amount, guild_id, user_id))
    conn.commit()

def set_points(guild_id: int, user_id: int, amount: int):
    c.execute("INSERT OR REPLACE INTO points (guild_id, user_id, points) VALUES (?, ?, ?)", (guild_id, user_id, amount))
    conn.commit()

def get_points(guild_id: int, user_id: int) -> int:
    c.execute("SELECT points FROM points WHERE guild_id = ? AND user_id = ?", (guild_id, user_id))
    row = c.fetchone()
    return row[0] if row else 0

# --------------------
# TITLES / SHOP HELPERS
# --------------------
def list_titles() -> List[Tuple[int, str, str, int]]:
    """Return (id, name, color, price)."""
    c.execute("SELECT id, name, color, price FROM titles ORDER BY price ASC")
    return c.fetchall()

def get_title_by_id(title_id: int):
    c.execute("SELECT id, name, color, price FROM titles WHERE id = ?", (title_id,))
    return c.fetchone()

def get_title_by_name(name: str):
    c.execute("SELECT id, name, color, price FROM titles WHERE name = ?", (name,))
    return c.fetchone()

def is_title_owned(title_id: int) -> bool:
    c.execute("SELECT 1 FROM user_titles WHERE title_id = ?", (title_id,))
    return c.fetchone() is not None

def get_user_title(guild_id: int, user_id: int):
    c.execute("""
        SELECT t.id, t.name, t.color FROM user_titles ut
        JOIN titles t ON ut.title_id = t.id
        WHERE ut.guild_id = ? AND ut.user_id = ?
    """, (guild_id, user_id))
    return c.fetchone()

def purchase_title(guild_id: int, user_id: int, title_id: int) -> str:
    """
    Attempt to purchase a title.
    Returns "OK", "NO_TITLE", "NO_POINTS", "ALREADY_OWNED".
    """
    t = get_title_by_id(title_id)
    if not t:
        return "NO_TITLE"
    tid, name, color, price = t

    # Check global uniqueness: title can only be owned by one user across guilds (or across guild?)
    # We enforce per-title uniqueness across all servers per your request.
    c.execute("SELECT 1 FROM user_titles WHERE title_id = ?", (title_id,))
    if c.fetchone():
        return "ALREADY_OWNED"

    pts = get_points(guild_id, user_id)
    if pts < price:
        return "NO_POINTS"

    # Deduct points and assign title
    set_points(guild_id, user_id, pts - price)
    c.execute("INSERT OR REPLACE INTO user_titles (guild_id, user_id, title_id) VALUES (?, ?, ?)", (guild_id, user_id, title_id))
    conn.commit()
    return "OK"

# --------------------
# ROLE / COLOR ASSIGNMENT
# --------------------
async def apply_title_color(member: discord.Member, color_hex: str, title_name: str):
    """
    Create or reuse a role named "[TITLE]" and apply color to the user.
    Removes previous [..] roles the member may have from this system.
    """
    if not member or not member.guild:
        return

    guild = member.guild
    # Convert hex to int safely
    try:
        color_int = int(color_hex.lstrip("#"), 16)
        discord_color = discord.Colour(color_int)
    except Exception:
        # fallback to default
        discord_color = discord.Colour.default()

    role_name = f"[{title_name}]"
    role = discord.utils.get(guild.roles, name=role_name)
    if role is None:
        try:
            role = await guild.create_role(name=role_name, colour=discord_color, reason="Title purchase")
        except Exception:
            # maybe missing perms — bail
            role = None

    # remove existing auto-title roles (those that are bracketed)
    try:
        to_remove = [r for r in member.roles if r.name.startswith("[") and r.name.endswith("]")]
        if to_remove:
            try:
                await member.remove_roles(*to_remove, reason="Replacing title")
            except Exception:
                # ignore removal failures
                pass
    except Exception:
        pass

    if role is not None:
        try:
            await member.add_roles(role, reason="Applying purchased title")
        except Exception:
            pass

# --------------------
# END OF QUADRANT 2
# --------------------

# ============================================================
#                  CHALLENGE SYSTEM (FAST)
# ============================================================

# Active challenges stored by user_id:
# { user_id: {"word": "…", "timestamp": 12345} }
active_challenges = {}

@bot.tree.command(name="challenge", description="Get a random struggle-word to define for points.")
async def challenge(interaction: discord.Interaction):
    user_id = str(interaction.user.id)

    # Prevent challenge spam
    if user_id in active_challenges:
        return await interaction.response.send_message(
            "❗ You already have a challenge active! Answer it first.",
            ephemeral=True
        )

    struggle_path = f"./data/struggle_words/{user_id}.json"

    if not os.path.exists(struggle_path):
        return await interaction.response.send_message(
            "❗ You don’t have any struggle words yet.", ephemeral=True
        )

    with open(struggle_path, "r") as f:
        words = json.load(f)

    if not words:
        return await interaction.response.send_message(
            "❗ Your struggle list is empty.", ephemeral=True
        )

    chosen_word = random.choice(words)

    # Save active challenge
    active_challenges[user_id] = {
        "word": chosen_word,
        "timestamp": time.time()
    }

    await interaction.response.send_message(
        f"📝 **Your challenge word:** `{chosen_word}`\n"
        "Reply with the **English meaning**.\n"
        "You only get *one attempt*. Good luck!"
    )


@bot.event
async def on_message(message):
    # Allow bot to process slash commands normally
    await bot.process_commands(message)

    if message.author.bot:
        return

    user_id = str(message.author.id)

    # Not currently in a challenge
    if user_id not in active_challenges:
        return

    challenge_data = active_challenges[user_id]
    word = challenge_data["word"]

    # Compute embedding of user answer
    user_answer = message.content.strip().lower()

    score = similarity(word, user_answer)

    # Auto-delete active challenge (ONLY ONE TRY)
    del active_challenges[user_id]

    # Award points & respond
    if score >= 0.48:
        new_total = add_points(user_id, 10)
        return await message.reply(
            f"✅ **Correct!** You earned **10 points.**\n"
            f"Your new total: **{new_total} pts**"
        )
    else:
        return await message.reply(
            f"❌ Incorrect.\n"
            f"**Correct meaning:** {word}\n"
            f"Try again with `/challenge`."
        )

# ============================================================
#                     ADMIN CONFIG
# ============================================================

ADMIN_ID = "848935997358211122"  # ← YOU ARE THE ONLY ADMIN


def is_admin(user_id: str):
    return user_id == ADMIN_ID


# ============================================================
#                       SCORE COMMANDS
# ============================================================

@bot.tree.command(name="score", description="Check the score of a user.")
async def score(interaction: discord.Interaction, user: discord.User):
    user_id = str(user.id)
    points = get_points(user_id)
    await interaction.response.send_message(
        f"🏅 **{user.display_name}'s Score:** {points} points"
    )


@bot.tree.command(name="setscore", description="Admin: Set a user's score directly.")
async def setscore(interaction: discord.Interaction, user: discord.User, amount: int):
    if str(interaction.user.id) != ADMIN_ID:
        return await interaction.response.send_message("❌ You are not an admin.", ephemeral=True)

    user_id = str(user.id)
    set_points(user_id, amount)
    await interaction.response.send_message(
        f"🛠️ Set **{user.display_name}'s** score to **{amount}** points."
    )


@bot.tree.command(name="givepoints", description="Admin: Add points to a user.")
async def givepoints(interaction: discord.Interaction, user: discord.User, amount: int):
    if not is_admin(str(interaction.user.id)):
        return await interaction.response.send_message("❌ You are not an admin.", ephemeral=True)

    user_id = str(user.id)
    new_total = add_points(user_id, amount)
    await interaction.response.send_message(
        f"➕ Gave **{amount} points** to {user.display_name}. Total: {new_total}"
    )


@bot.tree.command(name="removepoints", description="Admin: Remove points from a user.")
async def removepoints(interaction: discord.Interaction, user: discord.User, amount: int):
    if not is_admin(str(interaction.user.id)):
        return await interaction.response.send_message("❌ You are not an admin.", ephemeral=True)

    user_id = str(user.id)
    new_total = max(0, get_points(user_id) - amount)
    set_points(user_id, new_total)

    await interaction.response.send_message(
        f"➖ Removed **{amount} points** from {user.display_name}. Total: {new_total}"
    )


# ============================================================
#                       SHOP SYSTEM
# ============================================================

SHOP_FILE = "./data/shop.json"

def load_shop():
    os.makedirs("./data", exist_ok=True)
    if not os.path.exists(SHOP_FILE):
        with open(SHOP_FILE, "w") as f:
            json.dump([], f)
    with open(SHOP_FILE, "r") as f:
        return json.load(f)


def save_shop(items):
    with open(SHOP_FILE, "w") as f:
        json.dump(items, f, indent=2)


@bot.tree.command(name="shop", description="View the shop of titles.")
async def shop(interaction: discord.Interaction):
    items = load_shop()

    if not items:
        return await interaction.response.send_message("🛒 Shop is empty!")

    msg = "🛒 **TITLE SHOP**\n\n"
    for item in items:
        msg += f"[{item['title']}] — {item['price']} pts\n"

    await interaction.response.send_message(msg)


@bot.tree.command(name="shop_add", description="Admin: Add a title to the shop.")
async def shop_add(interaction: discord.Interaction, title: str, price: int):
    if not is_admin(str(interaction.user.id)):
        return await interaction.response.send_message("❌ Admin only.", ephemeral=True)

    items = load_shop()
    items.append({"title": title, "price": price})
    save_shop(items)

    await interaction.response.send_message(
        f"🛒 Added **{title}** to the shop for **{price} pts**."
    )


@bot.tree.command(name="shop_remove", description="Admin: Remove a title from the shop.")
async def shop_remove(interaction: discord.Interaction, title: str):
    if not is_admin(str(interaction.user.id)):
        return await interaction.response.send_message("❌ Admin only.", ephemeral=True)

    items = load_shop()
    items = [item for item in items if item["title"].lower() != title.lower()]
    save_shop(items)

    await interaction.response.send_message(f"🗑️ Removed **{title}** from the shop.")


# ============================================================
#                       TITLES SYSTEM
# ============================================================

USER_TITLES_DIR = "./data/titles"


def get_title(user_id: str):
    os.makedirs(USER_TITLES_DIR, exist_ok=True)
    path = f"{USER_TITLES_DIR}/{user_id}.txt"
    if not os.path.exists(path):
        return ""
    with open(path, "r") as f:
        return f.read().strip()


def set_title(user_id: str, title: str):
    path = f"{USER_TITLES_DIR}/{user_id}.txt"
    os.makedirs(USER_TITLES_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write(title)


@bot.tree.command(name="buy", description="Buy a title from the shop.")
async def buy(interaction: discord.Interaction, title: str):
    user_id = str(interaction.user.id)
    items = load_shop()

    for item in items:
        if item["title"].lower() == title.lower():
            price = item["price"]
            current = get_points(user_id)

            if current < price:
                return await interaction.response.send_message(
                    f"❌ Not enough points. You need {price}."
                )

            set_points(user_id, current - price)
            set_title(user_id, item["title"])

            return await interaction.response.send_message(
                f"🎉 You bought the title **[{item['title']}]**!"
            )

    await interaction.response.send_message("❌ Title not found in shop.")


# ============================================================
#                  STRUGGLE LIST CLEAR
# ============================================================

@bot.tree.command(name="strugglelist_clear", description="Clear your struggle word list.")
async def struggle_clear(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    path = f"./data/struggle_words/{user_id}.json"

    if os.path.exists(path):
        os.remove(path)

    await interaction.response.send_message("🧽 Your struggle list has been cleared!")
