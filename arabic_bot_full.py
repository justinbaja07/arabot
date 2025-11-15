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

# ---------- Database connection ----------
DB = "arabic_bot.db"
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
    """Generate embedding bytes from text."""
    vec = embedding_model.encode(text)
    arr = np.asarray(vec, dtype=np.float32)
    return arr.tobytes()

def get_embedding_array(blob: bytes) -> np.ndarray:
    """Convert stored blob back to array."""
    return np.frombuffer(blob, dtype=np.float32)

def add_struggle_word(guild_id: int, user_id: int, word: str, definition: str):
    """Insert a struggle word + its embedding."""
    try:
        c.execute("""
            INSERT INTO struggle_words (guild_id, user_id, word, definition)
            VALUES (?, ?, ?, ?)
        """, (guild_id, user_id, word.lower(), definition.lower()))
        conn.commit()
    except sqlite3.IntegrityError:
        return False

    # Insert embedding
    struggle_id = c.lastrowid
    emb = embed_text(definition)
    c.execute("""
        INSERT INTO struggle_embeddings (struggle_id, embedding)
        VALUES (?, ?)
    """, (struggle_id, emb))
    conn.commit()
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

def get_random_struggle_word(guild_id: int, user_id: int):
    rows = get_user_struggle_words(guild_id, user_id)
    return random.choice(rows) if rows else None

def evaluate_answer(user_answer: str, correct_definition: str, struggle_id: int):
    """
    Compare embeddings of the correct definition vs. user's answer.
    Returns similarity (0 to ~1).
    """
    # Fetch stored embedding
    c.execute("SELECT embedding FROM struggle_embeddings WHERE struggle_id = ?", (struggle_id,))
    row = c.fetchone()
    if not row:
        return 0.0

    stored_vec = get_embedding_array(row[0])

    # Embed user's answer
    ans_vec = embedding_model.encode(user_answer)
    ans_vec = np.asarray(ans_vec, dtype=np.float32)

    sim = cosine_similarity([stored_vec], [ans_vec])[0][0]

    return sim


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
    c.execute("SELECT points FROM points WHERE guild_id = ? AND user_id = ?", (guild_id, user_id))
    row = c.fetchone()
    return row[0] if row else 0


# ----------------------------------------
# TITLES SYSTEM
# ----------------------------------------

def create_title(name: str, color: str, price: int):
    try:
        c.execute("""
            INSERT INTO titles (name, color, price)
            VALUES (?, ?, ?)
        """, (name, color, price))
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

    # Deduct points
    c.execute("UPDATE points SET points = points - ? WHERE guild_id = ? AND user_id = ?",
              (price, guild_id, user_id))

    # Assign title
    c.execute("""
        INSERT OR REPLACE INTO user_titles (guild_id, user_id, title_id)
        VALUES (?, ?, ?)
    """, (guild_id, user_id, title_id))

    conn.commit()
    return "OK"


# ----------------------------------------
# AUTO COLOR ROLE SYSTEM
# ----------------------------------------

async def apply_title_color(member: discord.Member, color_hex: str, title_name: str):
    """
    Creates/gets a role with the correct title & color and applies it to the user.
    """
    guild = member.guild
    color_int = int(color_hex.lstrip("#"), 16)

    # Check if role exists
    role = discord.utils.get(guild.roles, name=f"[{title_name}]")
    if role is None:
        role = await guild.create_role(
            name=f"[{title_name}]",
            colour=discord.Colour(color_int),
            reason="Title purchase"
        )

    # Remove old title roles
    for r in member.roles:
        if r.name.startswith("[") and r.name.endswith("]"):
            try:
                await member.remove_roles(r)
            except:
                pass

    # Give new one
    await member.add_roles(role)


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

    msg = "**Your Struggle Words:**\n\n"
    for _, word, definition in rows:
        msg += f"• **{word}** → `{definition}`\n"

    await interaction.response.send_message(msg, ephemeral=True)



# ------------------------------------------------------------
# /challenge — gives a random struggle word & waits for answer
# ------------------------------------------------------------

active_challenges = {}  # key: user_id, value: (struggle_id, correct_def)

@tree.command(name="challenge", description="Get quizzed on one of your struggle words.")
async def challenge(interaction: discord.Interaction):
    guild_id = interaction.guild_id
    user_id = interaction.user.id

    row = get_random_struggle_word(guild_id, user_id)
    if not row:
        await interaction.response.send_message(
            "❌ You have no struggle words. Add some first.",
            ephemeral=True
        )
        return

    struggle_id, word, definition = row

    active_challenges[user_id] = (struggle_id, definition)

    await interaction.response.send_message(
        f"🧠 **Challenge Time!**\n\nWhat is the definition of:\n\n👉 **{word}** ?\n\nType your answer in chat.",
        ephemeral=False
    )



# ------------------------------------------------------------
# /points — see your points
# ------------------------------------------------------------

@tree.command(name="points", description="Check your points.")
async def points_cmd(interaction: discord.Interaction):
    guild_id = interaction.guild_id
    user_id = interaction.user.id

    pts = get_points(guild_id, user_id)
    await interaction.response.send_message(
        f"⭐ You have **{pts}** points.",
        ephemeral=True
    )


# ------------------------------------------------------------
# /shop — list titles
# ------------------------------------------------------------

@tree.command(name="shop", description="View the title shop.")
async def shop_cmd(interaction: discord.Interaction):
    rows = list_titles()
    if not rows:
        await interaction.response.send_message(
            "Shop is empty.",
            ephemeral=True
        )
        return

    msg = "**🏪 TITLE SHOP:**\n\n"
    for tid, name, color, price in rows:
        msg += f"• **[{name}]** — `{color}` — **{price} pts** (id: `{tid}`)\n"

    await interaction.response.send_message(msg, ephemeral=True)


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

    # Success → apply role
    c.execute("SELECT name, color FROM titles WHERE id = ?", (title_id,))
    row = c.fetchone()
    name, color = row

    try:
        await apply_title_color(interaction.guild.get_member(user.id), color, name)
    except Exception as e:
        await interaction.response.send_message(f"⚠️ Title purchased but color assignment failed.", ephemeral=True)
        return

    await interaction.response.send_message(
        f"🎉 You bought the title **[{name}]**!",
        ephemeral=False
    )


# ------------------------------------------------------------
# /done — the main daily check-in
# ------------------------------------------------------------

@tree.command(name="done", description="Mark today as completed.")
async def done_cmd(interaction: discord.Interaction):
    guild = interaction.guild
    user = interaction.user

    date_str = today_cst_str()
    time_str = now_cst().strftime("%I:%M %p")

    success = record_completion(guild.id, user, date_str, time_str)

    if not success:
        await interaction.response.send_message(
            "You already marked today as done.",
            ephemeral=True
        )
        return

    await interaction.response.send_message(
        f"🔥 **{user.display_name}** marked today as DONE!",
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

    user_id = message.author.id
    guild_id = message.guild.id if message.guild else None

    # Check if user is in an active challenge
    if user_id in active_challenges:
        struggle_id, correct_definition = active_challenges[user_id]

        # Compare LLM-evaluated correctness
        user_answer = message.content.strip()
        score = ai_similarity(user_answer, correct_definition)

        if score >= 0.5:
            # Award points
            add_points(guild_id, user_id, 5)
            del active_challenges[user_id]

            await message.channel.send(
                f"🔥 **Correct!** You earned **5 points!**\n"
                f"Similarity score: `{score:.2f}`"
            )
        else:
            await message.channel.send(
                f"❌ Not quite. Try again!\n"
                f"Similarity score: `{score:.2f}`"
            )

    await bot.process_commands(message)


# ============================================================
#                 DAILY REMINDER TASK (12 PM)
# ============================================================

@tasks.loop(seconds=60)
async def daily_reminder_task():
    now = now_cst()
    target = now.replace(hour=12, minute=0, second=0, microsecond=0)
    if now >= target and now < (target + timedelta(minutes=1)):
        for guild in bot.guilds:
            settings = get_settings(guild.id)

            if settings["last_reminder_date"] == today_cst_str():
                continue

            await send_reminder_message(guild)
            upsert_settings(guild.id, last_reminder_date=today_cst_str())


# ============================================================
#                  MIDNIGHT SUMMARY TASK (12 AM)
# ============================================================

@tasks.loop(minutes=5)
async def midnight_task():
    now = now_cst()

    # Runs one time between 12:00–12:05 AM CST
    if now.hour == 0 and now.minute < 5:
        for guild in bot.guilds:
            await send_summary_embed(guild)
            reset_streaks_for_missed_yesterday(guild)

        await asyncio.sleep(60)  # prevent double firing


# ============================================================
#                          ON READY
# ============================================================

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

    # Sync commands
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


