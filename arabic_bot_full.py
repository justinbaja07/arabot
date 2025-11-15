# arabic_bot_full.py
import os
import sqlite3
import asyncio
from datetime import datetime, timedelta, date, time as dtime
from zoneinfo import ZoneInfo
from typing import Optional, List, Tuple

import discord
from discord import app_commands
from discord.ext import tasks, commands

# ---------------- CONFIG ----------------
TOKEN = os.getenv("DISCORD_TOKEN")
TIMEZONE = ZoneInfo("America/Chicago")
REMINDER_HOUR = 12  # 12:00 PM CST
REMINDER_MINUTE = 0
GUILD_ID = 1438553047344353293
# ----------------------------------------

# ⏰ Shutdown Schedule
SLEEP_START = dtime(0, 30)     # 12:30 AM CST
SLEEP_END   = dtime(8, 30)     #  8:30 AM CST

intents = discord.Intents.default()
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix="/", intents=intents)
tree = bot.tree

# ---------- Database setup ----------
DB = "arabic_bot.db"
# Allow access from async tasks — keep single connection but it's OK for small bots.
conn = sqlite3.connect(DB, check_same_thread=False)
c = conn.cursor()

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
conn.commit()

# ---------- Helpers ----------
def now_cst():
    return datetime.now(TIMEZONE)

def today_cst_str():
    return now_cst().date().isoformat()

def yesterday_cst_str():
    return (now_cst().date() - timedelta(days=1)).isoformat()

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

def record_completion(guild_id: int, user: discord.Member, date_str: str, time_str: str):
    c.execute("SELECT 1 FROM completions WHERE guild_id = ? AND user_id = ? AND date = ?", (guild_id, user.id, date_str))
    if c.fetchone():
        return False  # already recorded today

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
    c.execute("SELECT username, time FROM completions WHERE guild_id = ? AND date = ? ORDER BY time", (guild_id, date_str))
    return c.fetchall()

def get_user_streak(guild_id: int, user_id: int):
    c.execute("SELECT streak, last_done_date, total_completions FROM streaks WHERE guild_id = ? AND user_id = ?", (guild_id, user_id))
    row = c.fetchone()
    if row:
        return {"streak": row[0], "last_done_date": row[1], "total": row[2]}
    return {"streak": 0, "last_done_date": None, "total": 0}

def get_leaderboard(guild_id: int, limit=10):
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
        role = guild.get_role(settings["ping_role_id"])
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

# =====================================================
# New: Reminder embed + interactive button (behaves like /done)
# =====================================================
from discord import ui

# Global flag to announce wake in on_ready after a sleep cycle
WAS_ASLEEP = False

def build_reminder_embed(guild: discord.Guild, triggered_by: str = "scheduled") -> discord.Embed:
    """
    Build a nicer reminder embed: a small table of today's completions and a mini leaderboard.
    """
    today = today_cst_str()
    done_rows = get_today_completions(guild.id)  # [(username, time), ...]
    lb = get_leaderboard(guild.id, limit=5)

    embed = discord.Embed(title="⏰ Arabic Reminder", color=0x2ECC71)
    embed.set_footer(text=f"Trigger: {triggered_by} • {today} CST")

    # Build a small table (monospaced) for today's completions
    if done_rows:
        # compute width
        name_col = "Name"
        time_col = "Time"
        rows = [(r[0], r[1]) for r in done_rows]
        # compute widths
        name_w = max(len(name_col), *(len(r[0]) for r in rows))
        time_w = max(len(time_col), *(len(r[1]) for r in rows))
        lines = [f"{name_col.ljust(name_w)} | {time_col.ljust(time_w)}", f"{'-'*name_w}-+-{'-'*time_w}"]
        for name, t in rows:
            lines.append(f"{name.ljust(name_w)} | {t.ljust(time_w)}")
        table = "```\n" + "\n".join(lines) + "\n```"
        embed.add_field(name="✅ Today's Completions", value=table, inline=False)
    else:
        embed.add_field(name="✅ Today's Completions", value="No completions yet today. Be the first! ✨", inline=False)

    # mini leaderboard
    if lb:
        lb_lines = []
        medals = ["🥇", "🥈", "🥉"]
        for i, (username, streak, total) in enumerate(lb):
            medal = medals[i] if i < 3 else f"{i+1}."
            lb_lines.append(f"{medal} {username} — Streak: {streak} | Total: {total}")
        embed.add_field(name="🏆 Top Streaks", value="\n".join(lb_lines), inline=False)
    else:
        embed.add_field(name="🏆 Top Streaks", value="No streak data yet.", inline=False)

    embed.description = "Click the button below to mark today's Arabic done — it works just like `/done`."
    return embed

class ReminderView(ui.View):
    def __init__(self, guild: discord.Guild, *, timeout: Optional[float] = 3600):
        super().__init__(timeout=timeout)
        self.guild = guild

    @ui.button(label="Mark Arabic Done (today)", style=discord.ButtonStyle.primary, custom_id=None)
    async def mark_done(self, interaction: discord.Interaction, button: ui.Button):
        """
        This callback acts like the /done command: records completion, updates streaks, and updates the embed in-place.
        """
        guild = interaction.guild
        if guild is None:
            await interaction.response.send_message("This button can only be used in a server.", ephemeral=True)
            return

        date_str = today_cst_str()
        time_str = now_cst().time().strftime("%H:%M:%S")
        success = record_completion(guild.id, interaction.user, date_str, time_str)
        if not success:
            await interaction.response.send_message("You already marked done today.", ephemeral=True)
            return

        # Update the message's embed to reflect the new completion
        try:
            new_embed = build_reminder_embed(guild, triggered_by="interactive")
            # edit original message (if possible)
            if interaction.message:
                await interaction.message.edit(embed=new_embed, view=self)
        except Exception:
            pass

        await interaction.response.send_message(f"Marked done at {time_str} CST. Nice!", ephemeral=True)

async def send_reminder_message(guild: discord.Guild, triggered_by: str = "scheduled") -> Optional[discord.Message]:
    """
    Send the nicer reminder embed + button to the guild's configured channel (or system channel).
    Returns the message if sent.
    """
    settings = get_settings(guild.id)
    channel = guild.get_channel(settings.get("channel_id")) or guild.system_channel
    if not channel:
        return None

    msg_text = None
    if settings.get("ping_enabled") and settings.get("ping_role_id"):
        role = guild.get_role(settings["ping_role_id"])
        if role:
            msg_text = f"{role.mention} Have you done Arabic yet?"

    embed = build_reminder_embed(guild, triggered_by=triggered_by)
    view = ReminderView(guild)
    try:
        if msg_text:
            return await channel.send(content=msg_text, embed=embed, view=view)
        else:
            return await channel.send(embed=embed, view=view)
    except Exception as e:
        print("Failed sending reminder message to guild", guild.id, e)
        return None

# ========== Slash Commands ==========
@tree.command(name="done", description="Mark today's Arabic lesson as done.")
async def slash_done(interaction: discord.Interaction):
    guild = interaction.guild
    if guild is None:
        await interaction.response.send_message("Use this in a server channel.", ephemeral=True)
        return
    date_str = today_cst_str()
    time_str = now_cst().time().strftime("%H:%M:%S")
    success = record_completion(guild.id, interaction.user, date_str, time_str)
    if not success:
        await interaction.response.send_message("You already marked done today.", ephemeral=True)
        return
    await interaction.response.send_message(f"Marked done at {time_str} CST. Nice!", ephemeral=True)

@tree.command(name="undone", description="Undo your Arabic lesson for today if you marked it done.")
async def slash_undone(interaction: discord.Interaction):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("This command must be used in a server.", ephemeral=True)
        return

    user = interaction.user
    date_str = today_cst_str()

    c.execute("SELECT 1 FROM completions WHERE guild_id = ? AND user_id = ? AND date = ?", (guild.id, user.id, date_str))
    if not c.fetchone():
        await interaction.response.send_message("You never marked done today.", ephemeral=True)
        return

    c.execute("DELETE FROM completions WHERE guild_id = ? AND user_id = ? AND date = ?", (guild.id, user.id, date_str))
    c.execute("SELECT streak, total_completions FROM streaks WHERE guild_id = ? AND user_id = ?", (guild.id, user.id))
    row = c.fetchone()
    if row:
        streak = max(0, row[0] - 1)
        total = max(0, row[1] - 1)
        c.execute("UPDATE streaks SET streak = ?, total_completions = ? WHERE guild_id = ? AND user_id = ?", (streak, total, guild.id, user.id))
    conn.commit()
    await interaction.response.send_message(f"{user.mention}, your completion for today has been undone. Streak adjusted.", ephemeral=True)

@tree.command(name="stats", description="Show a user's streak and totals. Use without mention to see your own.")
@app_commands.describe(member="The member to view (optional)")
async def slash_stats(interaction: discord.Interaction, member: discord.Member = None):
    guild = interaction.guild
    if guild is None:
        await interaction.response.send_message("Use in server.", ephemeral=True)
        return
    if member is None:
        member = interaction.user
    s = get_user_streak(guild.id, member.id)
    await interaction.response.send_message(f"{member.mention} — streak: {s['streak']}, last done: {s['last_done_date']}, total completions: {s['total']}")

@tree.command(name="reminder", description="Send today's Arabic reminder immediately (for testing).")
async def slash_reminder(interaction: discord.Interaction):
    guild = interaction.guild
    if guild is None:
        await interaction.response.send_message("Use this command in a server channel.", ephemeral=True)
        return

    settings = get_settings(guild.id)
    channel = guild.get_channel(settings.get("channel_id")) or guild.system_channel
    if channel is None:
        await interaction.response.send_message("No channel set and no system channel found.", ephemeral=True)
        return

    sent = await send_reminder_message(guild, triggered_by="manual")
    if sent:
        # mark last_reminder_date so scheduled won't also send immediately
        upsert_settings(guild.id, last_reminder_date=today_cst_str())
        await interaction.response.send_message(f"Reminder sent to {channel.mention}", ephemeral=True)
    else:
        await interaction.response.send_message("Failed to send reminder. Check bot permissions and channel settings.", ephemeral=True)

@tree.command(name="summary", description="Send today's Arabic summary immediately (for testing).")
async def slash_summary(interaction: discord.Interaction):
    guild = interaction.guild
    if guild is None:
        await interaction.response.send_message("Use this command in a server channel.", ephemeral=True)
        return

    embed = await send_summary_embed(guild)
    await interaction.response.send_message(f"Summary sent!", ephemeral=True)

@tree.command(name="leaderboard", description="Show top streaks for Arabic lessons.")
async def slash_leaderboard(interaction: discord.Interaction):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message("Use in server.", ephemeral=True)
        return

    top = get_leaderboard(guild.id, limit=10)
    if not top:
        await interaction.response.send_message("No data yet.", ephemeral=True)
        return

    embed = discord.Embed(title="🏆 Arabic Streak Leaderboard", color=0xffd700)
    leaderboard_text = ""
    medals = ["🥇", "🥈", "🥉"]
    for i, (username, streak, total) in enumerate(top):
        medal = medals[i] if i < 3 else f"{i+1}."
        leaderboard_text += f"{medal} {username} | Streak: {streak} | Total: {total}\n"

    embed.description = leaderboard_text
    await interaction.response.send_message(embed=embed)

# ---------- Gamified summary embed ----------
async def send_summary_embed(guild: discord.Guild):
    settings = get_settings(guild.id)
    channel = guild.get_channel(settings.get("channel_id")) or guild.system_channel
    if not channel:
        return "No channel set."

    yesterday = yesterday_cst_str()
    c.execute("SELECT username, time FROM completions WHERE guild_id = ? AND date = ?", (guild.id, yesterday))
    done_rows = c.fetchall()
    done_usernames = {r[0] for r in done_rows}

    tracked = []
    if settings.get("ping_role_id"):
        role = guild.get_role(settings["ping_role_id"])
        if role:
            tracked = [m for m in role.members if not m.bot]
    else:
        tracked = [m for m in guild.members if not m.bot]

    not_done = [m for m in tracked if str(m) not in done_usernames]

    embed = discord.Embed(title=f"📊 Yesterday's Arabic Progress - {yesterday}", color=0x00ff00)
    if done_rows:
        done_text = ""
        for username, time_done in done_rows:
            c.execute("SELECT streak, total_completions FROM streaks WHERE guild_id = ? AND username = ?", (guild.id, username))
            row = c.fetchone()
            streak = row[0] if row else 0
            total = row[1] if row else 0
            done_text += f"- {username} | {time_done} | Streak: {streak} | Total: {total}\n"
        embed.add_field(name="✅ Done Yesterday", value=done_text, inline=False)
    else:
        embed.add_field(name="✅ Done Yesterday", value="No one did Arabic yesterday!", inline=False)

    if not_done:
        not_done_text = "\n".join(f"- {m.mention}" for m in not_done)
        embed.add_field(name="❌ Not Done", value=not_done_text, inline=False)
    else:
        embed.add_field(name="❌ Not Done", value="Everyone did Arabic yesterday! 🎉", inline=False)

    await channel.send(embed=embed)
    return embed

# ---------- Background tasks ----------
@bot.event
async def on_ready():
    global WAS_ASLEEP
    print(f"Logged in as {bot.user} (id {bot.user.id})")
    try:
        if GUILD_ID:
            guild = discord.Object(id=GUILD_ID)
            await tree.sync(guild=guild)
        else:
            await tree.sync()
        print("Slash commands synced.")
    except Exception as e:
        print("Failed to sync commands:", e)
    try:
        if not daily_reminder_task.is_running():
            daily_reminder_task.start()
        if not midnight_task.is_running():
            midnight_task.start()
    except Exception:
        pass
    print("Background tasks started.")

    # If we were asleep previously, announce wake to channels
    if WAS_ASLEEP:
        print("Announcing wake to guild channels...")
        for guild in bot.guilds:
            if GUILD_ID and guild.id != GUILD_ID:
                continue
            settings = get_settings(guild.id)
            channel = guild.get_channel(settings.get("channel_id")) or guild.system_channel
            if not channel:
                continue
            try:
                await channel.send("🌅 I'm back! The bot has resumed its normal schedule. Good morning! (8:30 AM CST)")
            except Exception as e:
                print("Failed to send wake message to guild", guild.id, e)
        WAS_ASLEEP = False

@tasks.loop(seconds=60)
async def daily_reminder_task():
    now = now_cst()
    target = now.replace(hour=REMINDER_HOUR, minute=REMINDER_MINUTE, second=0, microsecond=0)
    if now >= target and now < (target + timedelta(minutes=1)):
        for guild in bot.guilds:
            if GUILD_ID and guild.id != GUILD_ID:
                continue
            settings = get_settings(guild.id)
            channel = guild.get_channel(settings.get("channel_id")) or guild.system_channel
            if not channel:
                continue
            if settings.get("last_reminder_date") == today_cst_str():
                continue
            try:
                sent = await send_reminder_message(guild, triggered_by="scheduled")
                if sent:
                    upsert_settings(guild.id, last_reminder_date=today_cst_str())
            except Exception as e:
                print("Failed to send reminder to guild", guild.id, e)

@tasks.loop(minutes=5)
async def midnight_task():
    now = now_cst()
    if now.hour == 0 and now.minute < 5:
        for guild in bot.guilds:
            if GUILD_ID and guild.id != GUILD_ID:
                continue
            await send_summary_embed(guild)
            reset_streaks_for_missed_yesterday(guild)
        await asyncio.sleep(60)

# =====================================================================
# 🛏️ NIGHTLY SHUTDOWN + MORNING RESTART — RAILWAY SAFE
# =====================================================================
async def sleep_wake_loop():
    """
    This task runs 24/7 and performs:
    - Graceful shutdown at 12:30 AM CST
    - Sleep until 8:30 AM CST
    - Auto restart bot
    """
    global WAS_ASLEEP
    while True:
        now = now_cst()
        current = now.time()

        # If inside sleep window → shutdown
        if SLEEP_START <= current < SLEEP_END:
            print("⛔ Bot entering nightly sleep mode.")

            # Announce to guild channels that we're going to sleep (so users see a message at 12:30am)
            for guild in bot.guilds:
                if GUILD_ID and guild.id != GUILD_ID:
                    continue
                settings = get_settings(guild.id)
                channel = guild.get_channel(settings.get("channel_id")) or guild.system_channel
                if not channel:
                    continue
                try:
                    await channel.send("⛔ The bot is going to sleep for the night now (12:30 AM CST). It will return at 8:30 AM CST.")
                except Exception as e:
                    print("Failed to send sleep message to guild", guild.id, e)

            WAS_ASLEEP = True
            try:
                # Close the bot connection gracefully. The process remains alive because this loop continues.
                await bot.close()
            except Exception:
                pass

            # Calculate wake time
            today = now.date()
            wake_dt = datetime.combine(today, SLEEP_END).replace(tzinfo=TIMEZONE)
            if wake_dt <= now:
                wake_dt += timedelta(days=1)

            sleep_seconds = (wake_dt - now).total_seconds()
            print(f"💤 Sleeping for {sleep_seconds/3600:.2f} hours...")
            await asyncio.sleep(sleep_seconds)

            print("🌅 Restarting bot for the day...")
            # start the bot anew (this will re-enter on_ready where tasks are started and wake message announced)
            try:
                await bot.start(TOKEN)
            except Exception as e:
                print("Error restarting bot:", e)
                # backoff a bit and try again
                await asyncio.sleep(10)

        # Not in sleep window: wait and check again
        await asyncio.sleep(60)

# =====================================================================
# MAIN RUN WRAPPER
# =====================================================================
async def main():
    # Start the sleep/wake loop as a background task
    asyncio.create_task(sleep_wake_loop())
    # Start the bot normally
    await bot.start(TOKEN)

if __name__ == "__main__":
    asyncio.run(main())
