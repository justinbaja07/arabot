import os
import discord
import asyncio
from discord.ext import commands
from datetime import datetime, timedelta
import pytz

TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='/', intents=intents)

# ==============================
# USER PROGRESS DATA
# ==============================
user_data = {
    "users": {},
    "channel_id": None
}


def save_user(user_id):
    """Ensure user is registered in the system."""
    if user_id not in user_data["users"]:
        user_data["users"][user_id] = {
            "streak": 0,
            "last_done_date": None,
            "days_done": []
        }


# ==============================
# TIME HELPERS
# ==============================
TZ = pytz.timezone("America/Chicago")  # Adjust if needed


def today_date():
    return datetime.now(TZ).strftime("%Y-%m-%d")


def yesterday_date():
    return (datetime.now(TZ) - timedelta(days=1)).strftime("%Y-%m-%d")


# ==============================
# /setchannel
# ==============================
@bot.command()
async def setchannel(ctx):
    user_data["channel_id"] = ctx.channel.id
    await ctx.send("Channel set! Daily messages will appear here.")


# ==============================
# /done — mark today's Arabic as completed
# ==============================
@bot.command()
async def done(ctx):
    user_id = str(ctx.author.id)
    save_user(user_id)

    today = today_date()
    user = user_data["users"][user_id]

    if today in user["days_done"]:
        await ctx.send("You already marked today as done!")
        return

    # Add completion
    user["days_done"].append(today)

    # Streak logic
    last = user["last_done_date"]

    if last == yesterday_date():
        user["streak"] += 1
    elif last != today:
        user["streak"] = 1

    user["last_done_date"] = today

    await ctx.send(f"Nice work {ctx.author.mention}! You're now on a **{user['streak']} day streak!** 🔥")


# ==============================
# /undone — reverse today's done
# ==============================
@bot.command()
async def undone(ctx):
    user_id = str(ctx.author.id)
    save_user(user_id)

    today = today_date()
    user = user_data["users"][user_id]

    if today not in user["days_done"]:
        await ctx.send("You never marked today as done, so nothing to undo.")
        return

    # Remove today
    user["days_done"].remove(today)

    # Reduce streak
    if user["streak"] > 0:
        user["streak"] -= 1

    # Fix last_done_date if necessary
    if user["last_done_date"] == today:
        # Set last_done to most recent day done
        if user["days_done"]:
            user["last_done_date"] = sorted(user["days_done"])[-1]
        else:
            user["last_done_date"] = None

    await ctx.send(f"Done undone for today. Your streak is now **{user['streak']}**.")


# ==============================
# /leaderboard
# ==============================
@bot.command()
async def leaderboard(ctx):
    sorted_users = sorted(
        user_data["users"].items(),
        key=lambda x: x[1]["streak"],
        reverse=True
    )

    msg = "**🏆 Arabic Leaderboard 🏆**\n\n"

    for i, (uid, data) in enumerate(sorted_users, start=1):
        user = await bot.fetch_user(int(uid))
        msg += f"**{i}. {user.name} — {data['streak']}🔥 streak**\n"

    await ctx.send(msg)


# ==============================
# /reminder — test 12pm reminder
# ==============================
@bot.command()
async def reminder(ctx):
    await send_reminder()
    await ctx.send("Test reminder sent!")


# ==============================
# /summarytest — test the daily summary
# ==============================
@bot.command()
async def summarytest(ctx):
    await send_summary()
    await ctx.send("Test summary sent!")


# ==============================
# DAILY REMINDER MSG
# ==============================
async def send_reminder():
    if not user_data["channel_id"]:
        return

    channel = bot.get_channel(user_data["channel_id"])
    if channel:
        await channel.send("📘 **Daily Arabic Reminder!** Have you practiced today?")


# ==============================
# DAILY SUMMARY TABLE
# ==============================
async def send_summary():
    if not user_data["channel_id"]:
        return

    channel = bot.get_channel(user_data["channel_id"])
    if not channel:
        return

    yesterday = yesterday_date()

    # Build table header
    msg = "**📊 Yesterday's Arabic Progress Summary**\n"
    msg += "```\n"
    msg += f"{'User':<20} {'Done?':<10} {'Streak':<10}\n"
    msg += "-" * 45 + "\n"

    for uid, data in user_data["users"].items():
        user = await bot.fetch_user(int(uid))

        done = "Yes" if yesterday in data["days_done"] else "No"
        streak = data["streak"]

        msg += f"{user.name:<20} {done:<10} {streak:<10}\n"

    msg += "```"

    await channel.send(msg)


# ==============================
# BACKGROUND TASK
# ==============================
async def background_tasks():
    await bot.wait_until_ready()

    while not bot.is_closed():
        now = datetime.now(TZ)

        # 12 PM reminder
        if now.hour == 12 and now.minute == 0:
            await send_reminder()
            await asyncio.sleep(60)

        # 12 AM summary
        if now.hour == 0 and now.minute == 0:
            await send_summary()
            await asyncio.sleep(60)

        await asyncio.sleep(1)


# Start background loop
bot.loop.create_task(background_tasks())

bot.run(TOKEN)
