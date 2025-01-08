import subprocess
import os
import io
import asyncio
import discord
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
import openai
import pdfplumber
import requests

def install_packages():
    """
    Hack-Script, das bei jedem Start Dependencies aus requirements.txt
    installiert (optional für ZAP-Hosting).
    """
    try:
        print("Starte automatische Paketinstallation...")
        subprocess.check_call(["pip", "install", "--upgrade", "pip"])
        subprocess.check_call(["pip", "install", "--upgrade", "wheel", "setuptools"])
        subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
        print("Paketinstallation abgeschlossen.")
    except Exception as e:
        print(f"Fehler bei der Installation: {e}")

# Optional ausführen
install_packages()

# ======================================================================
# 1. KONFIG
# ======================================================================
DISCORD_TOKEN = ""   # Dein Discord-Bot-Token
OPENAI_API_KEY = ""  # Dein OpenAI-API-Key

# Nur diese UserIDs bekommen Antworten
WHITELIST = [
    12345678901234567890,  # Beispiel
]

# Falls du .env brauchst:
# load_dotenv()
# DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN", DISCORD_TOKEN)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)

openai.api_key = OPENAI_API_KEY

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# ======================================================================
# 2. Slash-Command: /bild (DALL·E 3)
# ======================================================================
@bot.tree.command(name="bild", description="Erstelle ein Bild mit DALL·E 3")
@app_commands.describe(prompt="Was soll gemalt werden?")
async def slash_bild_command(interaction: discord.Interaction, prompt: str):
    """
    Slash-Command: /bild prompt:<DeinPrompt>
    Nutzt DALL-E 3, model="dall-e-3", quality="hd", n=1.
    """
    # Whitelist-Check
    if interaction.user.id not in WHITELIST:
        await interaction.response.send_message(
            "Du bist nicht berechtigt, diesen Bot zu nutzen.",
            ephemeral=True
        )
        return

    await interaction.response.defer(thinking=True)

    try:
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            quality="hd",
            n=1
        )
        image_url = response.data[0].url
        await interaction.followup.send(f"Hier ist dein Bild:\n{image_url}")
    except Exception as e:
        await interaction.followup.send(f"[Fehler bei DALL·E 3] {e}")

# ======================================================================
# 3. on_message -> GPT-4o Vision + PDF + Text
# ======================================================================
@bot.event
async def on_message(message: discord.Message):
    """
    Reagiert auf alle Nachrichten:
      - Bild -> GPT-4o Vision
      - PDF  -> PDF -> Text -> GPT-4o
      - reiner Text -> GPT-4o
    """
    # 1) Eigene Nachrichten ignorieren
    if message.author == bot.user or message.author.bot:
        return

    # 2) Nur Whitelist
    if message.author.id not in WHITELIST:
        return

    # 3) Attachments abchecken
    if message.attachments:
        for att in message.attachments:
            fname = att.filename.lower()
            if fname.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                # GPT-4o Vision
                await handle_image_vision(message, att)
            elif fname.endswith(".pdf"):
                # PDF -> text
                await handle_pdf_to_text(message, att)
            else:
                await message.channel.send(
                    f"Dateityp {fname} wird nicht unterstützt (nur Bilder/PDF)."
                )
        # Falls zusätzlich noch Text in der Nachricht ist, 
        # kannst du es optional auch an GPT-4o schicken:
        if message.content.strip():
            await chat_gpt4o(message, message.content.strip())
    else:
        # Keine Attachments => reiner Text => GPT-4o
        if message.content.strip():
            await chat_gpt4o(message, message.content.strip())

    # Falls du !Befehle oder so hast
    await bot.process_commands(message)

# ======================================================================
# 4. GPT-4o Chat (Text)
# ======================================================================
async def chat_gpt4o(message: discord.Message, user_text: str):
    """
    Reines Chat Completion mit GPT-4o (ohne Bilderzeugung).
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Du bist ein freundlicher Assistent für meine Familie."},
                {"role": "user", "content": user_text}
            ],
            max_tokens=500,
            temperature=0.7
        )
        answer = response.choices[0].message.content
        await send_in_chunks(message.channel, answer)
    except Exception as e:
        await message.channel.send(f"[Fehler GPT-4o] {e}")

# ======================================================================
# 5. GPT-4o Vision
# ======================================================================
async def handle_image_vision(message: discord.Message, attachment: discord.Attachment):
    """
    Schickt das Bild an GPT-4o Vision.
    => image_url: { "url": ... , "detail": "auto" }
    """
    try:
        content_data = [
            {"type": "text", "text": "Was ist auf diesem Bild zu sehen?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": attachment.url,
                    "detail": "auto"
                },
            }
        ]
        response = openai.chat.completions.create(
            model="gpt-4o",  # Vision-fähiges GPT-4o
            messages=[{"role": "user", "content": content_data}],
            max_tokens=500
        )
        answer = response.choices[0].message.content
        await send_in_chunks(message.channel, answer)
    except Exception as e:
        await message.channel.send(f"[Fehler bei GPT-4o Vision] {e}")

# ======================================================================
# 6. PDF -> Text -> GPT-4o
# ======================================================================
async def handle_pdf_to_text(message: discord.Message, attachment: discord.Attachment):
    """
    1) Lade PDF herunter
    2) Konvertiere zu Text (pdfplumber)
    3) Schicke an GPT-4o (Text).
    """
    try:
        pdf_bytes = await attachment.read()
        pdf_text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                pdf_text += page.extract_text() or ""

        short_pdf_text = pdf_text[:8000]  # Bisschen kürzen, um Tokenlimit zu vermeiden
        user_query = message.content.strip() or "Bitte fasse den Inhalt zusammen."

        # Prompt
        system_text = (
            "Du bist ein Assistent, der PDF-Inhalte zusammenfassen oder erklären kann. "
            "Ich gebe dir den extrahierten Text. "
            "Der Nutzer fragt: '"+user_query+"'"
        )
        messages_data = [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": f"PDF-Inhalt (Auszug):\n{short_pdf_text}\n\n"
                           f"Meine Frage: {user_query}"
            }
        ]
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages_data,
            max_tokens=800,
            temperature=0.7
        )
        answer = response.choices[0].message.content
        await send_in_chunks(message.channel, answer)
    except Exception as e:
        await message.channel.send(f"[Fehler bei PDF-Verarbeitung] {e}")

# ======================================================================
# 7. Hilfsfunktion: lange Nachrichten in Chunks senden
# ======================================================================
async def send_in_chunks(channel: discord.abc.Messageable, text: str):
    chunk_size = 2000
    for i in range(0, len(text), chunk_size):
        await channel.send(text[i:i+chunk_size])

# ======================================================================
# 8. BOT-START
# ======================================================================
async def main():
    print("Starte Discord-Bot...")
    await bot.start(DISCORD_TOKEN)

@bot.event
async def on_ready():
    try:
        await bot.tree.sync()
        print(f"Eingeloggt als: {bot.user} (ID: {bot.user.id})")
        print("Bot ist bereit (GPT-4o Vision/PDF/Text + /bild via DALL·E 3).")
    except Exception as e:
        print(f"Fehler beim Slash-Command-Sync: {e}")

if __name__ == "__main__":
    asyncio.run(main())
