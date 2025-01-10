import subprocess
import os
import io
import base64
import asyncio
import pymysql
import discord
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
import pdfplumber
import requests
from pydub import AudioSegment  # Für die Audio-Konvertierung
import json
import logging

# ======================================================================
# 1. KONFIGURATION
# ======================================================================
# Lade Umgebungsvariablen aus einer .env Datei (empfohlen)
load_dotenv()

# Discord und OpenAI API-Schlüssel
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Modelle für OpenAI
MODEL_GPT4O = os.getenv("MODEL_GPT4O", "gpt-4o")
MODEL_AUDIO = os.getenv("MODEL_AUDIO", "gpt-4o-audio-preview")

# OpenAI API Konfiguration
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 3000))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))

# Datenbankkonfiguration
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

# Anzahl der gespeicherten Nachrichten pro Benutzer
MAX_CONVERSATIONS = int(os.getenv("MAX_CONVERSATIONS", 20))

# Anzahl der Nachrichten, für die der Bildkontext genutzt wird
IMAGE_CONTEXT_LIMIT = int(os.getenv("IMAGE_CONTEXT_LIMIT", 3))

# Whitelist-UserIDs (nur diese können den Bot nutzen)
WHITELIST = [
    773954516692500521,  # Michael
    824660719453863936,  # Luca
    824656617899032596,  # Neo
]

# ======================================================================
# 2. DATENBANK-KONFIG (MySQL/MariaDB)
# ======================================================================
def db_connect():
    """
    Baut eine Verbindung zur MySQL/MariaDB-Datenbank auf.
    """
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor
        )
        logger.debug("Datenbankverbindung erfolgreich hergestellt.")
        return connection
    except pymysql.MySQLError as e:
        logger.error(f"Datenbankverbindungsfehler: {e}")
        raise

def init_db():
    """
    Legt die Tabellen user_conversations und user_images an, falls sie nicht existieren.
    Fügt die Spalte image_context_counter hinzu, falls sie fehlt.
    """
    con = db_connect()
    with con.cursor() as cur:
        # Tabelle user_conversations erstellen
        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_conversations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id BIGINT NOT NULL,
            role VARCHAR(16) NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Tabelle user_images erstellen
        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_images (
            user_id BIGINT PRIMARY KEY,
            image_data JSON,
            image_context_counter INT DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """)
        
        # Überprüfen, ob die Spalte image_context_counter existiert, und sie hinzufügen, falls nicht
        cur.execute("""
        SHOW COLUMNS FROM user_images LIKE 'image_context_counter';
        """)
        result = cur.fetchone()
        if not result:
            cur.execute("""
            ALTER TABLE user_images
            ADD COLUMN image_context_counter INT DEFAULT 0;
            """)
            logger.info("Spalte 'image_context_counter' zur Tabelle 'user_images' hinzugefügt.")
        
    con.commit()
    con.close()
    logger.info("[Info] Datenbank initialisiert und aktualisiert.")

# ======================================================================
# 3. INSTANTIIERTEN CLIENT (OpenAI v1.x)
# ======================================================================
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# ======================================================================
# 4. LOGGING KONFIGURATION
# ======================================================================
logging.basicConfig(
    level=logging.DEBUG,  # Erhöht das Logging-Level auf DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),  # Log-Datei
        logging.StreamHandler()          # Log-Ausgabe in der Konsole
    ]
)
logger = logging.getLogger(__name__)

# ======================================================================
# 5. DB-FUNKTIONEN FÜR CHATVERLÄUFE UND BILDER
# ======================================================================
def load_conversations_from_db() -> dict:
    """
    Lädt sämtliche Chatverläufe aus DB:
    { user_id: [ {role:..., content:...}, ...], ... }
    """
    user_conversations = {}
    con = db_connect()
    with con.cursor() as cur:
        cur.execute("SELECT user_id, role, content FROM user_conversations ORDER BY id ASC")
        rows = cur.fetchall()
        for row in rows:
            uid = row["user_id"]
            if uid not in user_conversations:
                user_conversations[uid] = []
            user_conversations[uid].append({
                "role": row["role"],
                "content": row["content"]
            })
    con.close()
    logger.info("[Info] Chatverläufe aus DB geladen.")
    return user_conversations

def save_message_to_db(user_id: int, role: str, content: str):
    """
    Speichert eine einzelne Nachricht in der DB.
    """
    con = db_connect()
    with con.cursor() as cur:
        sql = """
        INSERT INTO user_conversations (user_id, role, content)
        VALUES (%s, %s, %s)
        """
        cur.execute(sql, (user_id, role, content))
    con.commit()
    con.close()
    logger.debug(f"[DB] Nachricht gespeichert: user_id={user_id}, role={role}, content={content}")

def ensure_limit_in_db(user_id: int, limit=MAX_CONVERSATIONS):
    """
    Löscht alte Einträge, falls mehr als 'limit' Nachrichten pro User in DB.
    """
    con = db_connect()
    with con.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS cnt FROM user_conversations WHERE user_id=%s", (user_id,))
        row = cur.fetchone()
        if row and row["cnt"] > limit:
            to_delete = row["cnt"] - limit
            cur.execute("""
                SELECT id FROM user_conversations
                WHERE user_id=%s
                ORDER BY id ASC
                LIMIT %s
            """, (user_id, to_delete))
            oldest = cur.fetchall()
            if oldest:
                ids_to_delete = [r["id"] for r in oldest]
                placeholders = ",".join(["%s"] * len(ids_to_delete))
                sql = f"DELETE FROM user_conversations WHERE id IN ({placeholders})"
                cur.execute(sql, tuple(ids_to_delete))
                logger.debug(f"[DB] Alte Nachrichten gelöscht: IDs={ids_to_delete}")
    con.commit()
    con.close()

def get_stored_image(user_id: int) -> dict:
    """
    Holt das gespeicherte Bild und den Kontextzähler für den Benutzer, falls vorhanden.
    """
    con = db_connect()
    with con.cursor() as cur:
        cur.execute("SELECT image_data, image_context_counter FROM user_images WHERE user_id=%s", (user_id,))
        row = cur.fetchone()
        if row:
            logger.debug(f"[DB] Bildkontext gefunden für user_id={user_id}")
            return {
                "image_data": json.loads(row["image_data"]),
                "image_context_counter": row["image_context_counter"]
            }
    con.close()
    logger.debug(f"[DB] Kein Bildkontext gefunden für user_id={user_id}")
    return {"image_data": [], "image_context_counter": 0}

def store_image(user_id: int, image_content: list):
    """
    Speichert das Bild für den Benutzer in der DB und setzt den Kontextzähler.
    """
    image_json = json.dumps(image_content)
    con = db_connect()
    with con.cursor() as cur:
        cur.execute("""
            INSERT INTO user_images (user_id, image_data, image_context_counter)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE image_data=%s, image_context_counter=%s, updated_at=NOW()
        """, (user_id, image_json, IMAGE_CONTEXT_LIMIT, image_json, IMAGE_CONTEXT_LIMIT))
    con.commit()
    con.close()
    logger.debug(f"[DB] Bildkontext für user_id={user_id} gespeichert/aktualisiert.")

def decrement_image_context(user_id: int):
    """
    Verringert den Bildkontextzähler für den Benutzer um 1.
    """
    con = db_connect()
    with con.cursor() as cur:
        cur.execute("""
            UPDATE user_images
            SET image_context_counter = image_context_counter - 1
            WHERE user_id = %s AND image_context_counter > 0
        """, (user_id,))
    con.commit()
    con.close()
    logger.debug(f"[DB] Bildkontextzähler für user_id={user_id} dekrementiert.")

# ======================================================================
# 6. DISCORD BOT
# ======================================================================
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)
user_conversations_cache = {}

def get_user_conversation(user_id: int) -> list:
    """
    Holt den Chatverlauf aus user_conversations_cache, init bei Bedarf.
    """
    if user_id not in user_conversations_cache:
        user_conversations_cache[user_id] = [
            {"role": "system", "content": "Du bist ein Assistent für diese Familie."}
        ]
        save_message_to_db(user_id, "system", user_conversations_cache[user_id][0]["content"])
        logger.debug(f"[Cache] Neuer Chatverlauf für user_id={user_id} erstellt.")
    return user_conversations_cache[user_id]

# ======================================================================
# 7. /bild - Slash-Command (DALL·E 3)
# ======================================================================
@bot.tree.command(name="bild", description="Erstelle ein Bild mit DALL-E 3")
@app_commands.describe(prompt="Was soll gemalt werden?")
async def slash_bild_command(interaction: discord.Interaction, prompt: str):
    """
    Erstellt ein Bild via DALL-E 3.
    """
    if interaction.user.id not in WHITELIST:
        await interaction.response.send_message("Du bist nicht berechtigt.", ephemeral=True)
        logger.warning(f"[Zugriff verweigert] user_id={interaction.user.id}")
        return

    await interaction.response.defer(thinking=True)
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url
        await interaction.followup.send(f"Hier ist dein Bild:\n{image_url}")
        logger.info(f"[Bild erstellt] user_id={interaction.user.id}, prompt='{prompt}'")

        # Speichere die Bildanfrage und die Antwort in der DB
        convo = get_user_conversation(interaction.user.id)
        convo.append({"role": "user", "content": f"/bild {prompt}"})
        save_message_to_db(interaction.user.id, "user", f"/bild {prompt}")

        convo.append({"role": "assistant", "content": f"Hier ist dein Bild:\n{image_url}"})
        save_message_to_db(interaction.user.id, "assistant", f"Hier ist dein Bild:\n{image_url}")
        ensure_limit_in_db(interaction.user.id, limit=MAX_CONVERSATIONS)

        # Speichere das Bild für zukünftige Anfragen und setze den Kontextzähler
        image_content = [
            {"type": "text", "text": "Du hast ein Bild in der Beilage. Mache, was der Benutzer wünscht."},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": "auto"
                }
            }
        ]
        store_image(interaction.user.id, image_content)

    except Exception as e:
        logger.error(f"[Fehler bei DALL-E 3] {e}")
        await interaction.followup.send(f"[Fehler bei DALL-E 3] {e}")

# ======================================================================
# 8. on_message -> GPT-4o (Vision + PDF + Text + Audio)
# ======================================================================
@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user or message.author.bot:
        return

    if message.author.id not in WHITELIST:
        return

    user_id = message.author.id

    # Sammle alle Inhalte (Anhänge und Text)
    attachments = message.attachments
    text = message.content.strip()

    # Wenn sowohl Anhänge als auch Text vorhanden sind, verarbeite sie zusammen
    if attachments and text:
        # Begrenze auf eine bestimmte Anzahl von Anhängen
        attachments = attachments[:5]  # Beispiel: Max. 5 Anhänge

        # Initialisiere Listen für verschiedene Typen
        image_attachments = []
        pdf_attachments = []
        audio_attachments = []
        unsupported_attachments = []

        # Sortiere Anhänge nach Typ
        for att in attachments:
            fname = att.filename.lower()
            if fname.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                image_attachments.append(att)
            elif fname.endswith(".pdf"):
                pdf_attachments.append(att)
            elif fname.endswith((".ogg", ".wav", ".mp3")):
                audio_attachments.append(att)
            else:
                unsupported_attachments.append(att)

        # Verarbeitung der Anhänge
        image_contents = []
        for att in image_attachments:
            try:
                image_content = await process_image_attachment(message, att)
                if image_content:
                    image_contents.append(image_content)
            except Exception as e:
                logger.error(f"[Fehler bei der Bildverarbeitung] {e}")
                await message.channel.send(f"[Fehler bei der Bildverarbeitung] {e}")

        for att in pdf_attachments:
            await handle_pdf_to_text(message, att)

        for att in audio_attachments:
            await handle_audio_message(message, att)

        for att in unsupported_attachments:
            await message.channel.send(
                f"Dateityp {att.filename} wird nicht unterstützt (nur Bilder/PDF/Audio)."
            )

        # Verarbeitung des Textes zusammen mit den Bildern
        if image_contents or text:
            # Lade das gespeicherte Bild, falls vorhanden
            stored_image_info = get_stored_image(user_id)
            stored_image = stored_image_info["image_data"]
            image_context_counter = stored_image_info["image_context_counter"]

            if image_contents:
                # Flachmachen der Bildinhalte
                flat_image_contents = [item for sublist in image_contents for item in sublist]
                # Update das gespeicherte Bild mit den neuen Bildern und setze den Kontextzähler
                store_image(user_id, flat_image_contents)
                stored_image = flat_image_contents
                image_context_counter = IMAGE_CONTEXT_LIMIT

            # Entscheide, ob der Bildkontext eingebunden werden soll
            if image_context_counter > 0:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                        ] + stored_image  # Füge das Bild zur Anfrage hinzu
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": text
                    }
                ]

            # Hole den Chatverlauf
            convo = get_user_conversation(user_id)

            try:
                response = client.chat.completions.create(
                    model=MODEL_GPT4O,
                    messages=convo + messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE
                )
                answer = response.choices[0].message.content

                # Sende die Antwort an Discord
                await send_in_chunks(message.channel, answer)
                logger.info(f"[Antwort gesendet] user_id={user_id}, message='{text}'")

                # Speichere die Antwort in der DB
                convo.append({"role": "user", "content": text})
                save_message_to_db(user_id, "user", text)
                convo.append({"role": "assistant", "content": answer})
                save_message_to_db(user_id, "assistant", answer)
                ensure_limit_in_db(user_id, limit=MAX_CONVERSATIONS)

                # Decrementiere den Bildkontextzähler
                if image_context_counter > 0:
                    decrement_image_context(user_id)

            except Exception as e:
                logger.error(f"[Fehler GPT-4o] {e}")
                await message.channel.send(f"[Fehler GPT-4o] {e}")

    elif attachments:
        # Verarbeitung nur von Anhängen
        # Initialisiere Listen für verschiedene Typen
        image_attachments = []
        pdf_attachments = []
        audio_attachments = []
        unsupported_attachments = []

        # Sortiere Anhänge nach Typ
        for att in attachments:
            fname = att.filename.lower()
            if fname.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                image_attachments.append(att)
            elif fname.endswith(".pdf"):
                pdf_attachments.append(att)
            elif fname.endswith((".ogg", ".wav", ".mp3")):
                audio_attachments.append(att)
            else:
                unsupported_attachments.append(att)

        # Verarbeitung der Anhänge
        image_contents = []
        for att in image_attachments:
            try:
                image_content = await process_image_attachment(message, att)
                if image_content:
                    image_contents.append(image_content)
            except Exception as e:
                logger.error(f"[Fehler bei der Bildverarbeitung] {e}")
                await message.channel.send(f"[Fehler bei der Bildverarbeitung] {e}")

        for att in pdf_attachments:
            await handle_pdf_to_text(message, att)

        for att in audio_attachments:
            await handle_audio_message(message, att)

        for att in unsupported_attachments:
            await message.channel.send(
                f"Dateityp {att.filename} wird nicht unterstützt (nur Bilder/PDF/Audio)."
            )

    elif text:
        # Verarbeitung nur von Text, eventuell mit gespeicherten Bildern
        stored_image_info = get_stored_image(user_id)
        stored_image = stored_image_info["image_data"]
        image_context_counter = stored_image_info["image_context_counter"]

        if image_context_counter > 0 and stored_image:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ] + stored_image  # Füge das Bild zur Anfrage hinzu
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": text
                }
            ]

        # Hole den Chatverlauf
        convo = get_user_conversation(user_id)

        try:
            response = client.chat.completions.create(
                model=MODEL_GPT4O,
                messages=convo + messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            answer = response.choices[0].message.content

            # Sende die Antwort an Discord
            await send_in_chunks(message.channel, answer)
            logger.info(f"[Antwort gesendet] user_id={user_id}, message='{text}'")

            # Speichere die Antwort in der DB
            convo.append({"role": "user", "content": text})
            save_message_to_db(user_id, "user", text)
            convo.append({"role": "assistant", "content": answer})
            save_message_to_db(user_id, "assistant", answer)
            ensure_limit_in_db(user_id, limit=MAX_CONVERSATIONS)

            # Decrementiere den Bildkontextzähler
            if image_context_counter > 0:
                decrement_image_context(user_id)

        except Exception as e:
            logger.error(f"[Fehler GPT-4o] {e}")
            await message.channel.send(f"[Fehler GPT-4o] {e}")

    await bot.process_commands(message)

# ======================================================================
# 9. Verarbeitung von Bildanhängen
# ======================================================================
async def process_image_attachment(message: discord.Message, attachment: discord.Attachment) -> list:
    """
    Verarbeitet einen Bildanhang und gibt die strukturierte Content-Liste zurück.
    """
    try:
        # Hier wird nur die URL verwendet, daher kein Download notwendig
        image_content = [
            {"type": "text", "text": "Du hast ein Bild in der Beilage. Mache, was der Benutzer wünscht."},
            {
                "type": "image_url",
                "image_url": {
                    "url": attachment.url,
                    "detail": "auto"
                }
            }
        ]
        logger.debug(f"[Bild verarbeitet] user_id={message.author.id}, URL={attachment.url}")
        return image_content
    except Exception as e:
        logger.error(f"[Fehler bei der Bildanhang-Verarbeitung] {e}")
        await message.channel.send(f"[Fehler bei der Bildanhang-Verarbeitung] {e}")
        return []

# ======================================================================
# 10. PDF -> GPT-4o
# ======================================================================
async def handle_pdf_to_text(message: discord.Message, attachment: discord.Attachment):
    """
    Liest PDF, extrahiert Text, schickt an GPT-4o (Text).
    """
    user_id = message.author.id
    try:
        pdf_bytes = await attachment.read()
        text_data = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text_data += extracted_text + "\n"

        short_pdf_text = text_data[:20000]
        user_query = message.content.strip() or "Bitte fasse den Inhalt zusammen."

        system_text = (
            "Du bist ein Assistent, der PDF-Inhalte zusammenfassen oder erklären kann. "
            f"Der Nutzer fragt: '{user_query}'"
        )
        response = client.chat.completions.create(
            model=MODEL_GPT4O,
            messages=[
                {"role": "system", "content": system_text},
                {
                    "role": "user",
                    "content": (
                        f"PDF-Inhalt (Auszug):\n{short_pdf_text}\n\nFrage: {user_query}"
                    )
                }
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        answer = response.choices[0].message.content

        # Sende die Antwort an Discord
        await send_in_chunks(message.channel, answer)
        logger.info(f"[PDF-Antwort gesendet] user_id={user_id}, PDF={attachment.filename}")

        # Speichere die PDF-Anfrage und die Antwort in der DB
        convo = get_user_conversation(user_id)
        convo.append({"role": "user", "content": f"PDF verarbeitet: {attachment.filename}"})
        save_message_to_db(user_id, "user", f"PDF verarbeitet: {attachment.filename}")

        convo.append({"role": "assistant", "content": answer})
        save_message_to_db(user_id, "assistant", answer)
        ensure_limit_in_db(user_id, limit=MAX_CONVERSATIONS)

    except Exception as e:
        logger.error(f"[Fehler bei PDF-Verarbeitung] {e}")
        await message.channel.send(f"[Fehler bei PDF-Verarbeitung] {e}")

# ======================================================================
# 11. GPT-4o Audio Beta -> Sprachnachrichten
# ======================================================================
async def handle_audio_message(message: discord.Message, attachment: discord.Attachment):
    """
    Nimmt z.B. voice-message.ogg -> GPT-4o Audio Beta
    => Schickt Audio rein, holt Text + Audio raus und schickt beides an Discord.
    """
    user_id = message.author.id
    try:
        # 1) Audio-Datei herunterladen
        audio_bytes = await attachment.read()

        # 2) Konvertiere OGG zu WAV
        audio_format = attachment.filename.split('.')[-1]
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_data = wav_io.getvalue()

        # 3) Base64 encode
        audio_base64 = base64.b64encode(wav_data).decode("utf-8")

        # 4) Audio-Input an GPT-4o Audio Beta senden
        messages = [
            {
                "role": "user",
                "content": [
                    { 
                        "type": "text",
                        "text": "Bitte antworte auf diese Nachricht."
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": "wav"
                        }
                    }
                ]
            },
        ]

        response = client.chat.completions.create(
            model=MODEL_AUDIO,  # gpt-4o-audio-preview
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )

        # 5) Antwort verarbeiten
        ans = response.choices[0].message
        text_answer = ans.content or (
            ans.audio.transcript if hasattr(ans, "audio") else "(Keine Textantwort erhalten)"
        )
        audio_info = ans.audio if hasattr(ans, "audio") else None

        # Speichern der Benutzeranfrage (transkribierter Text)
        user_transcript = ans.audio.transcript if hasattr(ans, "audio") else text_answer
        convo = get_user_conversation(user_id)
        convo.append({"role": "user", "content": user_transcript})
        save_message_to_db(user_id, "user", user_transcript)

        if audio_info and audio_info.data:
            # Audio-Daten dekodieren
            audio_reply = base64.b64decode(audio_info.data)

            # WAV zu OGG konvertieren mit FFmpeg
            ogg_data = convert_wav_to_ogg(audio_reply)

            # Senden: Text und Audio-File mit .opus-Erweiterung
            ogg_io = io.BytesIO(ogg_data)
            ogg_io.seek(0)  # Setze den Zeiger zurück auf den Anfang

            filename = f"reply_{message.id}.opus"
            await message.channel.send(
                text_answer,
                file=discord.File(ogg_io, filename=filename)
            )
            logger.info(f"[Audio-Antwort gesendet] user_id={user_id}, message_id={message.id}")

            # Speichere die Audioantwort in der DB
            convo.append({"role": "assistant", "content": text_answer})
            save_message_to_db(user_id, "assistant", text_answer)
            ensure_limit_in_db(user_id, limit=MAX_CONVERSATIONS)
        else:
            # Nur Text, kein Audio
            await message.channel.send(text_answer)
            # Speichere die Textantwort in der DB
            convo = get_user_conversation(user_id)
            convo.append({"role": "assistant", "content": text_answer})
            save_message_to_db(user_id, "assistant", text_answer)
            ensure_limit_in_db(user_id, limit=MAX_CONVERSATIONS)

    except Exception as e:
        logger.error(f"[Fehler bei GPT-4o-Audio] {e}")
        await message.channel.send(f"[Fehler bei GPT-4o-Audio] {e}")

# ======================================================================
# 12. Hilfsfunktion: Ausgabe in Chunks
# ======================================================================
async def send_in_chunks(channel: discord.abc.Messageable, text: str):
    chunk_size = 2000
    for i in range(0, len(text), chunk_size):
        await channel.send(text[i:i+chunk_size])
    logger.debug(f"[Chunked Output] Gesendet: {len(text)} Zeichen in {len(text) // chunk_size + 1} Nachrichten.")

# ======================================================================
# 13. Konvertierungsfunktion: WAV zu OGG mit FFmpeg
# ======================================================================
def convert_wav_to_ogg(wav_data: bytes) -> bytes:
    """
    Konvertiert WAV-Daten zu OGG mit Opus-Codec mittels FFmpeg.
    
    Args:
        wav_data (bytes): Die ursprünglichen WAV-Audiodaten.
    
    Returns:
        bytes: Die konvertierten OGG-Audiodaten.
    
    Raises:
        Exception: Wenn die Konvertierung fehlschlägt.
    """
    try:
        ffmpeg_path = os.getenv("FFMPEG_PATH", "ffmpeg")  # Fallback auf 'ffmpeg' im PATH
        process = subprocess.Popen(
            [
                ffmpeg_path,               # Pfad zu FFmpeg
                '-i', 'pipe:0',            # Eingabe von stdin
                '-c:a', 'libopus',         # Audio-Codec Opus
                '-b:a', '96k',             # Bitrate (anpassbar)
                '-ar', '48000',            # Abtastrate 48000 Hz
                '-ac', '1',                # Mono-Kanal
                '-f', 'ogg',               # Ausgabeformat OGG
                'pipe:1'                   # Ausgabe zu stdout
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        ogg_data, stderr = process.communicate(input=wav_data)
        if process.returncode != 0:
            raise Exception(f"FFmpeg-Konvertierung fehlgeschlagen: {stderr.decode('utf-8')}")
        return ogg_data
    except FileNotFoundError:
        raise Exception("FFmpeg ist nicht installiert oder nicht im angegebenen Pfad verfügbar.")
    except Exception as e:
        raise Exception(f"Fehler bei der Konvertierung: {e}")

# ======================================================================
# 14. BOT START
# ======================================================================
async def main():
    # DB initialisieren
    init_db()
    # Lade Chatverläufe in den lokalen Cache
    global user_conversations_cache
    user_conversations_cache = load_conversations_from_db()
    logger.info("Starte Discord-Bot...")
    await bot.start(DISCORD_TOKEN)

@bot.event
async def on_ready():
    try:
        await bot.tree.sync()
        logger.info(f"Eingeloggt als: {bot.user} (ID: {bot.user.id})")
        logger.info("Bot ist bereit (Audio Beta, PDF, Bilder, Memory, /bild).")
    except Exception as e:
        logger.error(f"Fehler beim Slash-Command-Sync: {e}")

if __name__ == "__main__":
    asyncio.run(main())
