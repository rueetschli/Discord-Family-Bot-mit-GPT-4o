#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging
from datetime import datetime as dt
from collections import defaultdict
import re
import io
import urllib.parse
import json
import uuid
import base64
from typing import Optional, List, Dict, Any

import discord
from discord import app_commands
import httpx
import yaml
import aiomysql
import aioftp

from openai import AsyncOpenAI, APITimeoutError
from PIL import Image

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# -----------------------------------------------------------------------------
# 1) CONFIG
# -----------------------------------------------------------------------------
def load_config(filename="config.yaml"):
    try:
        with open(filename, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logging.error(f"Fehler beim Laden von {filename}: {e}")
        return {}

cfg = load_config()

BOT_TOKEN = cfg.get("bot_token", "")
if not BOT_TOKEN:
    logging.warning("Kein bot_token in config.yaml gefunden!")

provider, default_model = cfg["model"].split("/", 1)
base_url = cfg["providers"][provider]["base_url"]
api_key  = cfg["providers"][provider].get("api_key", "sk-???")

# --- NEU: zentrale Timeout-/Retry-Defaults (per config überschreibbar) -------
OPENAI_TIMEOUT_SEC = cfg.get("timeouts", {}).get("openai_seconds", 300)
OPENAI_RETRIES     = cfg.get("timeouts", {}).get("openai_retries", 2)
OPENAI_BACKOFFS    = cfg.get("timeouts", {}).get("openai_backoff", [2.0, 5.0, 10.0])

# -----------------------------------------------------------------------------
# 2) OPENAI CLIENT
# -----------------------------------------------------------------------------
openai_client = AsyncOpenAI(
    api_key=api_key,
    base_url=base_url,
)
DEFAULT_MODEL = default_model  # z. B. "gpt-5"

# -----------------------------------------------------------------------------
# 2.1 DB (aiomysql)
# -----------------------------------------------------------------------------
db_pool = None

async def init_db_pool():
    global db_pool
    db_config = cfg.get("database", {})
    db_pool = await aiomysql.create_pool(
        host=db_config.get("host"),
        port=db_config.get("port", 3306),
        user=db_config.get("user"),
        password=db_config.get("password"),
        db=db_config.get("name"),
        autocommit=True,
    )
    logging.info("Datenbank-Pool initialisiert.")
    # Tabellen
    async with db_pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                channel_id VARCHAR(64) NOT NULL,
                role VARCHAR(20) NOT NULL,
                content MEDIUMTEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
            """)
            logging.info("Tabelle conversation_history geprüft/erstellt.")
    async with db_pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
            CREATE TABLE IF NOT EXISTS channel_uploads (
                id INT AUTO_INCREMENT PRIMARY KEY,
                channel_id VARCHAR(64) NOT NULL,
                file_url TEXT NOT NULL,
                upload_time DATETIME DEFAULT CURRENT_TIMESTAMP
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
            """)
            logging.info("Tabelle channel_uploads geprüft/erstellt.")

async def load_history_from_db(channel_id: str):
    history = []
    async with db_pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT role, content FROM conversation_history WHERE channel_id=%s ORDER BY id",
                (channel_id,),
            )
            result = await cur.fetchall()
            for row in result:
                try:
                    content_obj = json.loads(row["content"])
                except Exception:
                    content_obj = row["content"]
                history.append({"role": row["role"], "content": content_obj})
    return history

async def save_message_to_db(channel_id: str, role: str, content_obj):
    content_str = json.dumps(content_obj, ensure_ascii=False)
    async with db_pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO conversation_history (channel_id, role, content) VALUES (%s, %s, %s)",
                (channel_id, role, content_str),
            )

async def trim_history_in_db(channel_id: str, max_messages: int):
    async with db_pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT COUNT(*) FROM conversation_history WHERE channel_id=%s",
                (channel_id,),
            )
            (count,) = await cur.fetchone()
            if count > max_messages:
                to_delete = count - max_messages
                await cur.execute(
                    "DELETE FROM conversation_history WHERE channel_id=%s ORDER BY id ASC LIMIT %s",
                    (channel_id, to_delete),
                )

async def save_upload_record(channel_id: str, file_url: str):
    async with db_pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO channel_uploads (channel_id, file_url) VALUES (%s, %s)",
                (channel_id, file_url),
            )

async def trim_uploads_in_db(channel_id: str, max_uploads: int = 20):
    async with db_pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT COUNT(*) FROM channel_uploads WHERE channel_id=%s",
                (channel_id,),
            )
            (count,) = await cur.fetchone()
            if count > max_uploads:
                to_delete = count - max_uploads
                await cur.execute(
                    "SELECT file_url FROM channel_uploads WHERE channel_id=%s ORDER BY id ASC LIMIT %s",
                    (channel_id, to_delete),
                )
                rows = await cur.fetchall()
                ftp_remote_dir = cfg.get("ftp", {}).get("remote_dir", "public_html/upload-discord-bot").strip("/")
                for row in rows:
                    public_url = row[0]
                    try:
                        filename_only = public_url.rstrip("/").split("/")[-1]
                        ftp_path = f"{ftp_remote_dir}/{filename_only}"
                        await ftp_delete(ftp_path)
                    except Exception as e:
                        logging.warning(f"Fehler beim Löschen der Datei {public_url}: {e}")
                await cur.execute(
                    "DELETE FROM channel_uploads WHERE channel_id=%s ORDER BY id ASC LIMIT %s",
                    (channel_id, to_delete),
                )

# -----------------------------------------------------------------------------
# 3) FTP-Helfer
# -----------------------------------------------------------------------------
async def _ftp_ensure_dir(client: aioftp.Client, remote_dir: str):
    parts = [p for p in remote_dir.strip("/").split("/") if p]
    path = ""
    for p in parts:
        path = f"{path}/{p}" if path else p
        try:
            await client.make_directory(path)
        except Exception:
            pass  # existiert bereits

async def ftp_upload(file_bytes: bytes, filename: str) -> str:
    ftp_config = cfg.get("ftp", {})
    host = ftp_config.get("host")
    port = ftp_config.get("port", 21)
    user = ftp_config.get("user")
    password = ftp_config.get("password")
    base_url = (ftp_config.get("base_url") or "").rstrip("/")
    remote_dir = ftp_config.get("remote_dir", "public_html/upload-discord-bot").strip("/")

    remote_path = f"{remote_dir}/{filename}"

    async with aioftp.Client.context(host, port=port, user=user, password=password) as client:
        await _ftp_ensure_dir(client, remote_dir)
        async with client.upload_stream(remote_path) as stream:
            await stream.write(file_bytes)

    public_url = f"{base_url}/{remote_dir}/{filename}"
    return public_url

async def ftp_delete(filename_or_path: str):
    ftp_config = cfg.get("ftp", {})
    host = ftp_config.get("host")
    port = ftp_config.get("port", 21)
    user = ftp_config.get("user")
    password = ftp_config.get("password")
    async with aioftp.Client.context(host, port=port, user=user, password=password) as client:
        await client.remove(filename_or_path)

# -----------------------------------------------------------------------------
# 4) DISCORD CLIENT
# -----------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
discord_client = discord.Client(intents=intents)
tree = app_commands.CommandTree(discord_client)

channel_history = defaultdict(list)

# -----------------------------------------------------------------------------
# 5) LaTeX
# -----------------------------------------------------------------------------
# Klassische Delimiter
regex_block_dollar    = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)
regex_inline_dollar   = re.compile(r'(?<!\\)\$(?!\$)(.*?)(?<!\\)\$(?!\$)', re.DOTALL)
regex_block_brackets  = re.compile(r'\\\[([\s\S]*?)\\\]', re.DOTALL)
regex_inline_paren    = re.compile(r'\\\(([\s\S]*?)\\\)', re.DOTALL)

# NEU: Code-Fences und Backticks
regex_fenced_tex      = re.compile(r'```(?:tex|latex)\s+([\s\S]*?)```', re.IGNORECASE)
regex_fenced_generic  = re.compile(r'```([\s\S]*?)```', re.DOTALL)
regex_inline_backtick = re.compile(r'`([^`]{3,})`')  # >=3 Zeichen, um Rauschen zu senken

_LATEX_TOKENS = (
    r'\frac', r'\sqrt', r'\sum', r'\int', r'\lim', r'\alpha', r'\beta', r'\gamma',
    r'\cdot', r'\rightarrow', r'\to', r'\infty', r'\over', r'\begin', r'\end',
)

def _looks_like_latex(s: str) -> bool:
    s = s.strip()
    if len(s) < 5:
        return False
    if any(tok in s for tok in _LATEX_TOKENS):
        return True
    # Heuristik: viele { } ^ _
    score = sum(ch in '{}^_' for ch in s)
    return score >= 3

def _prepare_for_codecogs(expr: str) -> str:
    """Mehrzeilige Formeln robuster machen."""
    exp = expr.strip()
    # Falls schon eine Umgebung genutzt wird, nichts tun
    if r'\begin{' in exp and r'\end{' in exp:
        return exp
    # Bei Zeilenumbrüchen/Alignment besser in aligned einbetten
    if '\n' in exp or '\\\\' in exp:
        exp = r'\begin{aligned}' + '\n' + exp + '\n' + r'\end{aligned}'
    return exp

def extract_latex_expressions(text: str):
    found = []

    # Klassische Delimiter
    found += regex_block_dollar.findall(text)
    found += regex_inline_dollar.findall(text)
    found += regex_block_brackets.findall(text)
    found += regex_inline_paren.findall(text)

    # Fenced code in ```tex / ```latex
    found += regex_fenced_tex.findall(text)

    # Generische ```...```-Blöcke, aber nur solche, die nach LaTeX aussehen
    for block in regex_fenced_generic.findall(text):
        if _looks_like_latex(block):
            found.append(block)

    # Inline-Backticks `...` mit LaTeX-Heuristik
    for inline in regex_inline_backtick.findall(text):
        if _looks_like_latex(inline):
            found.append(inline)

    # Aufbereiten, deduplizieren, leere raus
    expressions = []
    seen = set()
    for expr in found:
        e = _prepare_for_codecogs(expr)
        if e and e not in seen:
            seen.add(e)
            expressions.append(e)
    return expressions

async def fetch_latex_png(latex_code: str) -> bytes:
    # etwas groesser und weiss hinterlegt fuer Lesbarkeit
    safe_expr = urllib.parse.quote(latex_code, safe='')
    url = f"https://latex.codecogs.com/png.latex?\\dpi{{170}}\\bg_white\\large {safe_expr}"
    async with httpx.AsyncClient() as c:
        resp = await c.get(url, timeout=30.0)
        resp.raise_for_status()
        return resp.content

async def render_latex_image(latex_expressions):
    if not latex_expressions:
        return None
    images = []
    for expr in latex_expressions:
        try:
            png_data = await fetch_latex_png(expr)
            img = Image.open(io.BytesIO(png_data)).convert("RGBA")
            images.append(img)
        except Exception as e:
            logging.warning(f"Fehler beim Rendern von LaTeX: {e} , Ausdruck: {expr[:80]}...")
    if not images:
        return None
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images) + 8 * (len(images) - 1)
    merged = Image.new("RGBA", (max_width, total_height), (255, 255, 255, 0))
    y_offset = 0
    for img in images:
        merged.paste(img, (0, y_offset))
        y_offset += img.height + 8
    buf = io.BytesIO()
    merged.save(buf, format="PNG")
    buf.seek(0)
    return buf


# -----------------------------------------------------------------------------
# 6) ATTACHMENTS
# -----------------------------------------------------------------------------
MAX_TEXT = cfg.get("max_text", 1500)

async def build_message_content(msg: discord.Message):
    contents: List[Dict[str, Any]] = []

    # Text
    text_part = msg.content.strip()
    if text_part:
        contents.append({"type": "text", "text": text_part[:MAX_TEXT]})

    # Attachments
    for att in msg.attachments:
        ctype = att.content_type or ""
        try:
            data = await att.read()

            # PDF -> Files API
            if "pdf" in ctype.lower():
                try:
                    uploaded_file = await openai_client.files.create(
                        file=(att.filename, data, "application/pdf"),
                        purpose="user_data"
                    )
                    contents.append({
                        "type": "file",
                        "file": {"file_id": uploaded_file.id}
                    })
                except Exception as e:
                    logging.warning(f"Fehler beim Upload des PDF {att.filename}: {e}")
                    contents.append({"type": "text", "text": f"[PDF-Upload fehlgeschlagen: {att.filename}]"})

            # Image -> Base64 ans Modell, URL nur als Referenz
            elif ctype.startswith("image/"):
                try:
                    image = Image.open(io.BytesIO(data))
                    image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                    if image.mode in ("RGBA", "P"):
                        image = image.convert("RGB")
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=90)
                    buffer.seek(0)
                    file_bytes = buffer.getvalue()
                    b64 = base64.b64encode(file_bytes).decode("ascii")

                    unique_filename = f"{dt.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}.jpg"
                    public_url = await ftp_upload(file_bytes, unique_filename)

                    contents.append({
                        "type": "image_base64",
                        "b64": b64,
                        "mime_type": "image/jpeg",
                        "url": public_url  # informativ
                    })
                    await save_upload_record(str(msg.channel.id), public_url)
                    await trim_uploads_in_db(str(msg.channel.id))

                except Exception as e:
                    logging.warning(f"Fehler beim Verarbeiten des Bildes {att.filename}: {e}")
                    contents.append({"type": "text", "text": "[Bild-Verarbeitung fehlgeschlagen]"})

            # Audio -> STT
            elif ctype.startswith("audio/"):
                try:
                    audio_file = io.BytesIO(data)
                    audio_file.name = att.filename
                    transcription = await openai_client.audio.transcriptions.create(
                        model="gpt-4o-transcribe",
                        file=audio_file,
                        response_format="text"
                    )
                    text_result = getattr(transcription, "text", None)
                    if not text_result:
                        text_result = str(transcription)
                    contents.append({"type": "text", "text": text_result})
                except Exception as e:
                    logging.warning(f"Fehler beim Transkribieren {att.filename}: {e}")
                    contents.append({"type": "text", "text": "[Sprachnachricht konnte nicht transkribiert werden]"})

            else:
                logging.info(f"Attachment {att.filename} mit unbekanntem Typ {ctype}")
                contents.append({"type": "text", "text": f"[Attachment: {att.filename}, Typ {ctype}]"})

        except Exception as e:
            logging.warning(f"Fehler beim Lesen von {att.filename}: {e}")

    return contents

# -----------------------------------------------------------------------------
# 7) TTS
# -----------------------------------------------------------------------------
async def generate_tts_audio(text: str) -> Optional[io.BytesIO]:
    try:
        audio_response = await openai_client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="ash",
            input=text,
            response_format="mp3"
        )
    except Exception as e:
        logging.warning(f"TTS Generierung fehlgeschlagen: {e}")
        return None
    try:
        audio_bytes = audio_response.content
    except Exception as e:
        logging.warning(f"Fehler beim Konvertieren der Audio-Antwort in Bytes: {e}")
        return None
    return io.BytesIO(audio_bytes)

# -----------------------------------------------------------------------------
# 8) BOT-LOGIK
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = cfg.get("system_prompt", "")
if SYSTEM_PROMPT:
    SYSTEM_PROMPT += f"\n(Heutiges Datum: {dt.now().strftime('%Y-%m-%d')})"

MAX_MESSAGES = cfg.get("max_messages", 10)

@discord_client.event
async def on_ready():
    logging.info(f"Bot eingeloggt als {discord_client.user} (ID: {discord_client.user.id})")
    status_message = cfg.get("status_message")
    if status_message:
        activity = discord.Activity(type=discord.ActivityType.watching, name=status_message)
        await discord_client.change_presence(activity=activity)
    try:
        await tree.sync()
        logging.info("Slash Commands wurden synchronisiert.")
    except Exception as e:
        logging.error(f"Fehler beim Synchronisieren der Slash Commands: {e}")

@discord_client.event
async def on_message(msg: discord.Message):
    if msg.author.bot:
        return
    asyncio.create_task(handle_normal_message(msg))

def extract_text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for c in content:
            if isinstance(c, dict):
                t = c.get("type")
                if t == "text":
                    text_parts.append(c.get("text", ""))
                elif t == "image_base64":
                    url = c.get("url", "(ohne URL)")
                    text_parts.append(f"[Bild: {url}]")
                elif t == "image_url":
                    img = c.get("image_url", {})
                    url = img.get("url") if isinstance(img, dict) else str(img)
                    text_parts.append(f"[Bild: {url}]")
                elif t == "file":
                    text_parts.append(f"[Datei: {c.get('file', {}).get('file_id', 'keine Datei')}]")
                else:
                    text_parts.append(str(c))
            else:
                text_parts.append(str(c))
        return "\n".join(text_parts)
    return str(content)

def map_content_for_responses(role: str, raw_content):
    """
    Mappt auf Responses-Eingabe.
    'system' wird NICHT gemappt, den Prompt geben wir über 'instructions'.
    """
    if role == "assistant":
        text = extract_text_from_content(raw_content)
        return {"role": "assistant", "content": [{"type": "output_text", "text": text}]}

    if role == "user":
        blocks: List[Dict[str, Any]] = []
        if isinstance(raw_content, list):
            for c in raw_content:
                if not isinstance(c, dict):
                    blocks.append({"type": "input_text", "text": str(c)})
                    continue
                t = c.get("type")
                if t == "text":
                    blocks.append({"type": "input_text", "text": c.get("text", "")})
                elif t == "image_base64":
                    data = c.get("b64")
                    mime = c.get("mime_type", "image/jpeg")
                    if data:
                        # Responses-API erwartet image_url als String; Base64 daher als Data-URL
                        blocks.append({
                            "type": "input_image",
                            "image_url": f"data:{mime};base64,{data}",
                            "detail": "high",
                        })
                elif t == "image_url":
                    img = c.get("image_url", {})
                    url = img.get("url") if isinstance(img, dict) else img
                    if url:
                        blocks.append({"type": "input_image", "image_url": url, "detail": "high"})
                elif t == "file":
                    fid = c.get("file", {}).get("file_id")
                    if fid:
                        blocks.append({"type": "input_file", "file_id": fid})
                else:
                    blocks.append({"type": "input_text", "text": str(c)})
        else:
            text = extract_text_from_content(raw_content)
            blocks.append({"type": "input_text", "text": text})
        return {"role": "user", "content": blocks}

    return None

def build_responses_input_from_history_full(channel_hist):
    items = []
    for item in channel_hist:
        mapped = map_content_for_responses(item["role"], item["content"])
        if mapped:
            items.append(mapped)
    return items

def _chunk_text_for_discord(text: str, limit: int = 1900):
    parts = []
    i = 0
    n = len(text)
    while i < n:
        cut = min(n, i + limit)
        nl = text.rfind("\n", i, cut)
        if nl != -1 and nl > i + 200:
            parts.append(text[i:nl])
            i = nl + 1
            continue
        sp = text.rfind(" ", i, cut)
        if sp != -1 and sp > i + 200:
            parts.append(text[i:sp])
            i = sp + 1
            continue
        parts.append(text[i:cut])
        i = cut
    return parts

async def send_long_message(channel: discord.TextChannel, text: str):
    for chunk in _chunk_text_for_discord(text):
        await channel.send(chunk)

async def send_long_followup(interaction: discord.Interaction, text: str):
    for chunk in _chunk_text_for_discord(text):
        await interaction.followup.send(chunk)

# --- NEU: generischer Retry-Wrapper für OpenAI-Requests ----------------------
async def _create_with_retry(factory_coro):
    last_err = None
    for attempt in range(OPENAI_RETRIES + 1):
        try:
            return await factory_coro()
        except (APITimeoutError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            last_err = e
            if attempt < OPENAI_RETRIES:
                await asyncio.sleep(OPENAI_BACKOFFS[min(attempt, len(OPENAI_BACKOFFS)-1)])
                continue
            raise

async def _responses_call(input_messages, instructions: Optional[str], *, timeout: float = None):
    timeout = timeout or OPENAI_TIMEOUT_SEC
    client_long = openai_client.with_options(timeout=timeout)
    return await _create_with_retry(lambda: client_long.responses.create(
        model=DEFAULT_MODEL,
        input=input_messages,
        instructions=instructions or None,
        text={
            "verbosity": cfg.get("search", {}).get("verbosity", "high"),
            "format": {"type": "text"}
        },
        reasoning={"effort": cfg.get("search", {}).get("reasoning_effort", "medium")},
        **cfg.get("extra_api_parameters", {})
    ))

def _extract_response_text(resp) -> str:
    txt = getattr(resp, "output_text", None) or ""
    if txt:
        return txt
    try:
        parts = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for ct in getattr(item, "content", []) or []:
                    t = getattr(ct, "text", None) or getattr(ct, "output_text", None)
                    if t:
                        parts.append(t)
        return "\n".join(parts).strip()
    except Exception:
        return ""

async def handle_normal_message(msg: discord.Message):
    channel_id = str(msg.channel.id)
    if channel_id not in channel_history or not channel_history[channel_id]:
        history = await load_history_from_db(channel_id)
        channel_history[channel_id] = history
    if SYSTEM_PROMPT and not any(x["role"] == "system" for x in channel_history[channel_id]):
        channel_history[channel_id].append({"role": "system", "content": SYSTEM_PROMPT})
        await save_message_to_db(channel_id, "system", SYSTEM_PROMPT)

    user_content = await build_message_content(msg)
    channel_history[channel_id].append({"role": "user", "content": user_content})
    await save_message_to_db(channel_id, "user", user_content)

    if len(channel_history[channel_id]) > 2 * MAX_MESSAGES:
        channel_history[channel_id] = channel_history[channel_id][-2 * MAX_MESSAGES:]
        await trim_history_in_db(channel_id, 2 * MAX_MESSAGES)

    input_messages = build_responses_input_from_history_full(channel_history[channel_id])

    img_cnt = sum(
        1 for m in input_messages if m.get("role") == "user"
        for b in m.get("content", []) if b.get("type") == "input_image"
    )
    txt_cnt = sum(
        1 for m in input_messages if m.get("role") == "user"
        for b in m.get("content", []) if b.get("type") == "input_text"
    )
    logging.info(f"Responses-Input: {img_cnt} input_image, {txt_cnt} input_text")

    try:
        async with msg.channel.typing():
            resp = await _responses_call(input_messages, SYSTEM_PROMPT)
            assistant_content = _extract_response_text(resp) or "(Keine Antwort)"

        channel_history[channel_id].append({"role": "assistant", "content": assistant_content})
        await save_message_to_db(channel_id, "assistant", assistant_content)
        await send_long_message(msg.channel, assistant_content)

        if not assistant_content.startswith("(Keine Antwort"):
            tts_audio = await generate_tts_audio(assistant_content)
            if tts_audio:
                file = discord.File(fp=tts_audio, filename="tts.mp3")
                await msg.channel.send(file=file)

        latex_expressions = extract_latex_expressions(assistant_content)
        if latex_expressions:
            try:
                latex_image_buffer = await render_latex_image(latex_expressions)
                if latex_image_buffer:
                    file = discord.File(fp=latex_image_buffer, filename="latex.png")
                    await msg.channel.send(file=file)
            except Exception as e:
                logging.warning(f"Fehler beim Rendern von LaTeX: {e}")

    except Exception as e:
        logging.exception("Fehler beim Erzeugen der Antwort (normaler Chat)")
        await msg.reply(f"Fehler: {e}")

# -----------------------------------------------------------------------------
# 9) SLASH COMMANDS
# -----------------------------------------------------------------------------
@tree.command(name="help", description="Zeigt die verfügbaren Funktionen an.")
async def help_command(interaction: discord.Interaction):
    txt = (
        "**Verfügbare Slash-Commands:**\n\n"
        "1. `/suche <frage>` – Nutzt GPT-5 mit integrierter Websuche.\n"
        "2. `/bild <prompt> <format>` – Erzeugt ein Bild mit gpt-image-1.\n"
        "3. `/vergessen` – Löscht die Kanal-Historie.\n\n"
        f"Zusätzlich kannst du normal chatten, Standardmodell: {DEFAULT_MODEL}.\n"
        "Ich splitte lange Antworten automatisch in mehrere Discord-Nachrichten."
    )
    await interaction.response.send_message(txt, ephemeral=True)

@tree.command(name="suche", description="Frage GPT-5 mit Websuche.")
@app_commands.describe(prompt="Worüber soll gesucht werden?")
async def suche_command(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()
    channel_id = str(interaction.channel_id)
    if channel_id not in channel_history or not channel_history[channel_id]:
        channel_history[channel_id] = await load_history_from_db(channel_id)

    # Verlauf ergänzen
    user_block = [{"type": "text", "text": prompt}]
    channel_history[channel_id].append({"role": "user", "content": user_block})
    await save_message_to_db(channel_id, "user", user_block)

    input_messages = build_responses_input_from_history_full(channel_history[channel_id])

    # System-Prompt für den /suche-Kontext überschreiben
    web_instructions = (
        (SYSTEM_PROMPT or "")
        + "\n\n[Websuche-Modus] Du befindest dich im Slash-Command /suche. "
          "Führe aktiv eine Websuche durch und beantworte die Frage direkt. "
          "Erwähne /suche nicht. Liefere zuerst eine kurze Zusammenfassung, "
          "dann Details. Hänge eine Quellenliste mit Titeln und URLs an. "
          "Wenn nichts Relevantes gefunden wird, sage das explizit und erkläre, "
          "wie man die Suche verfeinern kann."
    )

    # Location & Tool aus config lesen, mit Defaults
    loc = cfg.get("search", {}).get("user_location", {
        "type": "approximate",
        "country": "CH",
        "region": "Solothurn",
        "city": "Rüttenen",
    })
    tools = [{
        "type": "web_search_preview",
        "user_location": loc,
        "search_context_size": cfg.get("search", {}).get("search_context_size", "high"),
    }]

    try:
        async with interaction.channel.typing():
            client_long = openai_client.with_options(timeout=OPENAI_TIMEOUT_SEC)
            resp = await _create_with_retry(lambda: client_long.responses.create(
                model=DEFAULT_MODEL,
                input=input_messages,
                instructions=web_instructions,
                tools=tools,
                text={
                    "verbosity": cfg.get("search", {}).get("verbosity", "high"),
                    "format": {"type": "text"}
                },
                reasoning={"effort": cfg.get("search", {}).get("reasoning_effort", "high")},
                store=True
            ))

        content = _extract_response_text(resp) or ""

        # Fallback, falls das Modell trotzdem Befehle empfiehlt
        if "slash-command" in content.lower() or "/suche" in content.lower():
            resp = await _create_with_retry(lambda: client_long.responses.create(
                model=DEFAULT_MODEL,
                input=input_messages,
                instructions=web_instructions + " Antworte jetzt ohne Hinweis auf Befehle.",
                tools=tools,
                text={"verbosity": "high", "format": {"type": "text"}},
                reasoning={"effort": "high"},
                store=True
            ))
            content = _extract_response_text(resp) or content

    except Exception as e:
        logging.exception("Fehler bei /suche")
        await interaction.followup.send(f"Fehler bei der Websuche: {e}")
        return

    if not content.strip():
        content = "(Keine Antwort, Websuche lieferte kein verwertbares Resultat.)"

    channel_history[channel_id].append({"role": "assistant", "content": content})
    await save_message_to_db(channel_id, "assistant", content)
    await send_long_followup(interaction, content)

    latex_expressions = extract_latex_expressions(content)
    if latex_expressions:
        try:
            latex_image_buffer = await render_latex_image(latex_expressions)
            if latex_image_buffer:
                file = discord.File(fp=latex_image_buffer, filename="latex.png")
                await interaction.followup.send(file=file)
        except Exception as e:
            logging.warning(f"Fehler beim Rendern von LaTeX: {e}")

@tree.command(name="bild", description="Erzeuge ein Bild mit gpt-image-1.")
@app_commands.describe(
    prompt="Was soll auf dem Bild zu sehen sein?",
    format="Seitenverhältnis: quadratisch, hoch oder breit"
)
async def bild_command(interaction: discord.Interaction, prompt: str, format: str):
    await interaction.response.defer()
    fmt = format.lower()
    if fmt == "quadratisch":
        size = "1024x1024"
    elif fmt == "hoch":
        size = "1024x1536"
    elif fmt == "breit":
        size = "1536x1024"
    else:
        await interaction.followup.send("Bitte wähle 'quadratisch', 'hoch' oder 'breit'.")
        return

    try:
        async with interaction.channel.typing():
            response = await openai_client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                n=1,
                quality="high",
                size=size,
            )
    except Exception as e:
        logging.exception("Fehler bei /bild")
        await interaction.followup.send(f"Fehler beim Bilderzeugen: {e}")
        return

    if not response.data:
        await interaction.followup.send("Keine Bilddaten erhalten.")
        return

    b64_str = response.data[0].b64_json
    try:
        img_bytes = base64.b64decode(b64_str)
    except Exception:
        await interaction.followup.send("Fehler beim Decodieren des Bilds.")
        return

    file = discord.File(io.BytesIO(img_bytes), filename="bild.png")
    await interaction.followup.send("Hier dein Bild:", file=file)

    channel_id = str(interaction.channel_id)
    channel_hist = channel_history[channel_id]
    if SYSTEM_PROMPT and not any(x["role"] == "system" for x in channel_hist):
        channel_hist.append({"role": "system", "content": SYSTEM_PROMPT})
    channel_hist.append({"role": "user", "content": f"(Bild) {prompt} (Format: {format})"})
    channel_hist.append({"role": "assistant", "content": "Bild generiert und als Datei gesendet."})
    await save_message_to_db(channel_id, "user", f"(Bild) {prompt} (Format: {format})")
    await save_message_to_db(channel_id, "assistant", "Bild generiert und als Datei gesendet.")

@tree.command(name="vergessen", description="Löscht den gesamten Konversationsverlauf dieses Kanals.")
async def vergessen_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    channel_id = str(interaction.channel_id)
    try:
        async with db_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM conversation_history WHERE channel_id = %s", (channel_id,))
                logging.info(f"Konversationsverlauf für Kanal {channel_id} gelöscht.")
                await cur.execute("DELETE FROM channel_uploads WHERE channel_id = %s", (channel_id,))
                logging.info(f"Upload-Records für Kanal {channel_id} gelöscht.")
        channel_history[channel_id] = []
        await interaction.followup.send("Der gesamte Konversationsverlauf dieses Kanals wurde gelöscht.", ephemeral=True)
    except Exception as e:
        logging.exception("Fehler beim Löschen des Verlaufs")
        await interaction.followup.send(f"Fehler beim Löschen des Verlaufs: {e}", ephemeral=True)

# -----------------------------------------------------------------------------
# 10) START
# -----------------------------------------------------------------------------
async def main():
    if not BOT_TOKEN:
        logging.error("Kein bot_token in config.yaml gefunden!")
        return
    await init_db_pool()
    await discord_client.start(BOT_TOKEN)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
