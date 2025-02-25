#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging
from datetime import datetime as dt
from base64 import b64encode
from collections import defaultdict
import re
import io
import urllib.parse

import discord
import httpx
import yaml
import pdfplumber
import matplotlib.pyplot as plt

# Wichtig: Aus dem neuen SDK ab v1.0.0
from openai import AsyncOpenAI

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

##############################################################################
# 1) CONFIG LADEN
##############################################################################
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

provider, model = cfg["model"].split("/", 1)
base_url = cfg["providers"][provider]["base_url"]
api_key  = cfg["providers"][provider].get("api_key", "sk-???")

##############################################################################
# 2) OPENAI-CLIENT (AsyncOpenAI)
##############################################################################
# Kein openai.ChatCompletion mehr!
# Ab v1.0.0 nutzt man den eigenständigen Client.
openai_client = AsyncOpenAI(
    api_key=api_key,
    base_url=base_url,
    # Falls Du weitere Optionen brauchst, z.B. default_headers, proxies etc.
    # default_headers={"x-my-header": "123"},
    # proxies={"https": "http://proxy:8080"},
)

MAX_MESSAGES = cfg.get("max_messages", 10)
MAX_TEXT = cfg.get("max_text", 1500)

SYSTEM_PROMPT = cfg.get("system_prompt", "")
if SYSTEM_PROMPT:
    SYSTEM_PROMPT += f"\n(Heutiges Datum: {dt.now().strftime('%Y-%m-%d')})"

##############################################################################
# 3) DISCORD CLIENT
##############################################################################
intents = discord.Intents.default()
intents.message_content = True
discord_client = discord.Client(intents=intents)

# Speichert pro Channel die Chat-Historie
channel_history = defaultdict(list)

##############################################################################
# 4) LATEX-ERKENNUNG UND RENDERING VIA CODECOGS
##############################################################################
# In Python 3.12 sind Backslashes streng, wir verwenden doppelte oder Regex.
# $$...$$
regex_block_dollar    = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)
# $...$ (kein doppeltes $$)
regex_inline_dollar   = re.compile(r'(?<!\\)\$(?!\$)(.*?)(?<!\\)\$(?!\$)', re.DOTALL)
# \[...\]
regex_block_brackets  = re.compile(r'\\\[([\s\S]*?)\\\]', re.DOTALL)
# \(...\)
regex_inline_paren    = re.compile(r'\\\(([\s\S]*?)\\\)', re.DOTALL)

def extract_latex_expressions(text: str):
    """
    Sucht LaTeX-Ausdrücke in:
      - $$ ... $$
      - $ ... $
      - \[ ... \]
      - \( ... \)
    Gibt Liste mit allen Ausdrücken (duplikatfrei, getrimmt) zurück.
    """
    found = []
    found += regex_block_dollar.findall(text)
    found += regex_inline_dollar.findall(text)
    found += regex_block_brackets.findall(text)
    found += regex_inline_paren.findall(text)

    # Duplikate entfernen und Leerzeichen trimmen
    expressions = list({expr.strip() for expr in found if expr.strip()})
    return expressions

async def fetch_latex_png(latex_code: str) -> bytes:
    """
    Rendert LaTeX-Code via CodeCogs (externer Dienst).
    """
    safe_expr = urllib.parse.quote(latex_code, safe='')
    # \dpi{150} = mittlere Auflösung, \large = vergrösserte Schrift
    url = f"https://latex.codecogs.com/png.latex?\\dpi{{150}}\\bg_white\\large {safe_expr}"

    async with httpx.AsyncClient() as c:
        resp = await c.get(url)
        resp.raise_for_status()
        return resp.content

async def render_latex_image(latex_expressions):
    """
    Ruft CodeCogs für jeden Ausdruck auf und packt alle Einzelergebnisse
    in ein einziges PNG (untereinander).
    """
    if not latex_expressions:
        return None

    images = []
    for expr in latex_expressions:
        try:
            png_data = await fetch_latex_png(expr)
            img = Image.open(io.BytesIO(png_data)).convert("RGBA")
            images.append(img)
        except Exception as e:
            logging.warning(f"Fehler beim Laden gerenderter LaTeX-Grafik für '{expr}': {e}")

    if not images:
        return None

    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)
    merged = Image.new("RGBA", (max_width, total_height), (255, 255, 255, 0))

    y_offset = 0
    for img in images:
        merged.paste(img, (0, y_offset))
        y_offset += img.height

    buf = io.BytesIO()
    merged.save(buf, format="PNG")
    buf.seek(0)
    return buf

##############################################################################
# 5) PDF-EXTRAKTION (pdfplumber)
##############################################################################
async def convert_attachments_to_gpt4v_format(msg: discord.Message):
    contents = []
    text_part = msg.content.strip()
    if text_part:
        contents.append({"type": "text", "text": text_part[:MAX_TEXT]})

    for att in msg.attachments:
        ctype = att.content_type or ""
        try:
            data = await att.read()
            if any(t in ctype for t in ("image/", "pdf", "text")):
                if ctype.startswith("image/"):
                    contents.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{ctype};base64,{b64encode(data).decode('utf-8')}"}
                    })
                elif "pdf" in ctype:
                    try:
                        with pdfplumber.open(io.BytesIO(data)) as pdf:
                            text_str = ""
                            for page in pdf.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text_str += page_text + "\n"
                        text_str = text_str.strip()
                        if text_str:
                            if contents and contents[-1]["type"] == "text":
                                contents[-1]["text"] += f"\n{text_str[:MAX_TEXT]}"
                            else:
                                contents.append({"type": "text", "text": text_str[:MAX_TEXT]})
                        else:
                            logging.info(f"Kein Text im PDF {att.filename} gefunden.")
                    except Exception as e:
                        logging.warning(f"Fehler beim Extrahieren des Textes aus {att.filename}: {e}")
                elif ctype.startswith("text/"):
                    text_str = data.decode("utf-8", errors="replace")
                    if contents and contents[-1]["type"] == "text":
                        contents[-1]["text"] += f"\n{text_str}"
                    else:
                        contents.append({"type": "text", "text": text_str[:MAX_TEXT]})
                else:
                    logging.info(f"Unbekannter Dateityp: {ctype}")
            else:
                logging.info(f"Attachment {att.filename} mit unzulässigem Typ {ctype}")
        except Exception as e:
            logging.warning(f"Fehler beim Lesen von {att.filename}: {e}")

    if not contents:
        return ""
    if len(contents) == 1 and contents[0]["type"] == "text":
        return contents[0]["text"]
    return contents

##############################################################################
# 6) DISCORD-LOGIK
##############################################################################
@discord_client.event
async def on_ready():
    logging.info(f"Bot eingeloggt als {discord_client.user} (ID: {discord_client.user.id})")

@discord_client.event
async def on_message(msg: discord.Message):
    if msg.author.bot:
        return

    user_content = await convert_attachments_to_gpt4v_format(msg)
    channel_id = msg.channel.id
    channel_hist = channel_history[channel_id]

    # Verlauf begrenzen
    while len(channel_hist) > 2 * MAX_MESSAGES:
        channel_hist.pop(0)

    # Einmalig System-Prompt
    if SYSTEM_PROMPT and not channel_hist:
        channel_hist.append({"role": "system", "content": SYSTEM_PROMPT})

    channel_hist.append({"role": "user", "content": user_content})

    # Baue die Parameter für den AsyncOpenAI-Aufruf
    kwargs = {
        "model": model,
        "messages": channel_hist,
        **cfg.get("extra_api_parameters", {})
    }

    try:
        async with msg.channel.typing():
            # Asynchroner Aufruf via AsyncOpenAI-Client
            response = await openai_client.chat.completions.create(**kwargs)

        assistant_content = response.choices[0].message.content
        channel_hist.append({"role": "assistant", "content": assistant_content})

        # LaTeX erkennen
        latex_expressions = extract_latex_expressions(assistant_content)
        logging.info(f"LaTeX-Ausdrücke: {latex_expressions}")

        # Antwort stückeln, falls > 2000 Zeichen
        MAX_DISCORD_LEN = 2000
        if len(assistant_content) <= MAX_DISCORD_LEN:
            await msg.reply(assistant_content)
        else:
            lines = assistant_content.split("\n")
            chunk = ""
            for line in lines:
                if len(chunk) + len(line) + 1 > MAX_DISCORD_LEN:
                    await msg.channel.send(chunk)
                    chunk = line + "\n"
                else:
                    chunk += line + "\n"
            if chunk:
                await msg.channel.send(chunk)

        # Falls LaTeX, rendere Bild via CodeCogs
        if latex_expressions:
            try:
                latex_image_buffer = await render_latex_image(latex_expressions)
                if latex_image_buffer:
                    file = discord.File(fp=latex_image_buffer, filename="latex.png")
                    await msg.channel.send(file=file)
                else:
                    await msg.channel.send("Konnte kein LaTeX-Bild erzeugen (evtl. externer API-Fehler?).")
            except Exception as e:
                logging.warning(f"Fehler beim Rendern von LaTeX: {e}")
                await msg.channel.send("LaTeX konnte nicht in ein Bild gerendert werden (CodeCogs-Dienst?).")

    except Exception as e:
        logging.exception("Fehler beim Erzeugen der Antwort")
        try:
            await msg.reply(f"Fehler: {e}")
        except:
            pass

async def main():
    if not BOT_TOKEN:
        logging.error("Kein bot_token in config.yaml gefunden!")
        return
    await discord_client.start(BOT_TOKEN)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
