#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging
from typing import Literal, Optional
from dataclasses import dataclass, field
from datetime import datetime as dt
from base64 import b64encode
from collections import defaultdict

import discord
import httpx
import yaml
from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# ------------------------------------
# 1) CONFIG & KONSTANTEN
# ------------------------------------
def load_config(filename="config.yaml"):
    try:
        with open(filename, "r") as f:
            data = yaml.safe_load(f)
        return data if data else {}
    except Exception as e:
        logging.error(f"Fehler beim Laden von {filename}: {e}")
        return {}

cfg = load_config()

BOT_TOKEN = cfg.get("bot_token", "")
if not BOT_TOKEN:
    logging.warning("Kein bot_token in config.yaml gefunden!")

provider, model = cfg["model"].split("/", 1)
base_url = cfg["providers"][provider]["base_url"]
api_key = cfg["providers"][provider].get("api_key", "sk-???")

# Falls Vision-Features
VISION_MODEL_TAGS = ("gpt-4v", "vision", "vl", "llava")
IS_VISION = any(tag in model.lower() for tag in VISION_MODEL_TAGS)

ALLOWED_FILE_TYPES = ("image", "pdf", "text")  # => Bilder/PDFs/Text
MAX_MESSAGES = cfg.get("max_messages", 10)
MAX_TEXT = cfg.get("max_text", 1500)
USE_STREAM = False  # hier kein Streaming, da wir den gesamten Verlauf auf einmal schicken

# Optional: System-Prompt
SYSTEM_PROMPT = cfg.get("system_prompt", "")
if SYSTEM_PROMPT:
    SYSTEM_PROMPT += f"\n(Heutiges Datum: {dt.now().strftime('%Y-%m-%d')})"

# ------------------------------------
# 2) DISCORD & SPEICHER FÜR VERLÄUFE
# ------------------------------------
intents = discord.Intents.default()
intents.message_content = True

discord_client = discord.Client(intents=intents)
httpx_client = httpx.AsyncClient()

# Jede Channel-ID bekommt seine eigene Chat-Historie (Liste von {role, content})
channel_history = defaultdict(list)

# ------------------------------------
# 3) HILFSFUNKTION: ANHÄNGE => GPT-4V
# ------------------------------------
async def convert_attachments_to_gpt4v_format(msg: discord.Message):
    """
    Liest die Attachments und baut die GPT-4 Vision 'content' Einträge,
    z.B. "type=image_url" oder "type=file" für PDFs.
    """
    contents = []

    # Start: reiner Text
    text_part = msg.content.strip()
    if text_part:
        # user-Eingabetext
        contents.append({"type": "text", "text": text_part[:MAX_TEXT]})

    # Attachments
    for att in msg.attachments:
        ctype = att.content_type or ""
        try:
            # => ctype könnte "image/png", "application/pdf", "text/plain", ...
            if any(t in ctype for t in ("image/", "pdf", "text")):
                data = await att.read()
                b64_data = b64encode(data).decode("utf-8")

                if ctype.startswith("image/"):
                    # GPT-4 Vision => "image_url"
                    contents.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{ctype};base64,{b64_data}"}
                    })
                elif "pdf" in ctype:
                    # PDF => "file"
                    contents.append({
                        "type": "file",
                        "file": {"url": f"data:{ctype};base64,{b64_data}"}
                    })
                elif ctype.startswith("text/"):
                    # reinen Text-File-Inhalt an text_part anhängen?
                    # oder wir packen es als "text" ...
                    text_str = data.decode("utf-8", errors="replace")
                    # z.B. => an vorhandenen Text anhängen
                    # wenn du das lieber separat machen willst, kann man
                    # contents.append({"type": "text", "text": text_str[:MAX_TEXT]})
                    # hier mal: direkt an user text anfügen:
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

    # Falls nichts => leere Zeichenkette
    if not contents:
        return ""

    # Falls nur ein Eintrag und es "type=text" => direkt content
    if len(contents) == 1 and contents[0]["type"] == "text":
        return contents[0]["text"]  # normaler String
    return contents

# ------------------------------------
# 4) ON_MESSAGE-EVENT
# ------------------------------------
@discord_client.event
async def on_message(msg: discord.Message):
    # 1) Bot ignoriert eigene Nachrichten
    if msg.author.bot:
        return

    # 2) Die Eingabe des Users in GPT-4V-Format konvertieren
    user_content = await convert_attachments_to_gpt4v_format(msg)

    # 3) In channel_history ablegen
    channel_id = msg.channel.id
    channel_hist = channel_history[channel_id]

    # Ggf. begrenzen auf die letzten MAX_MESSAGES (Assistant+User). 
    # Hier ein simpler Trim:
    while len(channel_hist) > 2 * MAX_MESSAGES:
        channel_hist.pop(0)

    # a) system prompt, falls gewünscht (einmalig am Anfang)
    #    Du kannst den System-Prompt auch immer wieder am Ende anhängen.
    if SYSTEM_PROMPT and not channel_hist:
        channel_hist.append({"role": "system", "content": SYSTEM_PROMPT})

    # b) user message an Historie anhängen
    #    (kein "name" -> wir vermeiden die Numerik/Halluzination)
    channel_hist.append({"role": "user", "content": user_content})

    # 4) Alles an OpenAI schicken
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    # Wir geben dem Modell den gesamten Verlauf:
    kwargs = {
        "model": model,
        "messages": channel_hist,
        "stream": USE_STREAM,
        "extra_body": cfg.get("extra_api_parameters", {})
    }

    try:
        async with msg.channel.typing():
            response = await openai_client.chat.completions.create(**kwargs)

        # 5) Antwort extrahieren (stream=False => direct)
        assistant_content = response.choices[0].message.content

        # 6) Antwort in channel_history packen
        channel_hist.append({"role": "assistant", "content": assistant_content})

        # 7) An Discord schicken
        #    Falls Text zu lang -> in Blöcken
        MAX_DISCORD_LEN = 2000
        if len(assistant_content) <= MAX_DISCORD_LEN:
            await msg.reply(assistant_content)
        else:
            # stückeln
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

    except Exception as e:
        logging.exception("Fehler beim Erzeugen der Antwort")
        try:
            await msg.reply(f"Fehler: {e}")
        except:
            pass


# ------------------------------------
# 5) on_ready
# ------------------------------------
@discord_client.event
async def on_ready():
    logging.info(f"Bot eingeloggt als {discord_client.user} (ID: {discord_client.user.id})")


# ------------------------------------
# 6) main
# ------------------------------------
async def main():
    if not BOT_TOKEN:
        logging.error("Es wurde kein bot_token in config.yaml gefunden!")
        return
    await discord_client.start(BOT_TOKEN)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
