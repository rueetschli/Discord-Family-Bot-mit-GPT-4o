#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging
from datetime import datetime as dt
from collections import defaultdict
import re
import io
import urllib.parse

import discord
from discord import app_commands
import httpx
import yaml
import matplotlib.pyplot as plt

from openai import AsyncOpenAI
from PIL import Image
from base64 import b64encode

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

provider, default_model = cfg["model"].split("/", 1)
base_url = cfg["providers"][provider]["base_url"]
api_key  = cfg["providers"][provider].get("api_key", "sk-???")

##############################################################################
# 2) OPENAI-CLIENT (AsyncOpenAI)
##############################################################################
openai_client = AsyncOpenAI(
    api_key=api_key,
    base_url=base_url,
)

DEFAULT_MODEL = default_model  # z.B. "gpt-4o"

##############################################################################
# 3) DISCORD CLIENT + APP COMMANDS
##############################################################################
intents = discord.Intents.default()
intents.message_content = True
discord_client = discord.Client(intents=intents)
tree = app_commands.CommandTree(discord_client)

# Speichert pro Channel die Chat-Historie
channel_history = defaultdict(list)

##############################################################################
# 4) LATEX-ERKENNUNG UND RENDERING VIA CODECOGS
##############################################################################
regex_block_dollar    = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)
regex_inline_dollar   = re.compile(r'(?<!\\)\$(?!\$)(.*?)(?<!\\)\$(?!\$)', re.DOTALL)
regex_block_brackets  = re.compile(r'\\\[([\s\S]*?)\\\]', re.DOTALL)
regex_inline_paren    = re.compile(r'\\\(([\s\S]*?)\\\)', re.DOTALL)

def extract_latex_expressions(text: str):
    found = []
    found += regex_block_dollar.findall(text)
    found += regex_inline_dollar.findall(text)
    found += regex_block_brackets.findall(text)
    found += regex_inline_paren.findall(text)
    expressions = list({expr.strip() for expr in found if expr.strip()})
    return expressions

async def fetch_latex_png(latex_code: str) -> bytes:
    safe_expr = urllib.parse.quote(latex_code, safe='')
    url = f"https://latex.codecogs.com/png.latex?\\dpi{{150}}\\bg_white\\large {safe_expr}"
    async with httpx.AsyncClient() as c:
        resp = await c.get(url)
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
# 5) PDF-EXTRAKTION
##############################################################################
MAX_TEXT = cfg.get("max_text", 1500)

async def convert_attachments_to_gpt4v_format(msg: discord.Message):
    """
    Liest Attachments (Bilder, PDFs, Text) aus und wandelt sie in GPT-4Vision-ähnliches
    Format um. (Falls du es brauchst)
    """
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
                        import pdfplumber
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
# 6) DISCORD-BOT-LOGIK
##############################################################################
SYSTEM_PROMPT = cfg.get("system_prompt", "")
if SYSTEM_PROMPT:
    SYSTEM_PROMPT += f"\n(Heutiges Datum: {dt.now().strftime('%Y-%m-%d')})"

MAX_MESSAGES = cfg.get("max_messages", 10)

@discord_client.event
async def on_ready():
    logging.info(f"Bot eingeloggt als {discord_client.user} (ID: {discord_client.user.id})")
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

async def handle_normal_message(msg: discord.Message):
    """
    Normaler Chat, der die Chat-Completions-API mit 'model=DEFAULT_MODEL' nutzt.
    """
    channel_id = msg.channel.id
    channel_hist = channel_history[channel_id]

    if SYSTEM_PROMPT and not channel_hist:
        channel_hist.append({"role": "system", "content": SYSTEM_PROMPT})

    user_content = await convert_attachments_to_gpt4v_format(msg)
    channel_hist.append({"role": "user", "content": user_content})

    # Älteste entfernen
    while len(channel_hist) > 2 * MAX_MESSAGES:
        channel_hist.pop(0)

    kwargs = {
        "model": DEFAULT_MODEL,
        "messages": channel_hist,
        **cfg.get("extra_api_parameters", {})
    }

    try:
        async with msg.channel.typing():
            response = await openai_client.chat.completions.create(**kwargs)

        assistant_content = response.choices[0].message.content
        channel_hist.append({"role": "assistant", "content": assistant_content})

        await send_long_message(msg.channel, assistant_content)

        # LaTeX
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

async def send_long_message(channel: discord.TextChannel, text: str):
    MAX_DISCORD_LEN = 2000
    if len(text) <= MAX_DISCORD_LEN:
        await channel.send(text)
    else:
        lines = text.split("\n")
        chunk = ""
        for line in lines:
            if len(chunk) + len(line) + 1 > MAX_DISCORD_LEN:
                await channel.send(chunk)
                chunk = line + "\n"
            else:
                chunk += line + "\n"
        if chunk:
            await channel.send(chunk)

##############################################################################
# 7) SLASH COMMANDS
##############################################################################
@tree.command(name="help", description="Zeigt die verfügbaren Funktionen an.")
async def help_command(interaction: discord.Interaction):
    txt = (
        "**Verfügbare Slash-Commands:**\n\n"
        "1. `/suche <frage>` – Nutzt GPT-4o mit Websuche.\n"
        "2. `/denken <prompt>` – Nutzt das Modell o3-mini für schnelles Reasoning.\n"
        "3. `/bild <prompt> <format>` – Erzeugt ein Bild mit DALL-E 3.\n\n"
        f"Zusätzlich kann man normal chatten (Modell: {DEFAULT_MODEL})."
    )
    await interaction.response.send_message(txt, ephemeral=True)

##############################################################################
# HILFSFUNKTIONEN FÜR /SUCHEN UND /DENKEN (RESPONSES-API)
##############################################################################
def extract_text_from_content(content) -> str:
    """
    content kann entweder ein String oder eine Liste von Dicts sein.
    Diese Funktion extrahiert reinen Text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for c in content:
            if c.get("type") == "text":
                text_parts.append(c.get("text", ""))
            elif c.get("type") == "image_url":
                text_parts.append("[Bild-Anhang]")
            else:
                text_parts.append(str(c))
        return "\n".join(text_parts)
    return str(content)

def build_responses_input_from_history(channel_hist, new_prompt=None):
    """
    Baut ein Array für 'input=[...]' nach dem Schema des Playground-Beispiels:
      [
        {
          "role": "user"|"assistant",
          "content": [
            {
              "type": "input_text"|"output_text",
              "text": "..."
            }
          ]
        },
        ...
      ]

    new_prompt (falls != None) wird als neue user-Nachricht angehängt.
    """
    input_array = []

    for item in channel_hist:
        role = item["role"]
        raw_text = extract_text_from_content(item["content"])
        if not raw_text.strip():
            continue

        if role == "assistant":
            input_array.append({
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": raw_text}
                ]
            })
        elif role == "user":
            input_array.append({
                "role": "user",
                "content": [
                    {"type": "input_text", "text": raw_text}
                ]
            })
        else:
            # "system" o.Ä. überspringen oder ggf. als user?
            pass

    # Neue user-Nachricht
    if new_prompt:
        input_array.append({
            "role": "user",
            "content": [
                {"type": "input_text", "text": new_prompt}
            ]
        })

    return input_array

async def send_long_followup(interaction: discord.Interaction, text: str):
    """
    Sendet lange Texte auf mehrere Followups aufgeteilt.
    """
    MAX_LEN = 2000
    if len(text) <= MAX_LEN:
        await interaction.followup.send(text)
    else:
        lines = text.split("\n")
        chunk = ""
        for line in lines:
            if len(chunk) + len(line) + 1 > MAX_LEN:
                await interaction.followup.send(chunk)
                chunk = line + "\n"
            else:
                chunk += line + "\n"
        if chunk:
            await interaction.followup.send(chunk)

##############################################################################
# /suche
##############################################################################
@tree.command(name="suche", description="Frage GPT-4o mit Websuche.")
@app_commands.describe(prompt="Worüber soll gesucht werden?")
async def suche_command(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()
    channel_id = interaction.channel_id
    channel_hist = channel_history[channel_id]

    # Historie + Prompt in passendes Format
    input_for_responses = build_responses_input_from_history(channel_hist, new_prompt=prompt)

    try:
        async with interaction.channel.typing():
            # Wie im Playground-Beispiel:
            response = await openai_client.responses.create(
                model="gpt-4o",
                input=input_for_responses,
                text={
                    "format": {
                        "type": "text"
                    }
                },
                reasoning={},  # optional, kann leer sein
                tools=[
                    {
                        "type": "web_search_preview",
                        "user_location": {
                            "type": "approximate",
                            "country": "CH",
                            "region": "Solothurn",
                            "city": "Rüttenen"
                        },
                        "search_context_size": "high"
                    }
                ],
                temperature=1,
                max_output_tokens=2048,
                top_p=1,
                store=True
            )
    except Exception as e:
        logging.exception("Fehler bei /suche")
        await interaction.followup.send(f"Fehler bei der Websuche: {e}")
        return

    assistant_content = response.output_text or "(Keine Antwort)"

    # In den Verlauf übernehmen
    channel_hist.append({"role": "user", "content": prompt})
    channel_hist.append({"role": "assistant", "content": assistant_content})

    await send_long_followup(interaction, assistant_content)

    # LaTeX
    latex_expressions = extract_latex_expressions(assistant_content)
    if latex_expressions:
        try:
            latex_image_buffer = await render_latex_image(latex_expressions)
            if latex_image_buffer:
                file = discord.File(fp=latex_image_buffer, filename="latex.png")
                await interaction.followup.send(file=file)
        except Exception as e:
            logging.warning(f"Fehler beim Rendern von LaTeX: {e}")

##############################################################################
# /denken
##############################################################################
@tree.command(name="denken", description="Nutze das o3-mini Modell.")
@app_commands.describe(prompt="Dein Prompt für das kleine Reasoning-Modell")
async def denken_command(interaction: discord.Interaction, prompt: str):
    """
    Nutzt die Responses-API mit model='o3-mini', so wie im Playground-Beispiel.
    """
    await interaction.response.defer()
    channel_id = interaction.channel_id
    channel_hist = channel_history[channel_id]

    # Historie + Prompt in passendes Format
    input_for_responses = build_responses_input_from_history(channel_hist, new_prompt=prompt)

    try:
        async with interaction.channel.typing():
            response = await openai_client.responses.create(
                model="o3-mini",
                input=input_for_responses,
                text={
                    "format": {
                        "type": "text"
                    }
                },
                reasoning={
                    "effort": "high"
                },
                tools=[],
                store=True
                # Hier KEIN max_output_tokens, da Playground-Beispiel es nicht zeigt
            )
    except Exception as e:
        logging.exception("Fehler bei /denken")
        await interaction.followup.send(f"Fehler beim Denken: {e}")
        return

    assistant_content = response.output_text or "(Keine Antwort)"

    # In den Verlauf übernehmen
    channel_hist.append({"role": "user", "content": prompt})
    channel_hist.append({"role": "assistant", "content": assistant_content})

    await send_long_followup(interaction, assistant_content)

    latex_expressions = extract_latex_expressions(assistant_content)
    if latex_expressions:
        try:
            latex_image_buffer = await render_latex_image(latex_expressions)
            if latex_image_buffer:
                file = discord.File(fp=latex_image_buffer, filename="latex.png")
                await interaction.followup.send(file=file)
        except Exception as e:
            logging.warning(f"Fehler beim Rendern von LaTeX: {e}")

##############################################################################
# /bild
##############################################################################
@tree.command(name="bild", description="Erzeuge ein Bild mit DALL-E 3.")
@app_commands.describe(
    prompt="Was soll auf dem Bild zu sehen sein?",
    format="Seitenverhältnis: quadratisch, hoch oder breit"
)
async def bild_command(interaction: discord.Interaction, prompt: str, format: str):
    await interaction.response.defer()

    format_lower = format.lower()
    if format_lower == "quadratisch":
        size = "1024x1024"
    elif format_lower == "hoch":
        size = "1024x1792"
    elif format_lower == "breit":
        size = "1792x1024"
    else:
        await interaction.followup.send("Bitte wähle 'quadratisch', 'hoch' oder 'breit'.")
        return

    try:
        async with interaction.channel.typing():
            response = await openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                quality="hd",
                size=size,
            )
    except Exception as e:
        logging.exception("Fehler bei /bild")
        await interaction.followup.send(f"Fehler beim Bilderzeugen: {e}")
        return

    if not response.data:
        await interaction.followup.send("Keine Bilddaten erhalten.")
        return

    image_obj = response.data[0]
    image_url = image_obj.url

    await interaction.followup.send(f"Hier dein Bild:\n{image_url}")

    # Optional im Chat-Verlauf
    channel_id = interaction.channel_id
    channel_hist = channel_history[channel_id]
    if SYSTEM_PROMPT and not channel_hist:
        channel_hist.append({"role": "system", "content": SYSTEM_PROMPT})

    channel_hist.append({"role": "user", "content": f"(Bildgenerierung) {prompt} (Format: {format})"})
    channel_hist.append({"role": "assistant", "content": f"Bild generiert: {image_url}"})

##############################################################################
# 8) START
##############################################################################
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
