# Discord Family Bot mit GPT-4o (Vision, PDF-Verarbeitung) und DALL·E 3

Dies ist ein umfassender Discord-Bot, der auf folgende Weisen reagiert:

1. **GPT-4o** für Chat (Textnachrichten), Vision-Funktionen (Bilderkennung) und das Einlesen von PDF-Dateien (nach lokaler Umwandlung in Text).  
2. **DALL·E 3** für das explizite Erstellen von Bildern per Slash-Command `/bild`.

Der Bot ist auf den Einsatz in einem kleinen, privaten Discord-Server (z.B. für die Familie) ausgelegt. Er reagiert auf alle Nachrichten von bestimmten, freigegebenen Usern (Whitelist) und ignoriert den Rest.

---

## Inhaltsverzeichnis

1. [Übersicht der Features](#übersicht-der-features)  
2. [Funktionsweise im Detail](#funktionsweise-im-detail)  
   - [Textnachrichten (Chat)](#textnachrichten-chat)  
   - [Bilderkennung mit GPT-4o Vision](#bilderkennung-mit-gpt-4o-vision)  
   - [PDF-Dateien lesen](#pdf-dateien-lesen)  
   - [Bildgenerierung mit Slash-Command `/bild`](#bildgenerierung-mit-slash-command-bild)  
3. [Voraussetzungen](#voraussetzungen)  
4. [Installation & Start](#installation--start)  
5. [Konfiguration](#konfiguration)  
6. [Warnungen & Hinweise](#warnungen--hinweise)  
7. [Erweiterungen](#erweiterungen)  
8. [Lizenz](#lizenz)

---

## Übersicht der Features

- **Automatisches Reagieren auf Nachrichten**:  
  Der Bot lauscht auf alle Nachrichten im Kanal (mittels `on_message`).  

- **GPT-4o Vision**:  
  - Bilder (PNG, JPG, GIF, WEBP) werden an GPT-4o mit Vision-Funktion gesendet, damit der Bot beschreiben kann, was im Bild zu sehen ist.

- **PDF-Einlesen**:  
  - Wenn ein User eine PDF-Datei hochlädt, wird diese kurz heruntergeladen und mit [`pdfplumber`](https://pypi.org/project/pdfplumber/) in Text umgewandelt. Dann analysiert GPT-4o diesen Text.

- **Reiner Textchat**:  
  - Falls weder Anhänge noch bestimmte Befehle genutzt werden, nimmt GPT-4o einfach den Text entgegen und antwortet.

- **Bildgenerierung (`/bild`)**:  
  - Über den Slash-Command `/bild prompt:<Beschreibung>` kann man mit **DALL·E 3** ein Bild generieren lassen.

- **Whitelist**:  
  - Nur bestimmte User (Discord-UserIDs) können den Bot nutzen. Andere Nachrichten ignoriert der Bot.

---

## Funktionsweise im Detail

### Textnachrichten (Chat)

- Schreibt jemand (aus der Whitelist) eine **Textnachricht ohne Anhang**, wird diese an GPT-4o gesendet.  
- GPT-4o antwortet darauf wie ein Assistent (im Code wird jede Nachricht einzeln behandelt, ohne großen Verlauf).

### Bilderkennung mit GPT-4o Vision

- Lädt ein whitelisted User ein Bild hoch (z.B. `.png`, `.jpg`), verwendet der Bot die Vision-Funktionen von GPT-4o.  
- Die Bild-URL wird in der Struktur `{ "type": "image_url" }` an GPT-4o geschickt, damit das Modell beschreiben kann, was auf dem Bild zu sehen ist.

### PDF-Dateien lesen

- Bei hochgeladenen PDF-Dateien lädt der Bot diese herunter, wandelt sie in **Text** um und lässt GPT-4o den extrahierten Inhalt zusammenfassen oder analysieren.  
- So kann man Dokumente, Skripte oder Arbeitsblätter direkt in Discord hochladen und vom Bot erklären lassen.

### Bildgenerierung mit Slash-Command `/bild`

- Für das **Erstellen** neuer Bilder nutzen wir DALL·E 3.  
- Der Slash-Befehl:  
  ```bash
  /bild prompt:Ein Einhorn auf Wolken im Cartoon-Stil
  ```
löst eine OpenAI-API-Anfrage an dall-e-3 aus und liefert eine Bild-URL zurück.

## Voraussetzungen

1. **Python 3.9+** (empfohlen 3.10 oder höher)  
2. **Discord Developer-Anwendung**  
   - Ein Bot-Token mit den Scopes `bot` + `applications.commands`  
   - **Message Content Intent** (Privileged Gateway Intents) aktiviert  
3. **OpenAI-Account** mit Zugriff auf:  
   - `gpt-4o` (Vision-fähig)  
   - `dall-e-3` (Bildgenerierung)  
4. **Benötigte Python-Bibliotheken** (siehe `requirements.txt`):  
   - `discord.py>=2.3.2`  
   - `openai>=1.0.0`  
   - `pdfplumber` (für PDF-Verarbeitung)  
   - `requests`, `Pillow` etc. (teils optional)

---

## Installation & Start

1. **Repository klonen oder downloaden**  
   ```bash
   git clone https://github.com/deinNutzername/discord-family-bot-gpt4o.git
   cd discord-family-bot-gpt4o
   ```

2. **Abhängigkeiten installieren**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **main.py anpassen**
Trage in main.py deinen Discord-Bot- und OpenAI Token ein
```bash
DISCORD_TOKEN = "DEIN_BOT_TOKEN"
OPENAI_API_KEY = "DEIN_OPENAI_API_KEY"
```

4. **Bot starten**
```bash
python main.py
```

## Konfiguration
Alle Einstellungen liegen in main.py:

- DISCORD_TOKEN, OPENAI_API_KEY
- WHITELIST (Liste von Discord-UserIDs)
- install_packages() (Hack-Script für ZAP-Hosting; kann entfernt werden, wenn du lokal arbeitest)
- Modellnamen ("gpt-4o", "dall-e-3") können ggf. angepasst werden, falls OpenAI etwas ändert.

## Slash-Commands
Der Befehl /bild wird im Code so definiert:
```bash
@bot.tree.command(name="bild", description="Erstelle ein Bild mit DALL·E 3")
```

Dieser wird beim Start des Bots synchronisiert.
Manchmal muss man den Bot neu einladen oder ein paar Minuten warten, bis /bild in Discord sichtbar wird.

## Warnungen & Hinweise
**Kosten / Tokenverbrauch**
- GPT-4o Vision analysiert Bilder und PDFs; das kann schnell Tokens kosten.
- DALL·E 3-Bilder kosten je nach Prompt/Ergebnis.
**Rate Limits**
- OpenAI hat Rate Limits, Discord ebenso. Bei zu vielen Messages in kurzer Zeit kann es zu Verzögerungen kommen.
**Privatsphäre**
- Alle hochgeladenen Dateien (Bilder/PDFs) werden kurz an OpenAI geschickt. Bedenke das, wenn es vertrauliche Infos sind.
**Große PDFs**
- Wir kürzen die PDF-Auszüge auf 8000 Zeichen. Bei sehr langen Dokumenten musst du ggf. manuell aufteilen.
**Vision-Qualität**
- GPT-4o Vision ist nicht unfehlbar (besonders bei kleinen Details, Text in Bildern, speziellen Grafiken).
**Erweiterungen**
Persistente Chat-Verläufe
Du könntest ein Dictionary anlegen, das pro User den Chat-Verlauf speichert und an GPT-4o übergibt.
Kontextmenü statt on_message
Für Bilder könntest du einen Rechtsklick-Kontextbefehl programmieren („Analysiere dieses Bild“).
Feinabstimmung
Manche User möchten nur Vision oder nur Text. Man kann on_message so anpassen, dass pro Kanal unterschiedliche Features aktiv sind.
Mehr Slash-Commands
/pdf, /vision, /chat, etc. – je nachdem, wie du es organisieren willst.



