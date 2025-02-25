# Kevin – Denkender Discord-Bot mit PDF- und LaTeX-Funktionalität



Willkommen bei Kevin, Deinem AI-basierten Discord-Bot! Du kannst ihn als persönlichen Assistenten für Textfragen, PDF-Analyse und LaTeX-Rendering verwenden. Kevin nutzt die OpenAI-API und bietet so intelligente Antworten in Deinem Discord-Kanal.

---


## Was macht Kevin besonders?

- **Bilder und PDFs:**:  
  - Er verarbeitet Textnachrichten, Bilder und PDFs direkt im Chat.  

- **LaTeX-Ausdrücke:**:  
  - Er erkennt LaTeX-Ausdrücke und rendert sie als Bild via CodeCogs.

---

## Voraussetzungen

- Python 3.12 (oder höher)
- Ein funktionierendes Discord-Bot-Token
- Ein OpenAI-API-Key
- Eine Umgebung, in der Du Python-Bibliotheken aus der requirements.txt installieren kannst
- Internetzugriff, damit CodeCogs für das LaTeX-Rendering erreichbar ist


## Installation

ieses Repository klonen oder herunterladen
requirements.txt installieren:
  ```bash
pip install -r requirements.txt
```
Deine Zugangsdaten (bot_token, api_key) in der config.yaml eintragen
Bot starten:
  ```bash
python3 main.py
```

## Funktionen im Überblick

- Chat: Kevin reagiert auf Deine Nachrichten im Kanal und in Direkt-Nachrichten.
- PDF-Extraktion: Hänge ein PDF an oder schicke es direkt im Chat, Kevin liest den Text aus und berücksichtigt ihn in der Antwort.
- LaTeX-Erkennung: Formeln wie $$a^2 + b^2 = c^2$$ oder \[(a + b)^n\] rendert Kevin als PNG und hängt es an die Antwort an.
- Alternative Modelle: Du kannst das Modell wechseln, um z. B. ein „denkendes“ Modell (z. B. o3-mini) zu verwenden. Baue dafür ein Slash-Command oder ein Textkommando (z. B. /Denken) ein.

## Nutzungshinweise

- Die PDF-Extraktion ist auf eine gewisse Textlänge begrenzt (siehe config.yaml).
- LaTeX wird via CodeCogs gerendert. Beachte die Nutzungsbedingungen.
- Achte darauf, Deinen API-Key nicht versehentlich öffentlich zu machen.

## Support und Beiträge

- Erstelle Pull Requests, wenn Du Ideen oder Verbesserungen hast.
- Nutze die Issues, falls Du Fehler findest.

---


## Warnungen & Hinweise
**Kosten / Tokenverbrauch**
- GPT-4o Vision analysiert Bilder und PDFs; das kann schnell Tokens kosten.
- DALL·E 3-Bilder kosten je nach Prompt/Ergebnis.
**Rate Limits**
- OpenAI hat Rate Limits, Discord ebenso. Bei zu vielen Messages in kurzer Zeit kann es zu Verzögerungen kommen.
**Privatsphäre**
- Alle hochgeladenen Dateien (Bilder/PDFs) werden kurz an OpenAI geschickt. Bedenke das, wenn es vertrauliche Infos sind.
**Große PDFs**
- Kürzen die PDF-Auszüge auf die gewünschte Anzahl Zeichen. Bei sehr langen Dokumenten musst du ggf. manuell aufteilen.
**Vision-Qualität**
- GPT-4o Vision ist nicht unfehlbar (besonders bei kleinen Details, Text in Bildern, speziellen Grafiken).


<img width="539" alt="image" src="https://github.com/user-attachments/assets/a1c002ea-5b67-4365-a734-9b23f5e474c7" />




