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
            client_long = openai_client.with_options(timeout=120.0)
            resp = await client_long.responses.create(
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
            )

        content = _extract_response_text(resp) or ""

        # Fallback, falls das Modell trotzdem Befehle empfiehlt
        if "slash-command" in content.lower() or "/suche" in content.lower():
            resp = await client_long.responses.create(
                model=DEFAULT_MODEL,
                input=input_messages,
                instructions=web_instructions + " Antworte jetzt ohne Hinweis auf Befehle.",
                tools=tools,
                text={"verbosity": "high", "format": {"type": "text"}},
                reasoning={"effort": "high"},
                store=True
            )
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
