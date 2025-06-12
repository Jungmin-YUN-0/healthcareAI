def context_to_string(context_item):
    """
    context_item: list of dicts, each dict should have keys like 'from' and 'value'
    Returns a single string with format "client:내용 \n doctor:내용 \n ..."
    """
    lines = []
    for turn in context_item:
        # Try to find speaker and text keys
        if 'from' in turn and 'value' in turn:
            speaker = turn['from']
            text = turn['value']
        elif 'role' in turn and 'content' in turn:
            speaker = turn['role']
            text = turn['content']
        else:
            continue  # skip if format unknown

        # Map speaker to Korean label
        if speaker.lower() in ['client']:
            speaker_label = 'client'
        elif speaker.lower() in ['doctor']:
            speaker_label = 'doctor'
        else:
            speaker_label = speaker

        lines.append(f"{speaker_label}: {text}")

    return "\n".join(lines)