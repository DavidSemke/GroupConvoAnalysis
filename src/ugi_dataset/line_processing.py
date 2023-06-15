import re

def process_text(text, patts):
    text = re.sub(patts['punct'], '', text)
    text = re.sub(patts['nonverbal'], '', text)
    text = re.sub('\s+', ' ', text)

    return text.strip()