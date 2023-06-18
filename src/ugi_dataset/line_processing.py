import re
import nltk
from src.utils.timestamps import convert_to_timestamp

def process_line_elements(lines, line_index, patts):
    line = lines[line_index]
    secs, person_id, sents = parse_line(line, patts)

    next_line_is_text = (
        line_index+1 < len(lines)
        and not re.search(patts['p_match'], lines[line_index+1])
        and not re.search(patts['ts'], lines[line_index+1])
    )
    
    if next_line_is_text:
        next_text = lines[line_index+1]
        last_sent = sents[-1]
        del sents[-1]
        last_sent += ' ' + next_text
        last_sent = last_sent.strip()
        sents.extend(nltk.sent_tokenize(last_sent))
        del lines[line_index+1]

    for i, sent in enumerate(sents):
        sents[i] = process_text(sent, patts)
    
    sents = [s for s in sents if s]
    
    if not sents: return
    
    local_id = int(person_id[-1])
    timestamp = convert_to_timestamp(secs)

    return (timestamp, local_id, sents)


def parse_line(line, patts):
    person_matches = re.findall(patts['p_match'], line)

    if not person_matches:
        raise Exception(f'Line is missing person ID:\n\n{line}\n')
    
    elif len(person_matches) > 1: 
        raise Exception(f'Line contains multiple utts:\n\n{line}\n')

    person_id = person_matches[0]
    head, tail = line.split(person_id)    
    secs_match = re.search(patts['ts'], head)
    
    if not secs_match:
        raise Exception(f'Line is missing secs:\n\n{line}\n')
    
    secs = secs_match.group()
    sents = nltk.sent_tokenize(tail)

    return (secs, person_id, sents)


def process_text(text, patts):
    text = re.sub(patts['punct'], '', text)
    text = re.sub(patts['nonverbal'], '', text)
    text = re.sub('\s+', ' ', text)

    return text.strip()