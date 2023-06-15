import re
from src.utils.timestamps import convert_to_timestamp
from src.ugi_dataset.line_processing import process_text
import src.constants as const

def utterance_metadata(transcripts_path, patts):
    utt_metadata = {}

    for group_id in range(1, 23):
        txt_path = (
            transcripts_path 
            + rf'\TeamID_{group_id}_transcript.txt'
        )

        with open(txt_path, 'r') as file:
            lines = file.readlines()
        
        extract_utterances(lines, group_id, utt_metadata, patts)
    
    return utt_metadata


def extract_utterances(lines, group_id, utt_metadata, patts):
    lines = [l.strip() for l in lines if l.strip()]
    colored_utts = {}
    first_loop = True
    line_index = 0
    
    while line_index < len(lines):
        line = lines[line_index]
        person_matches = re.findall(patts['p_match'], line)

        if not person_matches:
            raise Exception(f'Line is missing person ID:\n\n{line}\n')
        
        elif len(person_matches) > 1: 
            raise Exception(f'Line contains multiple utts:\n\n{line}\n')

        splitter = person_matches[0]
        head, tail = line.split(splitter)

        text = process_text(tail, patts)

        if not text:
            line_index += 1
            continue
        
        secs_match = re.search(patts['ts'], head)
        
        if not secs_match:
            raise Exception(f'Line is missing secs:\n\n{line}\n')
        
        secs = secs_match.group()
        timestamp = convert_to_timestamp(secs)
        
        if (
            line_index+1 < len(lines)
            and not re.search(patts['p_match'], lines[line_index+1])
            and not re.search(patts['ts'], lines[line_index+1])
        ):
            next_line_text = process_text(lines[line_index+1], patts)
            text += ' ' + next_line_text
            text = text.strip()
            del lines[line_index+1]
        
        person_id = int(splitter[-1])
        color = const.speaker_colors[person_id-1]
        
        if color not in colored_utts:
            colored_utts[color] = 1
        else:
            colored_utts[color] += 1
        
        speaker_id = f'{group_id}.{color}'
        utt_id = f'{speaker_id}.{colored_utts[color]}'

        if first_loop:
            convo_id = utt_id
            first_loop = False
        
        utt_metadata[utt_id] = {
            'id': utt_id,
            'conversation_id': convo_id,
            'text': text,
            'speaker': speaker_id,
            'meta': {},
            'reply-to': None,
            'timestamp': timestamp
        }

        line_index += 1

    print(f"Utts for group {group_id} finished")