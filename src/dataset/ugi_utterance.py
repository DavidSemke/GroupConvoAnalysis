from src.utils.timestamps import convert_to_timestamp
import src.constants as const
import re

def utterance_metadata(transcripts_path):
    utt_metadata = {}

    for group_id in range(1, 23):
        txt_path = (
            transcripts_path 
            + rf'\TeamID_{group_id}_transcript.txt'
        )

        with open(txt_path, 'r') as file:
            lines = file.readlines()
        
        extract_utterances(
            lines,
            group_id,
            utt_metadata
        )
    
    return utt_metadata


def extract_utterances(lines, group_id, utt_metadata):
    lines = [l.strip() for l in lines if l.strip()]
    colored_utts = {}
    first_loop = True
    line_index = 0
    patt1 = '[pP]erson[1-5]'
    patt2 = '^[0-9]+(\.[0-9]+)?'
    patt3 = '[^ \w]' # for removing punctuation
    patt4 = 'HESITATION|LAUGHTER'
    
    while line_index < len(lines):
        line = lines[line_index]
        person_matches = re.findall(patt1, line)

        if not person_matches:
            raise Exception(f'Line is missing person ID:\n\n{line}\n')
        
        elif len(person_matches) > 1: 
            raise Exception(f'Line contains multiple utts:\n\n{line}\n')

        splitter = person_matches[0]
        head, tail = line.split(splitter)

        text = re.sub(patt3, '', tail)
        text = re.sub(patt4, '', text)
        text = text.strip()

        if not text:
            line_index += 1
            continue
        
        secs_match = re.search(patt2, head)
        
        if not secs_match:
            raise Exception(f'Line is missing secs:\n\n{line}\n')
        
        secs = secs_match.group()
        timestamp = convert_to_timestamp(secs)
        
        if (
            line_index+1 < len(lines)
            and not re.search(patt1, lines[line_index+1])
            and not re.search(patt2, lines[line_index+1])
        ):
            next_line = lines[line_index+1]
            next_line_text = re.sub(patt3, '', next_line)
            next_line_text = re.sub(patt4, '', next_line_text)
            next_line_text = next_line_text.strip()

            text += ' ' + next_line_text if next_line_text else ''
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