from src.ugi_dataset.line_processing import *
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
    colored_utts = {color:0 for color in const.speaker_colors}
    first_loop = True
    prior_utt_id = None
    line_index = 0
    
    while line_index < len(lines):

        elements = process_line_elements(lines, line_index, patts)
        
        if not elements:
            line_index += 1
            continue

        timestamp, local_id, sents = elements
        color = const.speaker_colors[local_id-1]
        
        speaker_id = f'{group_id}.{color}'
        utt_id_sent_pairs = []

        for sent in sents:
            colored_utts[color] += 1
            utt_id_sent_pairs.append(
                (f'{speaker_id}.{colored_utts[color]}', sent)
            )

        if first_loop:
            convo_id = utt_id_sent_pairs[0][0]
            first_loop = False
        
        for utt_id, sent in utt_id_sent_pairs:
            utt_metadata[utt_id] = {
                'id': utt_id,
                'conversation_id': convo_id,
                'text': sent,
                'speaker': speaker_id,
                'meta': {},
                'reply-to': prior_utt_id,
                'timestamp': timestamp
            }
        
        prior_utt_id = utt_id_sent_pairs[0][0]
        line_index += 1

    print(f"Utts for group {group_id} finished")