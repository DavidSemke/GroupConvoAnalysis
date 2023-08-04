from src.ugi_dataset.line_processing import process_line_elements
from src.ugi_dataset.utterance_periods import (
    approx_utterance_periods,
    explode_shared_timestamps
)
from src.utils.timestamps import convert_to_secs
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
        
        # merge dicts
        utt_metadata |= extract_utterances(lines, group_id, patts)
    
    return utt_metadata


def extract_utterances(lines, group_id, patts):
    group_utt_metadata = {}
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
            group_utt_metadata[utt_id] = {
                'id': utt_id,
                'conversation_id': convo_id,
                'text': sent,
                'speaker': speaker_id,
                'meta': {},
                'reply-to': prior_utt_id,
                'timestamp': timestamp
            }

            prior_utt_id = utt_id
        
        line_index += 1

    # ensure utterances are chronologically ordered
    # group_utt_list = list(group_utt_metadata.values())
    # group_utt_list.sort(
    #     key=lambda u: round(convert_to_secs(u['timestamp']), 1)
    # )

    # group_utt_list = explode_shared_timestamps(group_utt_list)
    
    # Add 'Duration' and 'End' fields to utt.meta property
    # Input is list of dicts from group_utt_metadata, so changes to the 
    # dicts change dict group_utt_metadata
    # approx_utterance_periods(group_utt_list)

    # reconstruct utt metadata dict
    # group_utt_metadata = {
    #     utt['id']:utt for utt in group_utt_list
    # }

    print(f"Utts for group {group_id} finished")

    return group_utt_metadata