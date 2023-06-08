import pandas as pd
from src.dataset.ugi_utterance import utterance_metadata
from src.dataset.ugi_convo import convo_metadata
from src.dataset.ugi_speaker import speaker_metadata

def main():
    transcripts_path = r'corpora\ugi-corpus\transcripts'

    utt_meta = utterance_metadata(transcripts_path)
    
    metadata_path = r'corpora\ugi-corpus\UGI_Metadata.xlsx'
    df = pd.read_excel(metadata_path, skiprows=1).drop(
        ['Not Ranked', 'Not Ranked.1'], axis=1
    )
    expert_ranking = [15,4,6,8,13,11,12,1,3,9,14,2,10,7,5]
    
    speaker_meta = speaker_metadata(df, expert_ranking)

    convo_meta = convo_metadata(df, expert_ranking, transcripts_path)

    create_utterances_jsonl(utt_meta)

    create_speakers_json(speaker_meta)

    create_conversations_json(convo_meta)
    

def create_utterances_jsonl(utt_metadata):
    
    with open(
        'corpora/ugi-corpus/convokit_v2/utterances.jsonl', 
        'w', 
        newline=''
    ) as file:
        lines = []

        for utt in utt_metadata.values():
            line = '{'
            
            for k, v in utt.items():
                
                if k == "meta":
                    line += f'"{k}": {v}, '
                elif k == "reply-to":
                    line += f'"{k}": null, '
                else:
                    line += f'"{k}": "{v}", '
            
            line = line[:-2] + '}\n'
            lines.append(line)
        
        file.writelines(lines)


def create_speakers_json(speaker_metadata):

    with open(
        'corpora/ugi-corpus/convokit_v2/speakers.json', 
        'w', 
        newline=''
    ) as file:
        line = '{'
        
        for s_id, meta in speaker_metadata.items():
            line += f'"{s_id}": {{'
            
            for k, v in meta.items():
                
                if k == "Group Number":
                    line += f'"{k}": "{v}", '
                else:
                    line += f'"{k}": {v}, '
            
            line = line[:-2] + '}, '
        
        line = line[:-2] + '}'
        file.write(line)


def create_conversations_json(convo_metadata):
    
    with open(
        'corpora/ugi-corpus/convokit_v2/conversations.json', 
        'w', 
        newline=''
    ) as file:
        line = '{'
        
        for c_id, meta in convo_metadata.items():
            line += f'"{c_id}": {{'
            
            for k, v in meta.items():
                
                if k == "Group Number":
                    line += f'"{k}": "{v}", '
                else:
                    line += f'"{k}": {v}, '
            
            line = line[:-2] + '}, '
        
        line = line[:-2] + '}'
        file.write(line)


if __name__ == "__main__":
    main()