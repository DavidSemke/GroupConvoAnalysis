import pandas as pd
import src.constants as const


def speaker_metadata(df, expert_ranking):
    speaker_meta = {}

    group_id = 0
    participant_id = 0
    for i in range(df.shape[0]):

        ais = speaker_ais(df, i, expert_ranking)
        
        gender = 1 if df.iloc[i, -3] == 'M' else 2

        next_group_id = df.iloc[i, 0]

        if group_id != next_group_id:
            group_id = next_group_id
            participant_id = 1
        else:
            participant_id += 1

        speaker_id = (
            f'{group_id}.{const.speaker_colors[participant_id-1]}'
        )
        
        speaker_meta[speaker_id] = {
            'Gender': gender,
            'AIS': ais,
            'Group Number': str(group_id)
        }

    return speaker_meta


def speaker_ais(df, row, expert_ranking):
    ais = 0
    
    for j in range(len(expert_ranking)):
        i_rank = df.iloc[row, 3+j]
        
        # add abs diff between expert rank and individ rank
        if not pd.isna(i_rank):
            ais += abs(i_rank - expert_ranking[j])
    
    return ais