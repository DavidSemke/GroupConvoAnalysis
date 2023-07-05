def stress_data_pts(stresses):
    data_pts = []
    position_to_index = {'U': 0, 'S': 1, 'P': 2}

    for utt_id in stresses:
        utt_stresses = list(stresses[utt_id].values())

        if len(utt_stresses) != 1:
            raise Exception(
                'Utterance does not have one stress pattern'
            )

        utt_stress = utt_stresses[0]
        positions = utt_stress.split('|')

        for pos in positions:
            for slot in pos:
                data_pts.append(position_to_index[slot])
        
    return data_pts