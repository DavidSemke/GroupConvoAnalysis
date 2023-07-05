def turn_taking_data_pts(convo):
    data_pts = []
    index_to_speaker = []
    speaker_to_index = {}
    new_idx = 0
    last_utt_speaker_id = None

    for utt in convo.iter_utterances():
        utt_speaker_id = utt.speaker.id

        if utt_speaker_id == last_utt_speaker_id: continue

        idx = speaker_to_index.get(utt_speaker_id)

        if idx is not None:
            data_pts.append(idx)
        
        else:
            speaker_to_index[utt_speaker_id] = new_idx
            data_pts.append(new_idx)
            index_to_speaker.append(utt_speaker_id)
            new_idx += 1
        
        last_utt_speaker_id = utt_speaker_id
    
    return data_pts, index_to_speaker