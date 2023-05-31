def convo_frame(convo, frame):

    utts = convo.get_chronological_utterance_list()
    total_utts = len(utts)
    half_frame_utt_count = round((frame/100) * (total_utts))
    
    first = utts[:half_frame_utt_count]
    last = utts[total_utts - half_frame_utt_count:]

    return first, last
