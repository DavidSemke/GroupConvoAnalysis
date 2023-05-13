from source.feature_extraction.utils.timestamps import convert_to_secs

# relies on timestamp, End keys of utt (not Duration) 
def avg_speech_overlap_period(convo):
    all_utts = convo.get_chronological_utterance_list()

    overlap_count = 0
    overlap_time = 0
    for i in range(len(all_utts)):
        if i == len(all_utts)-1:
            break

        curr = all_utts[i]
        next = all_utts[i+1]
        curr_end_time = convert_to_secs(curr.meta["End"])
        next_start_time = convert_to_secs(next.timestamp)

        if curr_end_time > next_start_time:
            overlap_count +=1
            overlap_time += curr_end_time - next_start_time

    return round(overlap_time/overlap_count, 1)


def sentiment_ratio():
    pass

def contrast_in_formality():
    # all_utts = convo.get_chronological_utterance_list()
    pass

