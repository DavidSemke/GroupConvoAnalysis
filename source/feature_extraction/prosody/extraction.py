from source.feature_extraction.utils.timestamps import convert_to_secs

def speech_pause_percentage(convo):
    
    all_utts = convo.get_chronological_utterance_list()
    pause_time = 0
    for i in range(len(all_utts)):
        
        if i == len(all_utts)-1:
            break

        curr = all_utts[i]
        next = all_utts[i+1]
        curr_end_time = convert_to_secs(curr.meta["End"])
        next_start_time = convert_to_secs(next.timestamp)

        if curr_end_time < next_start_time:
            pause_time += next_start_time - curr_end_time

    total_time = convo.meta['Meeting Length in Minutes'] * 60

    return round(pause_time/total_time, 2)


