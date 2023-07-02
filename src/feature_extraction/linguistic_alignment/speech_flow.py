import prosodic as pro
from src.utils.timestamps import convert_to_secs

def speech_rates(utts):
    rates = []

    for utt in utts:
        duration_timestamp = utt.meta['Duration']
        duration = convert_to_secs(duration_timestamp)

        text = pro.Text(utt.text)
        syll_count = len(text.syllables())
        
        rate = syll_count/duration
        rates.append(rate)
    
    return rates


def convo_speech_pauses(convo):
    pauses = []
    utts = convo.get_chronological_utterance_list()

    for i in range(len(utts)-1):
        curr_utt = utts[i]
        next_utt = utts[i+1]
        curr_end_time = convert_to_secs(curr_utt.meta["End"])
        next_start_time = convert_to_secs(next_utt.timestamp)

        if curr_end_time < next_start_time:
            pauses.append(next_start_time - curr_end_time)

    return pauses