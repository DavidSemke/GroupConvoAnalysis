import numpy as np
import prosodic as pro
from src.utils.timestamps import convert_to_secs, convert_to_timestamp

# Define End and Duration meta fields for utterances
def approx_utterance_periods(utt_meta, pause_secs=0.6):
    utt_list = list(utt_meta.values())
    rates = approx_speech_rates(utt_list)
    speaker_rates = {}

    # organize utt speech rates by speaker
    for i, utt in enumerate(utt_list):
        sid = utt['speaker']

        if sid in speaker_rates:
            speaker_rates[sid].append(rates[i])
        
        else:
            speaker_rates[sid] = [rates[i]]

    speaker_rate_medians = {
        sid: np.median(speaker_rates[sid]) for sid in speaker_rates
    }
    # standard deviation of speech rate for each speaker
    speaker_rate_stdevs = {
        sid: np.std(speaker_rates[sid]) for sid in speaker_rates}

    for i in range(len(utt_list)-1):
        curr_utt = utt_list[i]
        next_utt = utt_list[i+1]

        text = pro.Text(curr_utt['text'])
        syll_count = len(text.syllables())

        sid = curr_utt['speaker']
        # seconds = (syllables) / (syllables per second)
        duration = syll_count / speaker_rate_medians[sid]
        
        timestamp = curr_utt['timestamp']
        start_secs = convert_to_secs(timestamp)
        end_secs = start_secs + duration
        
        next_start_secs = convert_to_secs(next_utt['timestamp'])
        pause_period = round(next_start_secs - end_secs, 1)
        
        # let the minimum duration be 0.1 seconds
        if duration < 0.1:
            duration = 0.1
            end_secs = start_secs + duration

        # pause_secs is a typical pause length between utts
        # If the current pause length (pause_period) before the next utt is longer than the typical length, try slowing the speech 
        # rate by the speaker's standard deviation in speech rate to 
        # reduce pause length
        # If adjusted pause length is still longer than the typical pause length, use adjusted pause length
        elif pause_period > pause_secs:
            rate_stdev = speaker_rate_stdevs[sid]
            rate_median = speaker_rate_medians[sid]
            
            adjusted_rate = rate_median - rate_stdev
            adjusted_duration = syll_count / adjusted_rate
            
            if pause_period <= pause_secs:
                duration = adjusted_duration
                end_secs = start_secs + adjusted_duration

        duration_timestamp = convert_to_timestamp(duration)
        curr_utt['meta']['Duration'] = duration_timestamp
        end_timestamp = convert_to_timestamp(end_secs)
        curr_utt['meta']['End'] = end_timestamp


# This function only uses timestamps to determine speech rate
# (does not use utt.meta['Duration'])
# Pauses between any two utterances that are longer than 'pause_secs' 
# seconds will be reduced by 'pause_secs' seconds
# Utterances must be chronologically ordered
def approx_speech_rates(utt_list, pause_secs=0.6):
    rates = []

    for i in range(len(utt_list)-1):
        curr_utt = utt_list[i]
        next_utt = utt_list[i+1] 
        
        start_secs = convert_to_secs(curr_utt['timestamp'])
        end_secs = convert_to_secs(next_utt['timestamp'])
        
        if end_secs < start_secs:
            end_secs = start_secs
            next_utt['timestamp'] = convert_to_timestamp(end_secs)
        
        utt_period = end_secs - start_secs

        if utt_period < 0.1:
            duration = 0.1

        elif round(utt_period, 1) <= pause_secs:
            duration = utt_period

        else:
            duration = utt_period - pause_secs
        
        text = pro.Text(curr_utt['text'])
        syll_count = len(text.syllables())
        rate = syll_count/duration
        rates.append(rate)
    
    # add speech rate for final utt
    final_utt = utt_list[-1]
    final_speaker_rates = []

    for i, rate in enumerate(rates):
        final_sid = final_utt['speaker']

        if final_sid == utt_list[i]['speaker']:
            final_speaker_rates.append(rate)
            
    rates.append(np.median(final_speaker_rates))

    return rates