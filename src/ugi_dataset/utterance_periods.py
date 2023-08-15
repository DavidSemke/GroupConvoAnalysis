import numpy as np
import prosodic as pro
from src.utils.timestamps import convert_to_secs, convert_to_timestamp


"""
THESE METHODS DO NOT WORK PROPERLY - DEBUG IF DURATION AND END
META DATA FIELDS DEEMED IMPORTANT

Known Problem: Estimated utterance durations are being exaggerated due 
to the fact that consecutive utterances can share a timestamp if derived 
from the same transcript line (multiple sentences in one transcript 
line each become one utterance sharing the same timestamp). An 
utterance's duration is estimated by considering the start time of the 
following utterance, so start times should not be the same unless the utterance syllable count is negligibly small. 
"""

# Define End and Duration meta fields for utterances
def approx_utterance_periods(utt_list, pause_secs=0.6):
    rates = approx_speech_rates(utt_list)
    speaker_rates = {}

    # organize utt speech rates by speaker
    for i, utt in enumerate(utt_list):
        sid = utt['speaker']

        if sid in speaker_rates:
            speaker_rates[sid].append(rates[i])
        
        else:
            speaker_rates[sid] = [rates[i]]

    # median of speech rate for each speaker
    speaker_rate_medians = {
        sid: np.median(speaker_rates[sid]) for sid in speaker_rates
    }
    # standard deviation of speech rate for each speaker
    speaker_rate_stdevs = {
        sid: np.std(speaker_rates[sid]) for sid in speaker_rates}

    for i, utt in enumerate(utt_list):
        sid = utt['speaker']

        text = pro.Text(utt['text'])
        syll_count = len(text.syllables())
        # seconds = (syllables) / (syllables per second)
        duration = syll_count / speaker_rate_medians[sid]

        # let the minimum duration be 0.1 seconds
        if duration < 0.1:
            duration = 0.1
        
        start_secs = convert_to_secs(utt['timestamp'])
        end_secs = start_secs + duration

        # default values; only change if adjusted pause length is used
        utt['meta']['Duration'] = convert_to_timestamp(duration)
        utt['meta']['End'] = convert_to_timestamp(end_secs)

        if i == len(utt_list)-1: break
        
        next_utt = utt_list[i+1]
        next_start_secs = convert_to_secs(next_utt['timestamp'])
        pause_period = round(next_start_secs - end_secs, 1)
        
        # pause_secs is a typical pause length between utts
        # If the current pause length (pause_period) before the 
        # next utt is longer than the typical length, try slowing 
        # the speech rate by the speaker's standard deviation in 
        # speech rate to reduce pause length
        # If adjusted pause length is still longer than the typical 
        # pause length, use adjusted pause length
        
        if pause_period <= pause_secs: continue
        
        rate_stdev = speaker_rate_stdevs[sid]
        rate_median = speaker_rate_medians[sid]
        
        adjusted_rate = rate_median - rate_stdev
        adjusted_duration = max(syll_count/adjusted_rate, 0.1)
        adjusted_end_secs = start_secs + adjusted_duration
        adjusted_pause_period = round(
            next_start_secs - adjusted_end_secs, 1
        )
        
        if adjusted_pause_period <= pause_secs: continue

        utt['meta']['Duration'] = convert_to_timestamp(
            adjusted_duration
        )
        utt['meta']['End'] = convert_to_timestamp(
            adjusted_end_secs
        )


# This function only uses timestamps to determine speech rate
# (does not use utt.meta['Duration'])
# Pauses between any two utterances that are longer than 'pause_secs' 
# Seconds will be reduced by 'pause_secs' seconds
# Utterances must be chronologically ordered
# Returns list of estimated utterance speech rates
def approx_speech_rates(utt_list, pause_secs=0.6):
    rates = []

    for i in range(len(utt_list)-1):
        curr_utt = utt_list[i]
        next_utt = utt_list[i+1]
        
        start_secs = convert_to_secs(curr_utt['timestamp'])
        end_secs = convert_to_secs(next_utt['timestamp'])
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
    final_sid = final_utt['speaker']
    final_speaker_rates = []

    for i, rate in enumerate(rates):
        if final_sid == utt_list[i]['speaker']:
            final_speaker_rates.append(rate)
            
    rates.append(np.median(final_speaker_rates))

    return rates


# Consecutive utterances can share a timestamp if derived from the
# same transcript line; this function separates these utterances
# temporally by editing their timestamps
def explode_shared_timestamps(utt_list, pause_secs=0.6):
    utt_list = compress_shared_timestamp_end(utt_list)
    first_share_index = None
    
    for i in range(len(utt_list)-1):
        curr_utt = utt_list[i]
        next_utt = utt_list[i+1]

        next_shares_timestamp = (
            curr_utt['timestamp'] == next_utt['timestamp']
            and curr_utt['speaker'] == next_utt['speaker']
        )

        if next_shares_timestamp and not first_share_index:
            first_share_index = i

        if next_shares_timestamp or not first_share_index: continue
        
        last_share_index = i
        sharing_utts = utt_list[first_share_index:last_share_index+1]

        syll_counts = [
            len(pro.Text(sharing_utt['text']).syllables())
            for sharing_utt in sharing_utts
        ]
        total_sylls = sum(syll_counts)
        syll_percentages = [
            syll_count/total_sylls for syll_count in syll_counts
        ]

        start_secs = convert_to_secs(
            utt_list[first_share_index]['timestamp']
        )
        end_secs = convert_to_secs(next_utt['timestamp'])
        utt_period = end_secs - start_secs

        if utt_period < 0.1:
            total_duration = 0.1

        elif round(utt_period, 1) <= pause_secs:
            total_duration = utt_period

        else:
            total_duration = utt_period - pause_secs

        sub_durations = [
            sp * total_duration for sp in syll_percentages
        ]

        for i in range(len(sharing_utts)-1):
            sharing_utt = sharing_utts[i]
            next_sharing_utt = sharing_utts[i+1]

            start_secs = convert_to_secs(sharing_utt['timestamp'])
            next_sharing_utt['timestamp'] = convert_to_timestamp(
                start_secs + sub_durations[i]
            )

        first_share_index = None
    
    return utt_list

                
def compress_shared_timestamp_end(utt_list):
    text = utt_list[-1]['text']
    trim_count = 0
    
    for i in range(1, len(utt_list)+1):
        curr_utt = utt_list[-i]
        prior_utt = utt_list[-i-1]

        prior_shares_timestamp = (
            curr_utt['timestamp'] == prior_utt['timestamp']
            and curr_utt['speaker'] == prior_utt['speaker']
        )

        if not prior_shares_timestamp: break

        text = prior_utt['text'] + ' ' + text
        trim_count += 1
    
    if trim_count != 0:
        utt_list = utt_list[:-trim_count]
        utt_list[-1]['text'] = text

    return utt_list