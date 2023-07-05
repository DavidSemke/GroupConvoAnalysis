from src.utils.primes import generate_primes
from src.utils.timestamps import convert_to_secs
from src.utils.primes import is_prime


# 1 indicates simultaneous speech; 0 otherwise
def simult_binary_speech_sampling_data_pts(convo, time_delay=1):
    data_pts, _ = complete_speech_sampling_data_pts(convo, time_delay)
    data_pts = [
        0 if pt == 1 or is_prime(pt) else 1 for pt in data_pts
    ]

    return data_pts


# 1 indicates speech, 0 indicates pause
def binary_speech_sampling_data_pts(convo, time_delay=1):
    data_pts, _ = complete_speech_sampling_data_pts(convo, time_delay)
    data_pts = [1 if pt != 1 else 0 for pt in data_pts]

    return data_pts


# The following describes how to interpret the data points:
    # If no one is speaking, sample data is 1
    # If a speaker is speaking, sample data is a prime number that 
    # identifies the speaker
    # If multiple speakers are speaking (simultaneous speech), sample 
    # data is the product of prime numbers, where each prime number 
    # identifies a speaker
# time_delay = 1 by default, meaning data is sampled once per sec
def complete_speech_sampling_data_pts(convo, time_delay=1):
    speaker_ids = convo.get_speaker_ids()
    primes = generate_primes(len(speaker_ids))
    speaker_attributes = {
        'primes': {
            sid:primes[i] for i, sid in enumerate(speaker_ids)
        },
        'utts': {sid:[] for sid in speaker_ids},
        'utt_periods': {}
    }
    utts = convo.get_chronological_utterance_list()

    # Organize utts into speaker bins in reverse chronological order
    # Utts that have been scanned are removed from their bin; utts
    # are scanned in chronological order, so removal of an utt always 
    # takes place at the end of a list (bin)
    for utt in reversed(utts):
        speaker_attributes['utts'][utt.speaker.id].append(utt)

    for sid in speaker_ids:
        first_utt = speaker_attributes['utts'][sid][-1]
        
        start_secs = convert_to_secs(first_utt.timestamp)
        end_secs = convert_to_secs(first_utt.meta['End'])

        speaker_attributes['utt_periods'][sid] = (start_secs, end_secs)
    
    data_pts = []
    position_secs = 0

    # Keys are removed from dict speaker_utts when their corresponding
    # list value becomes empty, eventually leading to an empty dict
    while speaker_attributes['utts']:
        sample = sample_position(position_secs, speaker_attributes)
        position_secs += time_delay
        data_pts.append(sample)

    # trim data_pts by removing pause symbols (1's) from both ends of list
    data_pts = trim_pauses(data_pts)

    return data_pts, speaker_attributes['primes']


# time delay is the amount of time between samples
def sample_position(pos, speaker_attributes):
    # if a sample remains 1, that means no utterance was
    # occurring at that moment (which is position_secs)
    sample = 1

    for sid in tuple(speaker_attributes['utts']):
        start_secs, end_secs = speaker_attributes['utt_periods'][sid]
        first_loop = True
        pos_after_utt = False

        while first_loop or pos_after_utt:
            pos_after_utt = pos > end_secs
            first_loop = False
        
            # sampling occurs within utt period
            if start_secs <= pos <= end_secs:
                sample *= speaker_attributes['primes'][sid]
            
            if not pos_after_utt: break

            # utt was completely scanned or skipped; remove it
            speaker_attributes['utts'][sid].pop()

            # remove speaker if speaker has no more utts
            if not speaker_attributes['utts'][sid]: 
                del speaker_attributes['utts'][sid]
                break
                
            next_utt = speaker_attributes['utts'][sid][-1]
            start_secs = convert_to_secs(next_utt.timestamp)
            end_secs = convert_to_secs(next_utt.meta['End'])
            speaker_attributes['utt_periods'][sid] = (
                start_secs, end_secs
            )
        
    return sample


# remove pause samples from the beginning and end of the data pts
def trim_pauses(data_pts):
    trim_start = 0
    for pt in data_pts:
        if pt != 1: break 
        trim_start += 1
    
    trim_end = 0
    for pt in reversed(data_pts):
        if pt != 1: break 
        trim_end += 1
    
    data_pts = data_pts[trim_start:len(data_pts)-trim_end]

    return data_pts