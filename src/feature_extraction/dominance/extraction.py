import numpy as np
from src.recurrence.rqa.extraction import (
    epoch_rqa_lam, 
    epochless_rqa_stats
)
from src.recurrence.rqa.feature_rqa import (
    simult_binary_speech_sampling_rqa
)
from src.utils.token import speaker_word_count
from src.utils.timestamps import convert_to_secs


def speech_overlap_percentage(convo):
    utts = convo.get_chronological_utterance_list()
    overlap_time = 0
    lower_bound = 0
    curr_index = 0
    next_index = 1
    
    while next_index < len(utts):
        curr = utts[curr_index]
        next = utts[next_index]
        curr_end_secs = convert_to_secs(curr.meta["End"])
        next_start_secs = convert_to_secs(next.timestamp)
        next_end_secs = convert_to_secs(next.meta["End"])
        
        lower_bound = max(lower_bound, next_start_secs)

        if curr_end_secs <= next_start_secs:
            curr_index = next_index
            
        elif curr_end_secs < next_end_secs:
            overlap_time += curr_end_secs - lower_bound
            lower_bound = curr_end_secs
            curr_index = next_index

        elif next_end_secs > lower_bound:
            overlap_time += next_end_secs - lower_bound
            lower_bound = next_end_secs

        next_index += 1
    
    total_time = convo.meta['Meeting Length in Minutes'] * 60

    return round(overlap_time/total_time, 2)


def speech_distribution_score(convo, corpus):
    speaker_ids = convo.get_speaker_ids()
    speaker_word_counts = {}

    for id in speaker_ids:
        speaker = convo.get_speaker(id)
        speaker_word_counts[id] = speaker_word_count(
            speaker, convo, corpus
        )
    
    # convert counts to percentages
    total_words = sum(speaker_word_counts.values())
    speaker_word_percentages = speaker_word_counts
    for key in speaker_word_percentages:
        speaker_word_percentages[key] /= total_words/100

    # compute percentage variance
    percentages = list(speaker_word_percentages.values())
    var = np.var(percentages)

    return round(var, 2)


# Returns the max mean for frame epoch laminarity and the trial that 
# achieved the max mean
def speech_overlap_frame_lam(convo):
    lam, trial = epoch_rqa_lam(
        simult_binary_speech_sampling_rqa(convo, 'frame')
    )

    return round(lam, 4), trial 


# Returns the max aggregate score for laminarity diffs between 
# epochs (where epochs are adjacent such that they span the entire 
# time series) and the trial that achieved the score
# The aggregate function takes a list of numbers and outputs a number
# Default aggregate function takes the mean of differences between
# adjacent epochs (late epoch lam - early epoch lam)
def speech_overlap_sliding_lam(convo):
    lam, trial = epoch_rqa_lam(
        simult_binary_speech_sampling_rqa(convo, 'sliding'), 
        lambda lams: np.mean(np.diff(lams))
    )

    return round(lam, 4), trial


# Returns avg and longest vertical line len for trial with greatest
# laminarity (avg vertical line len = trapping time)
# Uses RQA without epochs (to avoid interrupting vertical lines)
def speech_overlap_vertical_stats(convo):
    stats, trial = epochless_rqa_stats(
        simult_binary_speech_sampling_rqa(convo),
        lambda e: {
            'trapping_time': e.trapping_time,
            'longest_vertical_line': e.longest_vertical_line
        },
        lambda trials: max(
            trials, key=lambda trial: trial['results'][0].laminarity
        )
    )
    stats['trapping_time'] = round(stats['trapping_time'], 2)

    return stats, trial