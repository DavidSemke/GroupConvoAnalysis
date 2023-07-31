import numpy as np
from src.recurrence.rqa.extraction import (
    epoch_rqa_lam, 
    epochless_rqa_stats
)
from src.recurrence.rqa.feature_rqa import (
    simult_binary_speech_sampling_rqa
)
from src.utils.token import speaker_word_count

# Ranges from 0 to 1
# A score of 0 means that no speaker spoke more words than another
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
    
    # compute variance where one percentages is 100%, others are 0%
    p_count = len(percentages)
    max_var_data_points = [0 for _ in range(p_count-1)] + [100]
    max_var = np.var(max_var_data_points)

    return round(var/max_var, 4)


# Returns the max mean for frame epoch laminarity and the trial that 
# achieved the max mean
def speech_overlap_frame_lam(convo):
    return epoch_rqa_lam(
        simult_binary_speech_sampling_rqa(convo, 'frame')
    )


# Returns the max aggregate score for laminarity diffs between 
# epochs (where epochs are adjacent such that they span the entire 
# time series) and the trial that achieved the score
# The aggregate function takes a list of numbers and outputs a number
# Default aggregate function takes the mean of differences between
# adjacent epochs (late epoch lam - early epoch lam)
def speech_overlap_sliding_lam(convo):
    return epoch_rqa_lam(
        simult_binary_speech_sampling_rqa(convo, 'sliding'), 
        lambda lams: np.mean(np.diff(lams))
    )


# Returns avg and longest vertical line len for trial with greatest
# laminarity (avg vertical line len = trapping time)
# Uses RQA without epochs (to avoid interrupting vertical lines)
def speech_overlap_vertical_stats(convo):
    return epochless_rqa_stats(
        simult_binary_speech_sampling_rqa(convo),
        lambda e: {
            'trapping_time': e.trapping_time,
            'longest_vertical_line': e.longest_vertical_line
        },
        lambda trials: max(
            trials, key=lambda trial: trial['results'][0].laminarity
        )
    )