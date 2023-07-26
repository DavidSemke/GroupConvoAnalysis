import numpy as np
from src.recurrence.rqa.extraction import (
    epoch_rqa_lam, 
    epochless_rqa_stats
)
from src.recurrence.rqa.feature_rqa import (
    simult_binary_speech_sampling_rqa
)

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