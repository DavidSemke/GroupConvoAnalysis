from src.feature_extraction.rhythm.meter import speaker_meter_affinity
from src.utils.stats import within_cluster_variance
from src.recurrence.rqa.extraction import (
    epoch_rqa_lam,
    epoch_rqa_det, 
    epochless_rqa_stats
)
from src.recurrence.rqa.feature_rqa import (
    binary_speech_sampling_rqa,
    convo_stress_rqa,
    dyad_stress_rqa
)
import numpy as np


def contrast_in_meter_affinity(convo):
    affinity_matrix = []

    for speaker in convo.iter_speakers():
        affinity, _ = speaker_meter_affinity(speaker, convo)

        affinity_matrix.append(
            list(affinity.values())
        )
            
    return within_cluster_variance(affinity_matrix)


# Returns the max mean for frame epoch laminarity and the trial that 
# achieved the max mean
def speech_pause_frame_lam(convo):
    return epoch_rqa_lam(
        binary_speech_sampling_rqa(convo, 'frame')
    )


# Returns the max aggregate score for laminarity diffs between 
# epochs (where epochs are adjacent such that they span the entire 
# time series) and the trial that achieved the score
# The aggregate function takes a list of numbers and outputs a number
# Default aggregate function takes the mean of differences between
# adjacent epochs (late epoch lam - early epoch lam)
def speech_pause_sliding_lam(convo):
    return epoch_rqa_lam(
        binary_speech_sampling_rqa(convo, 'sliding'),
        lambda lams: np.mean(np.diff(lams))
    )


# Returns avg and longest vertical line len for trial with greatest
# laminarity (avg vertical line len = trapping time)
# Uses RQA without epochs (to avoid interrupting vertical lines)
def speech_pause_vertical_stats(convo):
    return epochless_rqa_stats(
        binary_speech_sampling_rqa(convo),
        lambda e: {
            'trapping_time': e.trapping_time,
            'longest_vertical_line': e.longest_vertical_line
        },
        lambda trials: max(
            trials, key=lambda trial: trial['results'][0].laminarity
        )
    )

"""
The following feature extraction functions follow the same patterns
as the above functions (which extract laminarity and vertical line 
stats concerning speech pauses) except they extract determinism and 
diagonal line stats concerning stress in speech.
"""

def convo_stress_frame_det(convo):
    return epoch_rqa_det(convo_stress_rqa(convo, 'frame'))


def convo_stress_sliding_det(convo):
    return epoch_rqa_det(
        convo_stress_rqa(convo, 'sliding'), 
        lambda dets: np.mean(np.diff(dets))
    )


def convo_stress_diagonal_stats(convo):
    return epochless_rqa_stats(
        convo_stress_rqa(convo),
        lambda e: {
            'average_diagonal_line': e.average_diagonal_line,
            'longest_diagonal_line': e.longest_diagonal_line
        },
        lambda trials: max(
            trials, key=lambda trial: trial['results'][0].determinism
        )
    )


def dyad_stress_frame_det(convo):
    return epoch_rqa_det(dyad_stress_rqa(convo, 'frame'))


def dyad_stress_sliding_det(convo):
    return epoch_rqa_det(
        dyad_stress_rqa(convo, 'sliding'), 
        lambda dets: np.mean(np.diff(dets))
    )


def dyad_stress_diagonal_stats(convo):
    return epochless_rqa_stats(
        dyad_stress_rqa(convo),
        lambda e: {
            'average_diagonal_line': e.average_diagonal_line,
            'longest_diagonal_line': e.longest_diagonal_line
        },
        lambda trials: max(
            trials, key=lambda trial: trial['results'][0].determinism
        )
    )