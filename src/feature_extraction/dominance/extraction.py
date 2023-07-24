import numpy as np
from src.recurrence.rqa.feature_rqa import (
    simult_binary_speech_sampling_rqa
)

# Returns the RQA trial that produces the max mean in determinism
# using both early and late frame epochs
def simult_binary_speech_sampling_frame_det(convo):
    trials = simult_binary_speech_sampling_rqa(convo, 'frame')
    l = lambda t: (
        np.mean([epoch.determinism for epoch in t['results']])
    )
    best_trial = max(trials, key=l)
    det_mean = l(best_trial)

    return det_mean, best_trial


# Returns the max aggregate score for determinism diffs between 
# epochs, where epochs are adjacent such that they span the entire 
# time series
# The aggregate function takes a list of numbers and outputs a number
def simult_binary_speech_sampling_sliding_det(
        convo, aggregate_func=np.mean
):
    trials = simult_binary_speech_sampling_rqa(convo, 'sliding')
    l = lambda t: (
        aggregate_func([epoch.determinism for epoch in t['results']]
        )
    )
    best_trial = max(trials, key=l)
    aggregate = l(best_trial)

    return aggregate, best_trial