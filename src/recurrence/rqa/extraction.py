import numpy as np


def epoch_rqa_lam(feature_rqa_trials, aggregate_func=np.mean):
    l = lambda trial: (
        aggregate_func(
            [epoch.laminarity for epoch in trial['results']]
        )
    )
    best_trial_func = lambda trials: max(trials, key=l)

    stats_dict, best_trial = epoch_rqa_stats(
        feature_rqa_trials,
        lambda e: {'laminarity': e.laminarity},
        best_trial_func,
        aggregate_func
    )

    return stats_dict['laminarity'], best_trial


def epoch_rqa_det(feature_rqa_trials, aggregate_func=np.mean):
    l = lambda trial: (
        aggregate_func(
            [epoch.determinism for epoch in trial['results']]
        )
    )
    best_trial_func = lambda trials: max(trials, key=l)

    stats_dict, best_trial = epoch_rqa_stats(
        feature_rqa_trials,
        lambda e: {'determinism': e.determinism},
        best_trial_func,
        aggregate_func
    )

    return stats_dict['determinism'], best_trial


# Parameter best_trial_func takes a list of rqa trials and returns the
# trial from which stats will be taken
def epoch_rqa_stats(
        feature_rqa_trials, metrics_func, 
        best_trial_func=lambda trials: trials[0], 
        aggregate_func=np.mean
):
    best_trial = best_trial_func(feature_rqa_trials)
    first_loop = True

    for epoch in best_trial['results']:
        epoch_stats = metrics_func(epoch)

        if first_loop:
            trial_stats = {k:[] for k in epoch_stats.keys()}
            first_loop = False
        
        for key, val in epoch_stats.items():
            trial_stats[key].append(val)
    
    for key, val in trial_stats.items():
        trial_stats[key] = aggregate_func(val)
    
    return trial_stats, best_trial
    

def epochless_rqa_stats(
        feature_rqa_trials, metrics_func, 
        best_trial_func=lambda trials: trials[0]
):
    best_trial = best_trial_func(feature_rqa_trials)
    stats = metrics_func(best_trial['results'][0])

    return stats, best_trial