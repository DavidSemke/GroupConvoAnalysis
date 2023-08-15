import matplotlib.pyplot as plt
import numpy as np
from src.recurrence.rqa.extraction import (
    epochless_rqa_stats,
    epoch_rqa_lam,
    epoch_rqa_det
)


# Parameter metric_func takes an epoch and returns a dict with one RQA 
# metric value
# Parameter metric_transformer maps a list of RQA metric values to a
# new list of values
def plot_epoch_rqa_metric(
        title, xlabel, ylabel, feature_rqa_func, *args,
        metric_func, metric_transformer=None
):
    
    if not 'sliding' in args:
        raise Exception(
            'Feature RQA func must have epoch_type set to "sliding"'
        )
    
    plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    
    if not metric_transformer:
        metric_transformer = lambda vals: vals
        
    for trial in feature_rqa_func(*args):
        vals = [
            tuple(metric_func(epoch).values())[0]
            for epoch in trial['results']
        ]
        y = metric_transformer(vals)
        x = range(len(y))
        label = (
            f's={trial["size"]},o={trial["overlap"]},'
            + f'e={trial["embed"]}'
        )

        plt.plot(x, y, label=label)

    plt.legend()
    plt.show()


# Summarizes RQA output via RQA metric values and related stats
# Parameter metrics_func takes an epoch as input and outputs RQA metric 
# values of the epoch as a dict
# If output is epochless, RQA output is handled as one big epoch
def epoch_rqa_summary(feature_rqa_trials, epoch_type, metrics_func):
    
    if not epoch_type:

        for i, trial in enumerate(feature_rqa_trials):
            print_trial_metadata(trial, i)
            summarize_epochs([trial['results'][0]], metrics_func)
    
    elif epoch_type in ('frame', 'sliding'):

        for i, trial in enumerate(feature_rqa_trials):
            print_trial_metadata(trial, i)
            summarize_epochs(trial['results'], metrics_func)

    else:
        raise Exception(
            'Parameter epoch_type can only take on values "frame"' 
            + 'and "sliding"'
        )
    

def print_trial_metadata(trial, trial_num):
    print(f'TRIAL {trial_num}')
    print()

    for key, val in trial.items():
        
        if key == 'results': continue
        
        print(key, '=', val)
    
    print()


def summarize_epochs(epochs, metrics_func):

    for i, epoch in enumerate(epochs):
        print(f'\tEPOCH {i}')
        print()
        summary = metrics_func(epoch)

        if summary:

            for key, val in summary.items():
                print(f'\t{key} =', val)
        
        else:
            print('\t...')

        print()


# Parameter convo_groups is a dict of lists, where the dict labels
# groups of convos
def convo_group_rqa_det_summary(convo_groups, feature_rqa_func):
    stats = [
        'frame_dets', 'sliding_dets', 'avg_diagonals', 
        'longest_diagonals'
    ]

    for key in convo_groups:
        print()
        print('GROUP', key.upper())
        stats_dict = {stat:[] for stat in stats}

        for convo in convo_groups[key]:
            print()
            print('Convo ID -', convo.id)

            epochless_trials = feature_rqa_func(convo)
            frame_trials = feature_rqa_func(convo, 'frame')
            sliding_trials = feature_rqa_func(convo, 'sliding')
            
            sliding_aggregate_func = lambda epochs: (
                np.mean(np.diff(epochs))
            )

            frame_det = epoch_det_stats(frame_trials)
            sliding_det = epoch_det_stats(
                sliding_trials, sliding_aggregate_func
            )
            avg_diag, longest_diag = epochless_diagonal_stats(
                epochless_trials
            )

            stats_dict['frame_dets'].append(frame_det)
            stats_dict['sliding_dets'].append(sliding_det)
            stats_dict['avg_diagonals'].append(avg_diag)
            stats_dict['longest_diagonals'].append(longest_diag)
        
        print()
        print('Group Stats')
                
        for stat, val in stats_dict.items():
            print()
            print(stat, 'mean:', np.mean(val))
            print(stat, 'var:', np.var(val))

        print()



def epoch_det_stats(feature_rqa_trials, aggregate_func=np.mean):
    det_score, trial =  epoch_rqa_det(
        feature_rqa_trials, aggregate_func
    )

    if 'frame' in trial:
        frame = trial["frame"]
        print(f'Frame DET (frame={frame}):', det_score)
    
    elif 'size' in trial:
        size = trial['size']
        overlap = trial['overlap']
        print(
            f'Mean sliding DET (size={size}, overlap={overlap}):', 
            det_score
        )
    
    else:
        raise Exception(
            'Parameter feature_rqa_trials must be trials produced by a '
            + 'feature rqa func that uses epochs'
        )
    
    return det_score


def epochless_diagonal_stats(feature_rqa_trials):
    stats, _ = epochless_rqa_stats(
        feature_rqa_trials,
        lambda e: {
            'average_diagonal_line': e.average_diagonal_line,
            'longest_diagonal_line': e.longest_diagonal_line
        },
        lambda trials: max(
            trials, key=lambda trial: trial['results'][0].determinism
        )
    )
    avg_diag = stats['average_diagonal_line']
    longest_diag = stats['longest_diagonal_line']
    
    print('Avg diagonal:', avg_diag)
    print('Longest diagonal:', longest_diag)

    return avg_diag, longest_diag


# Parameter convo_groups is a dict of lists, where the dict labels
# groups of convos
def convo_group_rqa_lam_summary(convo_groups, feature_rqa_func):
    stats = [
        'frame_lams', 'sliding_lams', 'avg_verticals', 
        'longest_verticals'
    ]

    for key in convo_groups:
        print()
        print('GROUP', key.upper())
        stats_dict = {stat:[] for stat in stats}

        for convo in convo_groups[key]:
            print()
            print('Convo ID -', convo.id)

            epochless_trials = feature_rqa_func(convo)
            frame_trials = feature_rqa_func(convo, 'frame')
            sliding_trials = feature_rqa_func(convo, 'sliding')
            
            sliding_aggregate_func = lambda epochs: (
                np.mean(np.diff(epochs))
            )

            frame_lam = epoch_lam_stats(frame_trials)
            sliding_lam = epoch_lam_stats(
                sliding_trials, sliding_aggregate_func
            )
            avg_vert, longest_vert = epochless_vertical_stats(epochless_trials)

            stats_dict['frame_lams'].append(frame_lam)
            stats_dict['sliding_lams'].append(sliding_lam)
            stats_dict['avg_verticals'].append(avg_vert)
            stats_dict['longest_verticals'].append(longest_vert)

        print()
        print('Group Stats')
                
        for stat, val in stats_dict.items():
            print()
            print(stat, 'mean:', np.mean(val))
            print(stat, 'var:', np.var(val))

        print()


def epoch_lam_stats(feature_rqa_trials, aggregate_func=np.mean):
    lam_score, trial =  epoch_rqa_lam(
        feature_rqa_trials, aggregate_func
    )

    if 'frame' in trial:
        frame = trial["frame"]
        print(f'Frame LAM (frame={frame}):', lam_score)
    
    elif 'size' in trial:
        size = trial['size']
        overlap = trial['overlap']
        print(
            f'Mean sliding LAM (size={size}, overlap={overlap}):', 
            lam_score
        )
    
    else:
        raise Exception(
            'Parameter feature_rqa_trials must be trials produced by '
            + 'a feature rqa func that uses epochs'
        )
    
    return lam_score


def epochless_vertical_stats(feature_rqa_trials):
    stats, _ = epochless_rqa_stats(
        feature_rqa_trials,
        lambda e: {
            'trapping_time': e.trapping_time,
            'longest_vertical_line': e.longest_vertical_line
        },
        lambda trials: max(
            trials, key=lambda trial: trial['results'][0].laminarity
        )
    )
    avg_vert = stats['trapping_time']
    longest_vert = stats['longest_vertical_line']
    
    print('Avg vertical:', avg_vert)
    print('Longest vertical:', longest_vert)

    return avg_vert, longest_vert