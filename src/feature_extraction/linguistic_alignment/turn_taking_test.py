import matplotlib.pyplot as plt
from convokit import Corpus
import numpy as np
from src.recurrence.data_pts.turn_taking import turn_taking_data_pts
from src.recurrence.rqa.feature_rqa import turn_taking_rqa
from src.utils.filter_convos import convos_by_ags
from src.feature_extraction.linguistic_alignment.extraction import (
    turn_taking_frame_det,
    turn_taking_sliding_det,
    turn_taking_diagonal_stats
)
from src.constants import gap_corpus

def plot_epoch_determinism(convo, diffs=False):
    trials = turn_taking_rqa(convo, 'sliding')

    plt.title(f'Epoch Determinism - {convo.id}')
    plt.ylabel('Determinism')
    plt.xlabel('Epochs')
    dets_transformer = lambda dets: dets

    if diffs:
        plt.xlabel('Diffs')
        dets_transformer = np.diff
        
    for t in trials:
        dets = [epoch.determinism for epoch in t['results']]
        y = dets_transformer(dets)
        x = range(len(y))
        label = f's={t["size"]},o={t["overlap"]},e={t["embed"]}'

        plt.plot(x, y, label=label)

    plt.legend()
    plt.show()


def det_stats(corpus):
    group_labels = ('LOWER', 'UPPER')
    convo_groups = extreme_turn_taking_convos(corpus, 0.5)

    for i, group in enumerate(convo_groups):
        print()
        print('GROUP', group_labels[i])

        frame_dets = []
        sliding_dets = []
        avg_diag_lens = []
        longest_diag_lens = []

        for convo in group:
            print()
            frame_det, sliding_det = epoch_det_stats(convo)
            frame_dets.append(frame_det)
            sliding_dets.append(sliding_det)
            avg_diag, longest_diag = diagonal_stats(convo)
            avg_diag_lens.append(avg_diag)
            longest_diag_lens.append(longest_diag)
        
        print()
        print('Frame dets mean:', np.mean(frame_dets))
        print('Frame dets var:', np.var(frame_dets))
        print('Sliding dets mean:', np.mean(sliding_dets))
        print('Sliding dets var:', np.var(sliding_dets))
        print('Avg diagonal len mean:', np.mean(avg_diag_lens))
        print('Avg diagonal len var:', np.var(avg_diag_lens))
        print(
            'Longest diagonal len mean:', np.mean(longest_diag_lens)
        )
        print('Longest diagonal len var:', np.var(longest_diag_lens))
        print()


def epoch_det_stats(convo):
    frame_det, trial = turn_taking_frame_det(convo)
    frame = trial["frame"]
    print(f'Turn taking frame DET (frame = {frame}):', frame_det)
    sliding_det, trial = turn_taking_sliding_det(convo)
    print(f'Turn taking mean sliding DET:', sliding_det)

    return frame_det, sliding_det


def diagonal_stats(convo):
    avg_diag_len, longest_diag_len = turn_taking_diagonal_stats(convo)
    print('Turn taking avg diagonal len:', avg_diag_len)
    print('Turn taking longest diagonal len:', longest_diag_len)

    return avg_diag_len, longest_diag_len


# Returns a list of two lists (convos that had low/high ags)
# Get conversations with
# 1) enough turn taking data
# 2) enough speakers (more than 2)
# 3) a high/low enough AGS score
# Parameter std_pad determines how far from the mean a convo's AGS
# (abs group score) should be to be added to either extreme group
def extreme_turn_taking_convos(corpus, std_pad, min_data_count=140):
    lower = lambda ags_list: (
        np.mean(ags_list) - np.std(ags_list)*std_pad
    )
    upper = lambda ags_list: (
        np.mean(ags_list) + np.std(ags_list)*std_pad
    )
    convo_groups = convos_by_ags(corpus, lower, upper)
    turn_taking_convos = []

    for group in convo_groups:
        group_convos = []
        
        for convo in group:
            data_pts, _ = turn_taking_data_pts(convo)

            if (
                len(data_pts) < min_data_count
                or len(convo.get_speaker_ids()) < 3
            ): 
                continue

            group_convos.append(convo)
        
        turn_taking_convos.append(group_convos)

    return turn_taking_convos


if __name__ == '__main__':
    # convo_groups = turn_taking_convos(corpus)

    # for i, group in enumerate(convo_groups):
    #     print('GROUP', i)

    #     for convo in group:
    #         plot_epoch_determinism(convo, False)
    
    det_stats(gap_corpus)