import matplotlib.pyplot as plt
from convokit import Corpus
import numpy as np
from src.recurrence.data_pts.turn_taking import turn_taking_data_pts
from src.recurrence.rqa.feature_rqa import turn_taking_rqa
from src.utils.filter_convos import convos_by_ags
from src.feature_extraction.linguistic_alignment.extraction import (
    turn_taking_frame_det,
    turn_taking_sliding_det
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


def epoch_determinism_stats(corpus):
    group_labels = ('LOWER', 'UPPER')
    convo_groups = extreme_turn_taking_convos(corpus, 0.5)

    for i, group in enumerate(convo_groups):
        print()
        print('GROUP', group_labels[i])

        frame_dets = []
        sliding_dets = []

        for convo in group:
            det, trial = turn_taking_frame_det(convo)
            frame_dets.append(det)
            frame = trial["frame"]
            print(f'Turn taking frame DET (frame = {frame}):', det)
            det, trial = turn_taking_sliding_det(convo)
            sliding_dets.append(det)
            print(f'Turn taking mean sliding DET:', det)
        
        print()
        print('Frame dets mean:', np.mean(frame_dets))
        print('Frame dets var:', np.var(frame_dets))
        print('Sliding dets mean:', np.mean(sliding_dets))
        print('Sliding dets var:', np.var(sliding_dets))
        print()


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
    
    epoch_determinism_stats(gap_corpus)