import matplotlib.pyplot as plt
from convokit import Corpus
import numpy as np
from src.recurrence.data_pts.turn_taking import turn_taking_data_pts
from src.recurrence.rqa.feature_rqa import turn_taking_rqa
from src.utils.filter_convos import high_ags_convos

def plot_turn_taking_epoch_determinism(convo, diffs=False):
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
    

# get conversations with
# 1) enough turn taking data
# 2) enough speakers (more than 2)
# 3) a high enough AGS score (look for patterns in successful groups)
def turn_taking_convos(corpus, min_data_count=140):
    successful_convos = high_ags_convos(corpus, 0.25)
    turn_taking_convos = []

    for convo in successful_convos:
        data_pts, _ = turn_taking_data_pts(convo)

        if (
            len(data_pts) < min_data_count
            or len(convo.get_speaker_ids()) < 3
        ): 
            continue

        turn_taking_convos.append(convo)

    return turn_taking_convos


if __name__ == '__main__':
    corpus = Corpus('corpora/gap-corpus')
    convos = turn_taking_convos(corpus)

    for convo in convos:
        plot_turn_taking_epoch_determinism(convo, True)