import numpy as np
from meter import speaker_meter_affinity
from src.utils.rqa_data_pts import stress_data_pts
from src.feature_extraction.rhythm.recurrence import stress_rqa
from src.feature_extraction.rhythm.meter import *
from src.utils.stats import within_cluster_variance
import src.constants as const

def main():
    for convo in const.gap_convos:

        print()
        print(f'{convo.id.upper()} - SPEAKER METER VARIANCES')
        print()

        print('Order of meters:', const.meters)
        print()

        affinity_matrix = []
        
        for speaker in convo.iter_speakers():
            affinity, _ = speaker_meter_affinity(speaker, convo)

            affinity_matrix.append(
                list(affinity.values())
            )
        
        affinity_matrix = np.array(affinity_matrix)
        vars = []

        for i in range(len(affinity_matrix[0])):
            vars.append(round(np.var(affinity_matrix[:, i]), 2))

        print('\tVARS:', vars)
        print('\tWCV:', within_cluster_variance(affinity_matrix))


def stress_rqa_test():
    convo = const.gap_convos[0]
    all_utt_stresses = []

    for speaker in convo.iter_speakers():
        utt_meter_ps, utt_stresses = speaker_meter_affinity(
            speaker, convo
        )

        utt_stresses = best_utterance_stresses(
            utt_meter_ps, utt_stresses
        )
        all_utt_stresses.append(utt_stresses)

    stresses = convo_stresses(convo, all_utt_stresses)

    print(stresses)

    data_pts = stress_data_pts(stresses)

    rplot_path = r'recurrence_plots\stress'

    for embed in (3, 4, 5, 6):
        print(f'Embedding Dimn = {embed}:')
        print()
        
        res = stress_rqa(
            data_pts, 
            embed, 
            rf'{rplot_path}\rplot{convo.id}-embed{embed}.png')
        
        print(res)
        print()


if __name__ == '__main__':
    # main()
    stress_rqa_test()