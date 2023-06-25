import numpy as np
from itertools import combinations
from meter import speaker_meter_affinity
from src.utils.rqa_data_pts import stress_data_pts
from src.feature_extraction.rhythm.recurrence import *
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


def convo_stress_rqa_test():
    convo = const.gap_convos[0]
    speakers = list(convo.iter_speakers())
    utt_stresses = speaker_subset_best_stresses(speakers, convo)
    stresses = convo_stresses(convo, utt_stresses)

    data_pts = stress_data_pts(stresses)

    rplot_folder = r'recurrence_plots\rqa\convo_stress'

    for embed in (3, 4, 5, 6):
        print()
        print(f'Embedding Dimn = {embed}:')
        print()
        
        rplot_path = rf'{rplot_folder}\rplot_{convo.id}_embed{embed}.png'
            
        rqa_res, rp_res = stress_rqa(
            data_pts, embed, rplot_path
        )
        
        print(rqa_res)
        print()


def dyad_stress_rqa_test():
    convo = const.gap_convos[0]
    speakers = list(convo.iter_speakers())
    speaker_pairs = list(combinations(speakers, 2))

    for pair in speaker_pairs:
        s1, s2 = pair
        meter_affinity = dyad_meter_affinity(s1, s2, convo)
        stresses = best_utterance_stresses(meter_affinity)
        data_pts = stress_data_pts(stresses)
    
        rplot_folder = r'recurrence_plots\rqa\dyad_stress'

        for embed in (3, 4, 5, 6):
            print()
            print(f'Embedding Dimn = {embed}:')
            print()
            
            rplot_path = rf'{rplot_folder}\rplot_{convo.id}_{s1.id}-{s2.id}_embed{embed}.png'
                
            rqa_res, rp_res = stress_rqa(
                data_pts, embed, rplot_path
            )
            
            print(rqa_res)
            print()


def dyad_stress_crqa_test():
    convo = const.gap_convos[0]
    speakers = list(convo.iter_speakers())
    speaker_pairs = list(combinations(speakers, 2))

    for pair in speaker_pairs:
        data_pts = {}
        s1, s2 = pair

        for speaker in pair:
            filter = lambda u: u.speaker.id == speaker.id
            meter_affinity = dyad_meter_affinity(
                s1, s2, convo, filter
            )
            stresses = best_utterance_stresses(meter_affinity)
            data = stress_data_pts(stresses)
            data_pts[speaker.id] = data
    
        rplot_folder = r'recurrence_plots\cross_rqa\dyad_stress'

        for embed in (3, 4, 5, 6):
            print()
            print(f'Embedding Dimn = {embed}:')
            print()
            
            rplot_path = rf'{rplot_folder}\rplot_{convo.id}_{s1.id}-{s2.id}_embed{embed}.png'
                
            rqa_res, rp_res = stress_crqa(
                data_pts[s1.id], data_pts[s2.id], embed, rplot_path
            )
            
            print(rqa_res)
            print()
        

if __name__ == '__main__':
    # main()
    # dyad_stress_rqa_test()
    dyad_stress_crqa_test()