from itertools import combinations
from src.recurrence.cross_rqa.computation import stress_crqa
from src.recurrence.data_pts.stresses import stress_data_pts
from src.feature_extraction.rhythm.meter import *

def dyad_stress_crqa_test(convo):
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