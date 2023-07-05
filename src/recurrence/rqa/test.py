from itertools import combinations
from src.recurrence.rqa.computation import *
from src.recurrence.data_pts.ideas import idea_data_pts
from src.recurrence.data_pts.turn_taking import turn_taking_data_pts
from src.recurrence.data_pts.letters import letter_data_pts
from src.recurrence.data_pts.speech_sampling import (
    complete_speech_sampling_data_pts,
    binary_speech_sampling_data_pts,
    simult_binary_speech_sampling_data_pts
)
from src.recurrence.data_pts.stresses import stress_data_pts
from src.feature_extraction.rhythm.meter import *
from src.constants import gap_corpus, gap_convos

def idea_rqa_test():
    # for convo in gap_convos:
    convo = gap_convos[0]

    print()
    print(f'{convo.id.upper()} - IDEA RQA')
    print()

    data_pts, _ = idea_data_pts(convo, gap_corpus)
    rqa_res, rp_res = idea_rqa(
        data_pts, 1, 
        rf'recurrence_plots\rqa\ideas\rplot_{convo.id}.png'
    )

    print(rqa_res)
    print()


def turn_taking_rqa_test():
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - TURN-TAKING RQA')
        print()

        rplot_folder = r'recurrence_plots\rqa\turn-taking'

        data_pts, _ = turn_taking_data_pts(convo)

        for embed in (1, 2, 3):
            print(f'Embedding Dimn = {embed}:')
            print()

            rplot_path = rf'{rplot_folder}\rplot_{convo.id}_embed{embed}.png'
            
            rqa_res, rp_res = turn_taking_rqa(
                data_pts, embed, rplot_path
            )
            
            print(rqa_res.recurrence_points)
            print()


def letter_stream_rqa_test():
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - LETTER STREAM RQA')
        print()

        rplot_folder = r'recurrence_plots\rqa\letter_stream'

        data_pts = letter_data_pts(convo)

        for embed in (3, 4, 5):
            print(f'Embedding Dimn = {embed}:')
            print()

            rplot_path = rf'{rplot_folder}\rplot_{convo.id}_embed{embed}.png'
            
            rqa_res, rp_res = letter_stream_rqa(
                data_pts, embed, rplot_path
            )
            
            print(rqa_res)
            print()


def speech_sampling_rqa_test():
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - SPEECH SAMPLING RQA')
        print()

        rplot_folder = r'recurrence_plots\rqa\speech_sampling\complete'

        data_pts, _ = complete_speech_sampling_data_pts(convo)
        
        for embed in (3, 4):
            print(f'Embedding Dimn = {embed},')
            print()

            rplot_path = rf'{rplot_folder}\rplot_{convo.id}_embed{embed}.png'
            
            rqa_res, rp_res = speech_sampling_rqa(
                data_pts, embed, rplot_path
            )
            
            print(rqa_res)
            print()


def binary_speech_sampling_rqa_test():
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - BINARY SPEECH SAMPLING RQA')
        print()

        rplot_folder = r'recurrence_plots\rqa\speech_sampling\binary'

        data_pts = binary_speech_sampling_data_pts(convo)

        for embed in (3, 4):
            print(f'Embedding Dimn = {embed}:')
            print()

            rplot_path = rf'{rplot_folder}\rplot_{convo.id}_embed{embed}.png'
            
            rqa_res, rp_res = speech_sampling_rqa(
                data_pts, embed, rplot_path
            )
            
            print(rqa_res)
            print()


def simult_binary_speech_sampling_rqa_test():
    for convo in gap_convos:
        print()
        print(
            f'{convo.id.upper()} - SIMULT BINARY SPEECH SAMPLING RQA'
        )
        print()

        rplot_folder = (
            r'recurrence_plots\rqa\speech_sampling\simult_binary'
        )


        data_pts = simult_binary_speech_sampling_data_pts(convo)

        for embed in (3, 4):
            print(f'Embedding Dimn = {embed}:')
            print()

            rplot_path = rf'{rplot_folder}\rplot_{convo.id}_embed{embed}.png'
            
            rqa_res, rp_res = speech_sampling_rqa(
                data_pts, embed, rplot_path
            )
            
            print(rqa_res)
            print()


def convo_stress_rqa_test(convo):
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


def dyad_stress_rqa_test(convo):
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


if __name__ == '__main__':
    # idea_rqa_test()
    # letter_stream_rqa_test()
    # turn_taking_rqa_test()
    # convo_stress_rqa_test()
    # dyad_stress_rqa_test()
    speech_sampling_rqa_test()
    binary_speech_sampling_rqa_test()
    simult_binary_speech_sampling_rqa_test()