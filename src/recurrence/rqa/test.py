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
from src.recurrence.rqa.optimization import *
from src.constants import gap_corpus, gap_convos

def idea_rqa_test(verbose=False):
    
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - IDEA RQA')
        print()

        rplot_folder = r'recurrence_plots\rqa\ideas'

        data_pts, _ = idea_data_pts(convo, gap_corpus)
        rplot_path = rf'{rplot_folder}\rplot_{convo.id}.png'

        res = rqa(data_pts, rplot_path=rplot_path)
        
        if verbose:
            print(res[0])
            print()


def turn_taking_rqa_test(verbose=False):
    
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - TURN-TAKING RQA')
        print()

        rplot_folder = r'recurrence_plots\rqa\turn-taking'
        
        data_pts, _ = turn_taking_data_pts(convo)

        # time delay not set, so find optimal
        delay = optimal_delay(data_pts, 10, plot=False)
        
        # embedding dim not set, so find optimal
        # embed = optimal_embed(
        #     data_pts, delay, 10, maxnum=260, plot=verbose
        # )

        embed = delay//2
        
        print(f'Time Delay = {delay},')
        print(f'Embedding Dimn = {embed}')
        print()

        rplot_path = (
            rf'{rplot_folder}\rplot_{convo.id}_embed{embed}'
            + f'_delay{delay}.png'
        )

        res = rqa(data_pts, delay, embed, rplot_path=rplot_path)
        
        if verbose:
            print(res[0])
            print()


def letter_stream_rqa_test(verbose=False):
    
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - LETTER STREAM RQA')
        print()

        rplot_folder = r'recurrence_plots\rqa\letter_stream'
        
        data_pts = letter_data_pts(convo)

        # embedding dim not set, so find optimal
        # embed = optimal_embed(data_pts, 1, 5, plot=verbose)

        embed = 3

        print(f'Embedding Dimn = {embed}')
        print()

        rplot_path = (
            rf'{rplot_folder}\rplot_{convo.id}_embed{embed}.png'
        )

        res = rqa(data_pts, embed=embed, rplot_path=rplot_path)
        
        if verbose:
            print(res[0])
            print()


def complete_speech_sampling_rqa_test(verbose=False):
    
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - COMPLETE SPEECH SAMPLING RQA')
        print()

        rplot_folder = r'recurrence_plots\rqa\speech_sampling\complete'
        data_pts, _ = complete_speech_sampling_data_pts(convo)

        speech_sampling_rqa_test(
            convo, data_pts, rplot_folder, verbose
        )


def binary_speech_sampling_rqa_test(verbose=False):
    
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - BINARY SPEECH SAMPLING RQA')
        print()

        rplot_folder = r'recurrence_plots\rqa\speech_sampling\binary'
        data_pts = binary_speech_sampling_data_pts(convo, 0.1)

        speech_sampling_rqa_test(
            convo, data_pts, rplot_folder, verbose
        )


def simult_binary_speech_sampling_rqa_test(verbose=False):
    
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

        speech_sampling_rqa_test(
            convo, data_pts, rplot_folder, verbose
        )
        

def speech_sampling_rqa_test(
        convo, data_pts, rplot_folder, verbose=False
):
    # time delay not set, so find optimal
    delay = optimal_delay(data_pts, 10, plot=verbose)

    # embedding dim not set, so find optimal
    embed = optimal_embed(
        data_pts, delay, 10, plot=verbose
    )

    # embed = delay // 2
    
    print(f'Time Delay = {delay},')
    print(f'Embedding Dimn = {embed}')
    print()

    rplot_path = (
        rf'{rplot_folder}\rplot_{convo.id}_embed{embed}'
        + f'_delay{delay}.png'
    )
    
    res = rqa(data_pts, delay, embed, rplot_path=rplot_path)
    
    if verbose:
        print(res[0])
        print()
    

def convo_stress_rqa_test(verbose=False):

    for convo in gap_convos:
        print()
        print(
            f'{convo.id.upper()} - CONVO STRESS RQA'
        )
        print()

        speakers = list(convo.iter_speakers())
        utt_stresses = speaker_subset_best_stresses(speakers, convo)
        stresses = convo_stresses(convo, utt_stresses)
        data_pts = stress_data_pts(stresses)

        rplot_folder = r'recurrence_plots\rqa\convo_stress'
        
        # time delay not set, so find optimal
        delay = optimal_delay(data_pts, 10, plot=verbose)
        
        # embedding dim not set, so find optimal
        # embed = optimal_embed(
        #     data_pts, delay, 10, plot=verbose
        # )

        embed = delay

        rplot_path = (
        rf'{rplot_folder}\rplot_{convo.id}_embed{embed}'
        + f'_delay{delay}.png'
        )
        
        print(f'Time Delay = {delay},')
        print(f'Embedding Dimn = {embed}')
        print()

        res = rqa(data_pts, delay, embed, rplot_path=rplot_path)
        
        if verbose:
            print(res[0])
            print()


def dyad_stress_rqa_test(verbose=False):

    for convo in [gap_convos[1]]:
        print()
        print(
            f'{convo.id.upper()} - DYAD STRESS RQA'
        )
        print()

        speakers = list(convo.iter_speakers())
        speaker_pairs = list(combinations(speakers, 2))

        for pair in speaker_pairs:
            s1, s2 = pair
            
            meter_affinity = dyad_meter_affinity(s1, s2, convo)
            stresses = best_utterance_stresses(meter_affinity)
            data_pts = stress_data_pts(stresses)
            
            rplot_folder = r'recurrence_plots\rqa\dyad_stress'
            
            # time delay not set, so find optimal
            delay = optimal_delay(data_pts, 10, plot=verbose)

            # embedding dim not set, so find optimal
            # embed = optimal_embed(data_pts, delay, 10, plot=verbose)

            embed = 2

            rplot_path = (
                rf'{rplot_folder}\rplot_{convo.id}_{s1.id}-{s2.id}'
                + f'_embed{embed}_delay{delay}.png'
            )

            print(f'Time Delay = {delay},')
            print(f'Embedding Dimn = {embed}')
            print()

            res = rqa(data_pts, delay, embed, rplot_path=rplot_path)
            
            if verbose:
                print(res[0])
                print()
            

if __name__ == '__main__':
    # idea_rqa_test()
    # letter_stream_rqa_test()
    # turn_taking_rqa_test(True)
    # convo_stress_rqa_test()
    # dyad_stress_rqa_test()
    # complete_speech_sampling_rqa_test()
    binary_speech_sampling_rqa_test(True)
    # simult_binary_speech_sampling_rqa_test()


