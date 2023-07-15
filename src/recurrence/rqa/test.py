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

# delay and embed are both 1, so they are omitted from file names
def idea_rqa(verbose=False):
    
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


def letter_stream_rqa(verbose=False):
    
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - LETTER STREAM RQA')
        print()

        rplot_folder = r'recurrence_plots\rqa\letter_stream'
        
        data_pts = letter_data_pts(convo)

        delay = 1
        embed = 3

        print(f'Time Delay = {delay},')
        print(f'Embedding Dimn = {embed}')
        print()

        rplot_path = (
            rf'{rplot_folder}\rplot_{convo.id}_delay{delay}'
            + f'_embed{embed}.png'
        )

        res = rqa(data_pts, embed=embed, rplot_path=rplot_path)
        
        if verbose:
            print(res[0])
            print()


def turn_taking_rqa(verbose=False):
    
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - TURN-TAKING RQA')
        print()

        rplot_folder = r'recurrence_plots\rqa\turn-taking'
        data_pts, _ = turn_taking_data_pts(convo)

        total_speakers = len(convo.get_speaker_ids())

        # time delay not set, so find optimal
        delay = optimal_delay(
            data_pts, max_delay=total_speakers, plot=verbose
        )

        for embed in range(2, total_speakers+1):

            print(f'Time Delay = {delay},')
            print(f'Embedding Dimn = {embed}')
            print()

            rplot_path = (
                rf'{rplot_folder}\rplot_{convo.id}_delay{delay}'
                + f'_embed{embed}.png'
            )

            res = rqa(data_pts, delay, embed, rplot_path=rplot_path)
            
            if verbose:
                print(res[0])
                print()


def complete_speech_sampling_rqa(verbose=False):
    
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - COMPLETE SPEECH SAMPLING RQA')
        print()

        rplot_folder = r'recurrence_plots\rqa\speech_sampling\complete'
        data_pts, _ = complete_speech_sampling_data_pts(convo)

        speech_sampling_rqa(
            convo, data_pts, rplot_folder, verbose
        )


def binary_speech_sampling_rqa(verbose=False):
    
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - BINARY SPEECH SAMPLING RQA')
        print()

        rplot_folder = r'recurrence_plots\rqa\speech_sampling\binary'
        data_pts = binary_speech_sampling_data_pts(convo)

        speech_sampling_rqa(
            convo, data_pts, rplot_folder, verbose
        )


def simult_binary_speech_sampling_rqa(verbose=False):
    
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

        speech_sampling_rqa(
            convo, data_pts, rplot_folder, verbose
        )
        

def speech_sampling_rqa(
        convo, data_pts, rplot_folder, verbose=False
):
    # time delay not set, so find optimal
    delay = optimal_delay(data_pts, 6, plot=verbose)
    # embedding dim not set, so find optimal
    embed = optimal_embed(data_pts, delay, 6, plot=verbose)
    
    print(f'Time Delay = {delay},')
    print(f'Embedding Dimn = {embed}')
    print()

    rplot_path = (
        rf'{rplot_folder}\rplot_{convo.id}_delay{delay}'
        + f'_embed{embed}.png'
    )
    
    res = rqa(data_pts, delay, embed, rplot_path=rplot_path)
    
    if verbose:
        print(res[0])
        print()
    

def convo_stress_rqa(verbose=False):

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

        rplot_folder = r'recurrence_plots\rqa\stress\convo'
        
        # time delay not set, so find optimal
        delay = optimal_delay(data_pts, 6, plot=verbose)

        for embed in (2,4,6):

            rplot_path = (
            rf'{rplot_folder}\rplot_{convo.id}_delay{delay}'
            + f'_embed{embed}.png'
            )
            
            print(f'Time Delay = {delay},')
            print(f'Embedding Dimn = {embed}')
            print()

            res = rqa(data_pts, delay, embed, rplot_path=rplot_path)
            
            if verbose:
                print(res[0])
                print()


def dyad_stress_rqa(verbose=False):

    for convo in gap_convos:
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
            
            rplot_folder = r'recurrence_plots\rqa\stress\dyad'
            
            # time delay not set, so find optimal
            delay = optimal_delay(data_pts, 6, plot=verbose)

            for embed in (2,4,6):

                rplot_path = (
                    rf'{rplot_folder}\rplot_{convo.id}_{s1.id}-{s2.id}'
                    + f'_delay{delay}_embed{embed}.png'
                )

                print(f'Time Delay = {delay},')
                print(f'Embedding Dimn = {embed}')
                print()

                res = rqa(
                    data_pts, delay, embed, rplot_path=rplot_path
                )
                
                if verbose:
                    print(res[0])
                    print()
            

if __name__ == '__main__':
    # idea_rqa()
    # letter_stream_rqa()
    # turn_taking_rqa(True)
    # convo_stress_rqa(True)
    dyad_stress_rqa(True)
    # complete_speech_sampling_rqa(True)
    # binary_speech_sampling_rqa(True)
    # simult_binary_speech_sampling_rqa()