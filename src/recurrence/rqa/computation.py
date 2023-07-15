from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.computation import RQAComputation, RPComputation
from pyrqa.image_generator import ImageGenerator
from itertools import combinations
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

# delay and embed are both 1, so they are omitted from file names
def idea_rqa(corpus, convo):
    data_pts, _ = idea_data_pts(convo, corpus)
    rplot_folder = r'recurrence_plots\rqa\ideas'
    rplot_path = rf'{rplot_folder}\rplot_{convo.id}.png'
    
    return rqa(data_pts, rplot_path=rplot_path)


def letter_stream_rqa(convo):
    data_pts = letter_data_pts(convo)
    rplot_folder = r'recurrence_plots\rqa\letter_stream'
    
    delay = 1
    embed = 3

    rplot_path = (
        rf'{rplot_folder}\rplot_{convo.id}_delay{delay}'
        + f'_embed{embed}.png'
    )

    return rqa(data_pts, embed=embed, rplot_path=rplot_path)


def turn_taking_rqa(convo, plot=False):
        data_pts, _ = turn_taking_data_pts(convo)
        rplot_folder = r'recurrence_plots\rqa\turn-taking'
        
        total_speakers = len(convo.get_speaker_ids())

        # time delay not set, so find optimal
        delay = optimal_delay(
            data_pts, max_delay=total_speakers, plot=plot
        )

        results = []

        for embed in range(2, total_speakers+1):

            rplot_path = (
                rf'{rplot_folder}\rplot_{convo.id}_delay{delay}'
                + f'_embed{embed}.png'
            )

            out = rqa(data_pts, delay, embed, rplot_path=rplot_path)
            results.append(
                {'delay': delay, 'embed': embed, 'rqa': out}
            )
        
        return results


def complete_speech_sampling_rqa(convo, plot=False):
    data_pts, _ = complete_speech_sampling_data_pts(convo)
    rplot_folder = r'recurrence_plots\rqa\speech_sampling\complete'

    return speech_sampling_rqa(convo, data_pts, rplot_folder, plot)


def binary_speech_sampling_rqa(convo, plot=False):
    data_pts = binary_speech_sampling_data_pts(convo)
    rplot_folder = r'recurrence_plots\rqa\speech_sampling\binary'

    return speech_sampling_rqa(convo, data_pts, rplot_folder, plot)


def simult_binary_speech_sampling_rqa(convo, plot=False):
    data_pts = simult_binary_speech_sampling_data_pts(convo)
    rplot_folder = (
        r'recurrence_plots\rqa\speech_sampling\simult_binary'
    )
    
    return speech_sampling_rqa(convo, data_pts, rplot_folder, plot)
        

def speech_sampling_rqa(convo, data_pts, rplot_folder, plot=False):
    # time delay not set, so find optimal
    delay = optimal_delay(data_pts, 6, plot=plot)
    # embedding dim not set, so find optimal
    embed = optimal_embed(data_pts, delay, 6, plot=plot)

    rplot_path = (
        rf'{rplot_folder}\rplot_{convo.id}_delay{delay}'
        + f'_embed{embed}.png'
    )

    out = rqa(data_pts, delay, embed, rplot_path=rplot_path)
    
    return {'delay': delay, 'embed': embed, 'rqa': out}
    

def convo_stress_rqa(convo, embeds=(2,4,6), plot=False):
    speakers = list(convo.iter_speakers())
    utt_stresses = speaker_subset_best_stresses(speakers, convo)
    stresses = convo_stresses(convo, utt_stresses)
    data_pts = stress_data_pts(stresses)
    
    rplot_folder = r'recurrence_plots\rqa\stress\convo'
    
    # time delay not set, so find optimal
    delay = optimal_delay(data_pts, 6, plot=plot)

    results = []

    for embed in embeds:

        rplot_path = (
        rf'{rplot_folder}\rplot_{convo.id}_delay{delay}'
        + f'_embed{embed}.png'
        )

        out = rqa(data_pts, delay, embed, rplot_path=rplot_path)
        results.append(
            {'delay': delay, 'embed': embed, 'rqa': out}
        )

    return results 


def dyad_stress_rqa(convo, embeds=(2,4,6), plot=False):
    speakers = list(convo.iter_speakers())
    speaker_pairs = list(combinations(speakers, 2))
    results = []

    for pair in speaker_pairs:
        s1, s2 = pair
        
        meter_affinity = dyad_meter_affinity(s1, s2, convo)
        stresses = best_utterance_stresses(meter_affinity)
        data_pts = stress_data_pts(stresses)
        
        rplot_folder = r'recurrence_plots\rqa\stress\dyad'
        
        # time delay not set, so find optimal
        delay = optimal_delay(data_pts, 6, plot=plot)

        for embed in embeds:

            rplot_path = (
                rf'{rplot_folder}\rplot_{convo.id}_{s1.id}-{s2.id}'
                + f'_delay{delay}_embed{embed}.png'
            )

            out = rqa(data_pts, delay, embed, rplot_path=rplot_path)
            results.append(
                {'delay': delay, 'embed': embed, 'rqa': out}
            )

    return results


def rqa(data_pts, delay=1, embed=1, radius=0.1, rplot_path=None):
    time_series = TimeSeries(
        data_pts, embedding_dimension=embed, time_delay=delay
    )
    settings = Settings(
        time_series, analysis_type=Classic, 
        neighbourhood=FixedRadius(radius)
    )

    results = []

    computation = RQAComputation.create(settings)
    rqa_result = computation.run()
    results.append(rqa_result)

    if rplot_path:
        computation = RPComputation.create(settings)
        rp_result = computation.run()
        ImageGenerator.save_recurrence_plot(
            rp_result.recurrence_matrix_reverse, rplot_path
        )
        results.append(rp_result)

    return results