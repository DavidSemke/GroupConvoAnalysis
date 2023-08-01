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
import numpy as np

def idea_rqa(corpus, convo):
    data_pts, _ = idea_data_pts(convo, corpus)
    rplot_folder = r'recurrence_plots\rqa\ideas'
    delay = 1
    embeds = [1]

    return epochless_trials(
        data_pts, delay, embeds, convo.id, rplot_folder
    )


def letter_stream_rqa(convo):
    data_pts = letter_data_pts(convo)
    rplot_folder = r'recurrence_plots\rqa\letter_stream'
    delay = 1
    embeds = [3]

    return epochless_trials(
        data_pts, delay, embeds, convo.id, rplot_folder
    )


def turn_taking_rqa(convo, epoch_type=None):
    data_pts, _ = turn_taking_data_pts(convo)
    rplot_folder = r'recurrence_plots\rqa\turn-taking'
    delay = 1
    embeds = [4]
    
    return rqa_trials(
        data_pts, delay, embeds, convo.id, rplot_folder, epoch_type
    )


def complete_speech_sampling_rqa(convo, epoch_type=None, time_delay=1):
    data_pts, _ = complete_speech_sampling_data_pts(convo, time_delay)
    rplot_folder = r'recurrence_plots\rqa\speech_sampling\complete'
    delay = 1
    embeds = [4]
    
    return rqa_trials(
        data_pts, delay, embeds, convo.id, rplot_folder, epoch_type
    )


def binary_speech_sampling_rqa(convo, epoch_type=None, time_delay=1):
    data_pts = binary_speech_sampling_data_pts(convo, time_delay)
    rplot_folder = r'recurrence_plots\rqa\speech_sampling\binary'
    delay = 1
    embeds = [9]
    
    return rqa_trials(
        data_pts, delay, embeds, convo.id, rplot_folder, epoch_type
    )


def simult_binary_speech_sampling_rqa(
        convo, epoch_type=None, time_delay=1
):
    data_pts = simult_binary_speech_sampling_data_pts(
        convo, time_delay
    )
    rplot_folder = (
        r'recurrence_plots\rqa\speech_sampling\simult_binary'
    )
    delay = 1
    embeds = [9]
    
    return rqa_trials(
        data_pts, delay, embeds, convo.id, rplot_folder, epoch_type
    )
        

def convo_stress_rqa(convo, epoch_type=None):
    speakers = list(convo.iter_speakers())
    utt_stresses = speaker_subset_best_stresses(speakers, convo)
    stresses = convo_stresses(convo, utt_stresses)
    data_pts = stress_data_pts(stresses)

    rplot_folder = r'recurrence_plots\rqa\stress\convo'

    delay = 1
    embeds=[9]

    return rqa_trials(
        data_pts, delay, embeds, convo.id, rplot_folder, epoch_type
    )


def dyad_stress_rqa(convo, epoch_type=None):
    speakers = list(convo.iter_speakers())
    speaker_pairs = list(combinations(speakers, 2))

    rplot_folder = r'recurrence_plots\rqa\stress\dyad'
    
    delay = 1
    embeds=[9]
    results = []
    
    for pair in speaker_pairs:
        s1, s2 = pair
        
        meter_affinity = dyad_meter_affinity(s1, s2, convo)
        stresses = best_utterance_stresses(meter_affinity)
        data_pts = stress_data_pts(stresses)

        trials = rqa_trials(
            data_pts, delay, embeds, convo.id, rplot_folder, epoch_type,
            (s1.id, s2.id)
        )
        
        for trial in trials:
            trial['sid1'] = s1.id
            trial['sid2'] = s2.id
        
        results.extend(trials)
    
    return results


def rqa_trials(
        data_pts, delay, embeds, convo_id, rplot_folder, epoch_type=None,
        speaker_id_pair=None
):
    
    if not epoch_type:
        trials = epochless_trials(
            data_pts, delay, embeds, convo_id, rplot_folder, 
            speaker_id_pair
        )
    
    elif epoch_type == 'frame':
        frames = (10, 20)
        trials = frame_epochs_trials(
            data_pts, frames, delay, embeds
        )
            
    elif epoch_type == 'sliding':
        overlap_percentages = (60, 80)
        size_overlap_pairs = []
        
        for op in overlap_percentages:
            data_count = len(data_pts)
            size = round(data_count * 0.25)
            overlap = round(size * op/100)
            size_overlap_pairs.append((size, overlap))
               
        trials = sliding_epochs_trials(
            data_pts, size_overlap_pairs, delay, embeds
        )

    else:
        raise Exception(
            'Parameter epoch_type can only take on values "frame"' 
            + 'and "sliding"'
        )
    
    return trials


def epochless_trials(
        data_pts, delay, embeds, convo_id, rplot_folder, 
        speaker_ids=None
):
    rplot_name = f'rplot_{convo_id}_'

    if speaker_ids:
        for sid in speaker_ids:
            rplot_name += sid + '-'
        
        rplot_name = rplot_name[:-1] + '_'
    
    results = []

    for embed in embeds:
        trial_rplot_name = rplot_name + f'delay{delay}_embed{embed}.png'
        rplot_path = rf'{rplot_folder}\{trial_rplot_name}'

        out = rqa(data_pts, delay, embed, rplot_path=rplot_path)
        results.append(
            {'delay': delay, 'embed': embed, 'results': out}
        )
    
    return results


def frame_epochs_trials(data_pts, frames, delay, embeds):
    results = []

    for frame in frames:
        for embed in embeds:  
            out = frame_epochs(data_pts, frame, delay, embed)
            results.append(
                {
                    'frame': frame, 
                    'delay': delay, 
                    'embed': embed, 
                    'results': out
                }
            )
    
    return results


def sliding_epochs_trials(
        data_pts, size_overlap_pairs, delay, embeds
):
    results = []

    for pair in size_overlap_pairs:
        size, overlap = pair

        for embed in embeds:  
            out = sliding_epochs(
                data_pts, size, overlap, delay, embed
            )
            results.append(
                {
                    'size': size,
                    'overlap': overlap, 
                    'delay': delay, 
                    'embed': embed, 
                    'results': out
                }
            )

            # rec_too_low = any(
            #     [True if epoch.recurrence_rate < 0.001 else False 
            #      for epoch in out]
            # )

            # if rec_too_low:
            #     raise Exception(
            #         'REC fell below 0.1%. Consider increasing '
            #         + f'epoch size (={size}) or decreasing embedding ' 
            #         + f'dimension (={embed})'
            #     )

            # zero_det = any(
            #     [True if epoch.determinism == 0 else False 
            #      for epoch in out]
            # )

            # if zero_det:
            #     raise Exception(
            #         'Determinism fell to zero. Consider increasing '
            #         + f'epoch size (={size}) or decreasing embedding ' 
            #         + f'dimension (={embed})'
            #     )
    
    return results