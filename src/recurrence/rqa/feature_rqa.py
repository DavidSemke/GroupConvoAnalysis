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

def idea_rqa(corpus, convo, sparsity_check=False):
    data_pts, _ = idea_data_pts(convo, corpus)
    rplot_folder = r'recurrence_plots\rqa\ideas'
    delay = 1
    embeds = [1]

    return epochless_trials(
        data_pts, delay, embeds, convo.id, rplot_folder, sparsity_check
    )


def letter_stream_rqa(convo, sparsity_check=False):
    data_pts = letter_data_pts(convo)
    rplot_folder = r'recurrence_plots\rqa\letter_stream'
    delay = 1
    embeds = [3]

    return epochless_trials(
        data_pts, delay, embeds, convo.id, rplot_folder, sparsity_check
    )


def turn_taking_rqa(convo, epoch_type=None, sparsity_check=False):
    data_pts, _ = turn_taking_data_pts(convo)
    rplot_folder = r'recurrence_plots\rqa\turn-taking'
    delay = 1
    embeds = [4]
    
    return rqa_trials(
        data_pts, delay, embeds, convo.id, rplot_folder, epoch_type,
        sparsity_check
    )


def complete_speech_sampling_rqa(
        convo, epoch_type=None, sparsity_check=False, time_delay=1
):
    data_pts, _ = complete_speech_sampling_data_pts(convo, time_delay)
    rplot_folder = r'recurrence_plots\rqa\speech_sampling\complete'
    delay = 1
    embeds = [4]
    
    return rqa_trials(
        data_pts, delay, embeds, convo.id, rplot_folder, epoch_type,
        sparsity_check
    )


def binary_speech_sampling_rqa(
        convo, epoch_type=None, sparsity_check=False, time_delay=1
):
    data_pts = binary_speech_sampling_data_pts(convo, time_delay)
    rplot_folder = r'recurrence_plots\rqa\speech_sampling\binary'
    delay = 1
    embeds = [9]
    
    return rqa_trials(
        data_pts, delay, embeds, convo.id, rplot_folder, epoch_type,
        sparsity_check
    )


def simult_binary_speech_sampling_rqa(
        convo, epoch_type=None, sparsity_check=False, time_delay=1
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
        data_pts, delay, embeds, convo.id, rplot_folder, epoch_type,
        sparsity_check
    )
        

def convo_stress_rqa(convo, epoch_type=None, sparsity_check=False):
    speakers = list(convo.iter_speakers())
    utt_stresses = speaker_subset_best_stresses(speakers, convo)
    stresses = convo_stresses(convo, utt_stresses)
    data_pts = stress_data_pts(stresses)

    rplot_folder = r'recurrence_plots\rqa\stress\convo'

    delay = 1
    embeds=[9]

    return rqa_trials(
        data_pts, delay, embeds, convo.id, rplot_folder, epoch_type,
        sparsity_check
    )


def dyad_stress_rqa(convo, epoch_type=None, sparsity_check=False):
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
            data_pts, delay, embeds, convo.id, rplot_folder, 
            epoch_type, (s1.id, s2.id), sparsity_check
        )
        
        for trial in trials:
            trial['sid1'] = s1.id
            trial['sid2'] = s2.id
        
        results.extend(trials)
    
    return results


def rqa_trials(
        data_pts, delay, embeds, convo_id, rplot_folder, 
        epoch_type=None, speaker_id_pair=None, sparsity_check=False
):
    
    if not epoch_type:
        trials = epochless_trials(
            data_pts, delay, embeds, convo_id, rplot_folder, 
            speaker_id_pair, sparsity_check
        )
    
    elif epoch_type == 'frame':
        frames = (10, 20)
        trials = frame_epochs_trials(
            data_pts, frames, delay, embeds, sparsity_check
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
            data_pts, size_overlap_pairs, delay, embeds, sparsity_check
        )

    else:
        raise Exception(
            'Parameter epoch_type can only take on values "frame"' 
            + 'and "sliding"'
        )
    
    return trials


def epochless_trials(
        data_pts, delay, embeds, convo_id, rplot_folder, 
        speaker_ids=None, sparsity_check=False
):
    rplot_name = f'rplot_{convo_id}_'

    if speaker_ids:
        for sid in speaker_ids:
            rplot_name += sid + '-'
        
        rplot_name = rplot_name[:-1] + '_'
    
    trials = []

    for embed in embeds:
        trial_rplot_name = rplot_name + f'delay{delay}_embed{embed}.png'
        rplot_path = rf'{rplot_folder}\{trial_rplot_name}'

        out = rqa(data_pts, delay, embed, rplot_path=rplot_path)
        trial =  {'delay': delay, 'embed': embed, 'results': out}
        trials.append(trial)

        if not sparsity_check: continue

        err_on_sparsity(trial)
    
    return trials


def frame_epochs_trials(
        data_pts, frames, delay, embeds, sparsity_check=False
):
    trials = []

    for frame in frames:
        for embed in embeds:  
            out = frame_epochs(data_pts, frame, delay, embed)
            trial = {
                    'frame': frame, 
                    'delay': delay, 
                    'embed': embed, 
                    'results': out
            }
            trials.append(trial)

            if not sparsity_check: continue

            err_on_sparsity(trial, 'frame')
    
    return trials


def sliding_epochs_trials(
        data_pts, size_overlap_pairs, delay, embeds, 
        sparsity_check=False
):
    trials = []

    for size, overlap in size_overlap_pairs:
        for embed in embeds:  
            out = sliding_epochs(
                data_pts, size, overlap, delay, embed
            )
            trial = {
                    'size': size,
                    'overlap': overlap, 
                    'delay': delay, 
                    'embed': embed, 
                    'results': out
            }
            trials.append(trial)

            if not sparsity_check: continue

            err_on_sparsity(trial, 'sliding')

    return trials


def err_on_sparsity(
        trial, epoch_type=None, min_rec_rate=0.001, min_det=0
):
    low_rec_rate_msg = (
        f'Recurrence rate fell to {min_rec_rate}; consider ' 
        + f'decreasing embedding dimension (={trial["embed"]})'
    )
    low_det_msg = (
        f'Determinism fell to {min_det}; consider ' 
        + f'decreasing embedding dimension (={trial["embed"]})'
    )

    if not epoch_type:
        epochs = [trial['results'][0]]
        
    elif epoch_type == 'frame':
        epochs = trial['results']
        addon = f' or increasing frame (={trial["frame"]}%)'
        low_rec_rate_msg += addon
        low_det_msg += addon

    elif epoch_type == 'sliding':
        epochs = trial['results']
        addon = f' or increasing epoch size (={trial["size"]})'
        low_rec_rate_msg += addon
        low_det_msg += addon
    
    rec_too_low = any(
        [True if epoch.recurrence_rate <= min_rec_rate else False 
        for epoch in epochs]
    )

    if rec_too_low:
        raise Exception(low_rec_rate_msg)
        
    det_too_low = any(
        [True if epoch.determinism <= min_det else False 
        for epoch in epochs]
    )

    if det_too_low:
        raise Exception(low_det_msg)