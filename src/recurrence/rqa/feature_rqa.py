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

def idea_rqa(corpus, convo):
    data_pts, _ = idea_data_pts(convo, corpus)
    rplot_folder = r'recurrence_plots\rqa\ideas'
    delay = 1
    embeds = (1,)

    return epochless_trials(
        data_pts, delay, embeds, convo.id, rplot_folder
    )


def letter_stream_rqa(convo):
    data_pts = letter_data_pts(convo)
    rplot_folder = r'recurrence_plots\rqa\letter_stream'
    delay = 1
    embeds = (3,)

    return epochless_trials(
        data_pts, delay, embeds, convo.id, rplot_folder
    )


def turn_taking_rqa(convo, epoch_type=None):
    data_pts, _ = turn_taking_data_pts(convo)
    embeds = (2, 3)
    delay = 1
        
    if not epoch_type:
        rplot_folder = r'recurrence_plots\rqa\turn-taking'
        results = epochless_trials(
            data_pts, delay, embeds, convo.id, rplot_folder
        )
    
    elif epoch_type == 'frame':
        frames = (10, 20)
        results = frame_epochs_trials(
            data_pts, frames, delay, embeds
        )
            
    elif epoch_type == 'sliding':
        overlaps = (20, 30)
        size_overlap_pairs = []
        
        for overlap in overlaps:
            size = stretched_epoch_size(len(data_pts), overlap, 50) 
            size_overlap_pairs.append((size, overlap))
               
        results = sliding_epochs_trials(
            data_pts, size_overlap_pairs, delay, embeds
        )

    else:
        raise Exception(
            'Parameter epoch_type can only take on values "frame"' 
            + 'and "sliding"'
        )

    return results


def complete_speech_sampling_rqa(convo, epoch_type=None, time_delay=1):
    data_pts, _ = complete_speech_sampling_data_pts(convo, time_delay)
    rplot_folder = r'recurrence_plots\rqa\speech_sampling\complete'

    return speech_sampling_rqa(
        data_pts, convo.id, rplot_folder, epoch_type
    )


def binary_speech_sampling_rqa(convo, epoch_type=None, time_delay=1):
    data_pts = binary_speech_sampling_data_pts(convo, time_delay)
    rplot_folder = r'recurrence_plots\rqa\speech_sampling\binary'

    return speech_sampling_rqa(
        data_pts, convo.id, rplot_folder, epoch_type
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
    
    return speech_sampling_rqa(
        data_pts, convo.id, rplot_folder, epoch_type
    )
        

def speech_sampling_rqa(
        data_pts, convo_id, rplot_folder, epoch_type=None
):
    delay = 1
    embeds=(4,5,6)
    
    if not epoch_type:
        results = epochless_trials(
            data_pts, delay, embeds, convo_id, rplot_folder
        )
    
    elif epoch_type == 'frame':
        frames = (10, 20)
        results = frame_epochs_trials(
            data_pts, frames, delay, embeds
        )
            
    elif epoch_type == 'sliding':
        overlaps = (10, 15, 20)
        size_overlap_pairs = []
        
        for overlap in overlaps:
            size = stretched_epoch_size(len(data_pts), overlap, 50) 
            size_overlap_pairs.append((size, overlap))
               
        results = sliding_epochs_trials(
            data_pts, size_overlap_pairs, delay, embeds
        )

    else:
        raise Exception(
            'Parameter epoch_type can only take on values "frame"' 
            + 'and "sliding"'
        )
    
    return results
    

def convo_stress_rqa(convo, epoch_type=None):
    speakers = list(convo.iter_speakers())
    utt_stresses = speaker_subset_best_stresses(speakers, convo)
    stresses = convo_stresses(convo, utt_stresses)
    data_pts = stress_data_pts(stresses)
    
    delay = 1
    embeds=(2,4,6)

    if not epoch_type:
        rplot_folder = r'recurrence_plots\rqa\stress\convo'
        results = epochless_trials(
            data_pts, delay, embeds, convo.id, rplot_folder
        )
    
    elif epoch_type == 'frame':
        frames = (10, 20)
        results = frame_epochs_trials(
            data_pts, frames, delay, embeds
        )
            
    elif epoch_type == 'sliding':
        overlaps = (10, 15, 20)
        size_overlap_pairs = []
        
        for overlap in overlaps:
            size = stretched_epoch_size(len(data_pts), overlap, 50) 
            size_overlap_pairs.append((size, overlap))
               
        results = sliding_epochs_trials(
            data_pts, size_overlap_pairs, delay, embeds
        )

    else:
        raise Exception(
            'Parameter epoch_type can only take on values "frame"' 
            + 'and "sliding"'
        )

    return results 


def dyad_stress_rqa(convo, epoch_type=None):
    speakers = list(convo.iter_speakers())
    speaker_pairs = list(combinations(speakers, 2))
    
    delay = 1
    embeds=(2,4,6)
    results = []
    
    for pair in speaker_pairs:
        s1, s2 = pair
        
        meter_affinity = dyad_meter_affinity(s1, s2, convo)
        stresses = best_utterance_stresses(meter_affinity)
        data_pts = stress_data_pts(stresses)

        if not epoch_type:
            rplot_folder = r'recurrence_plots\rqa\stress\dyad'
            trials = epochless_trials(
                data_pts, delay, embeds, convo.id, rplot_folder, 
                (s1.id, s2.id)
            )
            results.append(
                {'sid1': s1.id, 'sid2': s2.id, 'trials': trials})
    
        elif epoch_type == 'frame':
            frames = (10, 20)
            trials = frame_epochs_trials(
                data_pts, frames, delay, embeds
            )
            results.append(
                {'sid1': s1.id, 'sid2': s2.id, 'trials': trials})
                
        elif epoch_type == 'sliding':
            overlaps = (10, 15, 20)
            size_overlap_pairs = []
            
            for overlap in overlaps:
                size = stretched_epoch_size(
                    len(data_pts), overlap, 50
                ) 
                size_overlap_pairs.append((size, overlap))
                
            trials = sliding_epochs_trials(
                data_pts, size_overlap_pairs, delay, embeds
            )
            results.append(
                {'sid1': s1.id, 'sid2': s2.id, 'trials': trials})

        else:
            raise Exception(
                'Parameter epoch_type can only take on values "frame"' 
                + 'and "sliding"'
            )

    return results


def epochless_trials(
        data_pts, delay, embeds, convo_id, rplot_folder, speaker_ids=None
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
    
    return results


# Increases epoch size until excess data excluded from epochs cannot 
# be distributed such that all epochs retain the same size
def stretched_epoch_size(data_count, overlap, min_size):
    epoch_count = data_count // (min_size - overlap)

    if data_count % (min_size - overlap) < overlap:
        epoch_count -= 1
    
    epoch_data_count = (
        min_size * epoch_count - overlap * (epoch_count - 1)
    )
    excess_data_count = data_count - epoch_data_count

    stretched_size = min_size + excess_data_count // epoch_count
    
    return stretched_size