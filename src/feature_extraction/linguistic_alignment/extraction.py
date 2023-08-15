import numpy as np
from convokit import Coordination
from itertools import combinations
from src.utils.timestamps import convert_to_secs
from src.feature_extraction.linguistic_alignment.speech_flow import *
from src.feature_extraction.linguistic_alignment.idea_flow import idea_flows
from src.utils.filter_utterances import convo_frame
from src.recurrence.rqa.extraction import (
    epoch_rqa_det, 
    epochless_rqa_stats
)
from src.recurrence.rqa.feature_rqa import turn_taking_rqa
from src.utils.filter_utterances import strict_dyad_utterances
from src.utils.token import word_count


def median_idea_discussion_time(convo, corpus):
    flows_dict = idea_flows(convo, corpus)
    
    return median_idea_discussion_time_part(flows_dict)


def median_idea_discussion_time_part(idea_flows_dict):
    times = [convert_to_secs(idea_flow['time_spent']) 
             for key in idea_flows_dict 
             for idea_flow in idea_flows_dict[key]]
    
    return round(np.median(times), 1)


def avg_idea_participation_percentage(convo, corpus):
    flows_dict = idea_flows(convo, corpus)
    avg_ipp = avg_idea_participation_percentage_part(
        convo, flows_dict
    )
    
    return avg_ipp


def avg_idea_participation_percentage_part(
        convo, idea_flows_dict
):
    total_ideas = 0
    total_speakers = len(convo.get_speaker_ids())
    sum_of_fractions = 0
    for key in idea_flows_dict:
        total_ideas += len(idea_flows_dict[key])
        for idea_flow in idea_flows_dict[key]:
            participant_fraction = (
                idea_flow["total_participants"]/total_speakers)
            sum_of_fractions += participant_fraction
    
    return round(100*sum_of_fractions/total_ideas, 2)


def idea_distribution_score(convo, corpus):
    flows_dict = idea_flows(convo, corpus)
    score = idea_distribution_score_part(convo, flows_dict)
    
    return score


def idea_distribution_score_part(convo, idea_flows_dict):
    speaker_ids = convo.get_speaker_ids()
    idea_count_dict = {id:0 for id in speaker_ids}
    
    # get counts of idea flows started for each speaker
    idea_flows = [
        idea_flow for key in idea_flows_dict
        for idea_flow in idea_flows_dict[key]
    ]

    for idea_flow in idea_flows:
        # the speaker at index 0 of participant_ids started the flow
        participant_id = idea_flow['participant_ids'][0]
        idea_count_dict[participant_id] += 1
    
    # convert counts to percentages
    total_idea_flows = len(idea_flows)
    idea_percentage_dict = idea_count_dict
    
    for key in idea_percentage_dict:
        idea_percentage_dict[key] /= total_idea_flows/100

    # compute percentage variance
    percentages = list(idea_percentage_dict.values())
    var = np.var(percentages)

    return round(var, 2)


def dyad_exchange_distribution_score(convo):
    speakers = list(convo.iter_speakers())
    speaker_pairs = list(combinations(speakers, 2))
    word_totals = []

    for s1, s2 in speaker_pairs:
        utts = strict_dyad_utterances(convo, s1, s2)
        word_total = word_count(utts)
        word_totals.append(word_total)
    
    total_sum = sum(word_totals)
    word_percentages = [wt*100/total_sum for wt in word_totals]
    var = np.var(word_percentages)

    return round(var, 2)


# Frame is the first and last x% of utterances considered
# Early variance comes from speaker speech rate variance in first x%
# Late variance is variance in the last x%
# Negative value implies divergence, positive implies convergence
# Returns (early_var - late_var)
def speech_rate_convergence(convo, frame):
    early_utts, late_utts = list(convo_frame(convo, frame))
    early_medians = []
    late_medians = []

    for speaker in convo.iter_speakers():
        speaker_early_utts = [
            utt for utt in early_utts if utt.speaker.id == speaker.id
        ]
        early_rates = speech_rates(speaker_early_utts)

        if early_rates:
            early_medians.append(np.median(early_rates))

        speaker_late_utts = [
            utt for utt in late_utts if utt.speaker.id == speaker.id
        ]
        late_rates = speech_rates(speaker_late_utts)

        if late_rates:
            late_medians.append(np.median(late_rates))
    
    early_var = np.var(early_medians)
    late_var = np.var(late_medians)

    return round(early_var - late_var, 4)


def speech_pause_percentage(convo):
    pause_time = sum(convo_speech_pauses(convo))
    total_time = convo.meta['Meeting Length in Minutes'] * 60

    return round(pause_time/total_time, 2)


# Coordination scores for speakers are calculated, where a speaker
# Has a score for
#   1) how much they coordinate to the other speakers
#   2) how much other speakers coordinate to them
# Returns variance for both sets of scores
def coordination_variances(convo, corpus):
    
    corpus = corpus.filter_utterances_by(
        lambda u: u.conversation_id == convo.id
    )

    coord = Coordination()
    coord.fit(corpus)
    corpus = coord.transform(corpus)

    coord_from_dict = coord.summarize(
        corpus, focus="targets"
    ).averages_by_speaker()
    coord_from_var = np.var(list(coord_from_dict.values()))

    coord_to_dict = coord.summarize(corpus).averages_by_speaker()
    coord_to_var = np.var(list(coord_to_dict.values()))

    return round(coord_to_var, 5), round(coord_from_var, 5)


# Returns the max mean for frame epoch laminarity and the trial that 
# achieved the max mean
def turn_taking_frame_det(convo):
    det, trial = epoch_rqa_det(turn_taking_rqa(convo, 'frame'))

    return round(det, 4), trial


# Returns the max aggregate score for determinism diffs between 
# epochs, where epochs are adjacent such that they span the entire 
# time series
# The aggregate function takes a list of numbers and outputs a number
# Default aggregate function takes the mean of differences between
# adjacent epochs (late epoch det - early epoch det)
def turn_taking_sliding_det(convo):
    det, trial = epoch_rqa_det(
        turn_taking_rqa(convo, 'sliding'), 
        lambda dets: np.mean(np.diff(dets))
    )

    return round(det, 4), trial


# Returns avg and longest diagonal line len for trial with greatest
# determinism
# Uses RQA without epochs (to avoid interrupting diagonal lines)
def turn_taking_diagonal_stats(convo):
    stats, trial = epochless_rqa_stats(
        turn_taking_rqa(convo),
        lambda e: {
            'average_diagonal_line': e.average_diagonal_line,
            'longest_diagonal_line': e.longest_diagonal_line
        },
        lambda trials: max(
            trials, key=lambda trial: trial['results'][0].determinism
        )
    )
    stats['average_diagonal_line'] = round(
        stats['average_diagonal_line'], 2
    )

    return stats, trial