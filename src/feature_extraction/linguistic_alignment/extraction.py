from convokit import Coordination
from src.utils.timestamps import convert_to_secs
from src.utils.stats import variance, median
from src.feature_extraction.linguistic_alignment.speech_rate import speaker_median_speech_rate

# returns seconds
# median is used to avoid influence of outliers
def median_idea_discussion_time(idea_flows_dict):
    times = [convert_to_secs(idea_flow['time_spent']) 
             for key in idea_flows_dict 
             for idea_flow in idea_flows_dict[key]]
    
    return median(times)


def avg_idea_participation_percentage(convo, idea_flows_dict):
    total_ideas = 0
    total_speakers = len(convo.get_speaker_ids())
    sum_of_fractions = 0
    for key in idea_flows_dict:
        total_ideas += len(idea_flows_dict[key])
        for idea_flow in idea_flows_dict[key]:
            participant_fraction = (
                idea_flow["total_participants"]/total_speakers)
            sum_of_fractions += participant_fraction
    
    return round(100*sum_of_fractions/total_ideas, 1)


# ranges from 0 to 1
# a score of 0 means that each speaker started an equal number of idea flows
def idea_distribution_score(convo, idea_flows_dict):
    speaker_ids = convo.get_speaker_ids()
    idea_count_dict = {}
    for id in speaker_ids:
        idea_count_dict[id] = 0
    
    # get counts of idea flows started for each speaker
    idea_flows = [idea_flow for key in idea_flows_dict
                      for idea_flow in idea_flows_dict[key]]
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
    percentages = idea_percentage_dict.values()
    var = variance(percentages)
    
    # compute variance where one percentages is 100%, others are 0%
    max_var_data_points = [0 for i in range(len(percentages)-1)]
    max_var_data_points.append(100)
    max_var = variance(max_var_data_points)

    return round(var/max_var, 4)


# frame is the first and last x% of utterances considered
# early variance comes from speaker speech rate variance in first x%
# late variance is variance in the last x%
# negative value implies divergence, positive implies convergence
# returns (early_var - late_var)
def speech_rate_convergence(convo, frame):
    
    early_medians = []
    late_medians = []
    for speaker in convo.iter_speakers():
        
        medians = speaker_median_speech_rate(speaker, convo, frame)
        early_medians.append(medians[0])
        late_medians.append(medians[1])
    
    early_var = variance(early_medians)
    late_var = variance(late_medians)

    return round(early_var - late_var, 4)


# coordination scores for speakers are calculated, where a speaker
# has a score for
#   1) how much they coordinate to the other speakers
#   2) how much other speakers coordinate to them
# returns variance for both sets of scores
def coordination_variances(convo, corpus):
    
    corpus = corpus.filter_utterances_by(lambda u: u.conversation_id == convo.id)

    coord = Coordination()
    coord.fit(corpus)
    corpus = coord.transform(corpus)

    coord_from_dict = coord.summarize(corpus, focus="targets").averages_by_speaker()
    coord_from_var = variance(list(coord_from_dict.values()))

    coord_to_dict = coord.summarize(corpus).averages_by_speaker()
    coord_to_var = variance(list(coord_to_dict.values()))

    return round(coord_to_var, 6), round(coord_from_var, 6)





