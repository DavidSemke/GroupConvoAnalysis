from source.feature_extraction.utils.timestamps import convert_to_secs

# returns seconds
def avg_idea_discussion_time(idea_flows_dict):
    total_ideas = 0
    total_secs = 0
    for key in idea_flows_dict:
        total_ideas += len(idea_flows_dict[key])
        for idea_flow in idea_flows_dict[key]:
            secs = convert_to_secs(idea_flow['time_spent'])
            total_secs += secs
    
    return round(total_secs/total_ideas, 1)


def avg_idea_participation_percentage(convo, idea_flows_dict):
    total_ideas = 0
    sum_of_fractions = 0
    for key in idea_flows_dict:
        total_ideas += len(idea_flows_dict[key])
        for idea_flow in idea_flows_dict[key]:
            participant_fraction = (
                idea_flow["total_participants"]/len(convo.get_speaker_ids()))
            sum_of_fractions += participant_fraction
    
    return round(100*sum_of_fractions/total_ideas, 1)