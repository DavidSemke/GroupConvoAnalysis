from source.feature_extraction.utils.timestamps import convert_to_secs

# returns seconds
def median_idea_discussion_time(idea_flows_dict):
    times = [convert_to_secs(idea_flow['time_spent']) 
             for key in idea_flows_dict 
             for idea_flow in idea_flows_dict[key]]
    
    times.sort()

    if len(times) % 2 == 0:
        mid_right = len(times) // 2
        mid_left = mid_right - 1
        median = round((mid_left + mid_right)/2, 1)
    else:
        median = len(times) // 2
    
    return median


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