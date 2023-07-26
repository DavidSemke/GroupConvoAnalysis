import numpy as np

# Returns a list of two lists (convos that have low/high ags)
# Parameter std_pad determines how far from the mean a convo's AGS
# (abs group score) should be to be added to either extreme group
# (so std_pad should be nonneg)
# Parameter filter takes a convo and outputs True if it is passable,
# else False
def extreme_convo_groups(corpus, std_pad, filter):
    lower = lambda ags_list: (
        np.mean(ags_list) - np.std(ags_list)*std_pad
    )
    upper = lambda ags_list: (
        np.mean(ags_list) + np.std(ags_list)*std_pad
    )
    extreme_convo_groups = convos_by_ags(corpus, lower, upper)

    for i, group in enumerate(extreme_convo_groups):
        extreme_convo_groups[i] = [
            convo for convo in group if filter(convo)
        ]

    return extreme_convo_groups


# Returns all convos that score an AGS
def convos_by_ags(corpus, under_bound_func=None, over_bound_func=None):
    convos = [
        corpus.get_conversation(convo_id) 
        for convo_id in corpus.get_conversation_ids()
    ]
    ags_list = [convo.meta['AGS'] for convo in convos]

    unbounded_convos = []

    if under_bound_func:
        under_bound_convos = [
            convo for convo in convos 
            if convo.meta['AGS'] < under_bound_func(ags_list)
        ]
        unbounded_convos.append(under_bound_convos)

    if over_bound_func:
        over_bound_convos = [
            convo for convo in convos 
            if convo.meta['AGS'] > over_bound_func(ags_list)
        ]
        unbounded_convos.append(over_bound_convos)
    
    return unbounded_convos