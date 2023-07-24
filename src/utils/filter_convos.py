import numpy as np

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