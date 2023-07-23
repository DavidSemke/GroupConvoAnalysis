import numpy as np

# Returns all convos that score an AGS greater than 
# (mean of AGS) + (stdev of AGS) * stdev_pad
def high_ags_convos(corpus, stdev_pad=1):
    ags_list = []
    convos = [
        corpus.get_conversation(convo_id) 
        for convo_id in corpus.get_conversation_ids()
    ]

    for convo in convos:
        ags_list.append(convo.meta['AGS'])
    
    mean = np.mean(ags_list)
    stdev = np.std(ags_list)
    threshold = mean + stdev * stdev_pad
    high_ags_convos = []

    for convo in convos:

        if convo.meta['AGS'] < threshold: continue

        high_ags_convos.append(convo)
    
    return high_ags_convos