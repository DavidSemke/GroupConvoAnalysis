import numpy as np
from src.feature_extraction.word_psych_properties.liwc import personality_matrix
from src.utils.stats import within_cluster_variance


"""
aoa = age of acquisition
cnc = concreteness
fam = familiarity
img = imageability
"""

# get variance for each psych property (4 features)
def psych_property_score_variances(ratings_matrix):
    ratings_matrix = np.array(ratings_matrix)
    
    vars = []
    for i in range(len(ratings_matrix[0])):
        vars.append(round(np.var(ratings_matrix[:, i]), 2))
    
    return vars


def constrast_in_personality(convo, corpus):
    p_matrix = personality_matrix(convo, corpus)
    
    return round(within_cluster_variance(p_matrix), 2)