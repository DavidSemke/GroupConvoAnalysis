import numpy as np
from src.feature_extraction.word_psych_properties.psych_property_scores import ratings_matrix
from src.feature_extraction.word_psych_properties.liwc import (
    personality_matrix
)
from src.utils.stats import within_cluster_variance


# Get variance for each psych property (4 features)
# The four psych properties are:
    # aoa = age of acquisition
    # cnc = concreteness
    # fam = familiarity
    # img = imageability
def psych_property_score_variances(convo, corpus):
    r_matrix = np.array(ratings_matrix(convo, corpus))
    vars = []

    for i in range(len(r_matrix[0])):
        vars.append(round(np.var(r_matrix[:, i]), 2))
    
    return vars


def constrast_in_personality(convo, corpus):
    p_matrix = personality_matrix(convo, corpus)
    
    return round(within_cluster_variance(p_matrix), 2)