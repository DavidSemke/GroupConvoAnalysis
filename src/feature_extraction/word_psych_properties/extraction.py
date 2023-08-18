import numpy as np
from src.utils.stats import within_cluster_variance
from src.feature_extraction.word_psych_properties.psych_property_scores import (
    ratings_matrix
)
from src.feature_extraction.word_psych_properties.liwc import (
    personality_matrix
)


# Get variance for each psych property (4 features)
# The four psych properties are:
    # aoa = age of acquisition
    # cnc = concreteness
    # fam = familiarity
    # img = imageability
# Also returns within cluster variance
def psych_property_score_variances(convo, corpus):
    r_matrix = np.array(ratings_matrix(convo, corpus))
    vars = [
        round(np.var(r_matrix[:, i]), 2) 
        for i in range(r_matrix.shape[1])
    ]
    wcv = round(within_cluster_variance(r_matrix), 2)
    
    return vars, wcv


# Returns individual variances and within cluster variance
def personality_trait_variances(convo, corpus):
    p_matrix = personality_matrix(convo, corpus)
    vars = [
        round(np.var(p_matrix[:, i]), 2)
        for i in range(p_matrix.shape[1])
    ]
    wcv = round(within_cluster_variance(p_matrix), 2)

    return vars, wcv