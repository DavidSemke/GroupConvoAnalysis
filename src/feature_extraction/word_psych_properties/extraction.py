import numpy as np
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
def psych_property_score_variances(convo, corpus):
    r_matrix = np.array(ratings_matrix(convo, corpus))
    vars = [
        round(np.var(r_matrix[:, i]), 2) for i in r_matrix.shape[1]
    ]
    
    return vars


def personality_trait_variances(convo, corpus):
    p_matrix = personality_matrix(convo, corpus)
    vars = [
        round(np.var(p_matrix[:, i]), 2) for i in p_matrix.shape[1]
    ]
    
    return vars