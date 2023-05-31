import numpy as np
from src.feature_extraction.utils.stats import variance

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
        vars.append(round(variance(ratings_matrix[:, i], 2)))
    
    return vars