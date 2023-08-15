from src.utils.timestamps import convert_to_secs
from src.utils.stats import within_cluster_variance
from src.feature_extraction.politeness.sentiment import convo_sentiment_matrix
from src.utils.token import content_word_count, content_utterance_count


# calculates differences in politeness among speakers
# if word_level=False, sentence level is used
def contrast_in_formality(convo, corpus, word_level=False):
    sentiment_matrix = convo_sentiment_matrix(convo, corpus, word_level)
    
    return round(within_cluster_variance(sentiment_matrix), 2)


# returns these ratios:
# 1) (neg sentiment units) / (all units)
# 2) (pos sentiment units) / (all units)
# units are content words if word_level=True, else content sentences
# all units = pos units + neg units + neutral units
def sentiment_ratios(convo, corpus, word_level=False):
    
    sentiment_matrix = convo_sentiment_matrix(convo, corpus, word_level)
    
    # 21 columns
    # HASPOSITIVE has col index 17
    # HASNEGATIVE has col index 18
    positive_units_vector = sentiment_matrix[:, 17]
    negative_units_vector = sentiment_matrix[:, 18]

    positive_unit_count = sum(positive_units_vector)
    negative_unit_count = sum(negative_units_vector)

    if word_level:
        all_units = content_word_count(convo, corpus)
        
    else:
        all_units = content_utterance_count(convo, corpus)

    return (round(positive_unit_count/all_units, 4), 
            round(negative_unit_count/all_units, 4))