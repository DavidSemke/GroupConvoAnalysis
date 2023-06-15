from src.utils.timestamps import convert_to_secs
from src.utils.stats import within_cluster_variance
from src.feature_extraction.politeness.sentiment import convo_sentiment_matrix
from src.utils.token import content_word_count, content_utterance_count

def speech_overlap_percentage(convo):
    all_utts = convo.get_chronological_utterance_list()
    overlap_time = 0
    
    for i in range(len(all_utts)-1):
        curr = all_utts[i]
        next = all_utts[i+1]
        curr_end_time = convert_to_secs(curr.meta["End"])
        next_start_time = convert_to_secs(next.timestamp)

        if curr_end_time > next_start_time:
            overlap_time += curr_end_time - next_start_time
    
    total_time = convo.meta['Meeting Length in Minutes'] * 60

    return round(overlap_time/total_time, 2)


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