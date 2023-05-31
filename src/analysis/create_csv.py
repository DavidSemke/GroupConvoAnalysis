import csv
from src.feature_extraction.linguistic_alignment.idea_flow import idea_flows
from src.feature_extraction.linguistic_alignment.extraction import *
from src.feature_extraction.politeness.extraction import *
from src.feature_extraction.word_psych_properties.extraction import *
from src.constants import gap_corpus

def main():
    extraction_funcs = [
        align_features,
        polite_features,
        psych_features
    ]
    
    obsn_matrix, fields = observation_matrix(extraction_funcs, gap_corpus)

    with open('csv/gap_corpus.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows(obsn_matrix)


# returns
# 1) obsn matrix
# 2) column/feature names
def observation_matrix(extraction_funcs, corpus):
    
    obsn_matrix = []
    feat_names = []
    loop_counter = 0
    for convo in corpus.iter_conversations():
        
        obsn = []
        for func in extraction_funcs:
            feats = func(convo, corpus)
            obsn += list(feats.values())
            
            if loop_counter != 0: continue
            
            feat_names += list(feats.keys())
        
        obsn_matrix.append(obsn)
        loop_counter += 1
        print(f"Convo {loop_counter} complete")
    
    return obsn_matrix, feat_names
        

# linguistic alignment
def align_features(convo, corpus):
    flows_dict = idea_flows(convo, corpus)
    
    feats = {
        'median idt': median_idea_discussion_time(flows_dict),
        'avg ipp' : avg_idea_participation_percentage(convo, flows_dict),
        'ids': idea_distribution_score(convo, flows_dict)
    }

    return feats


# politeness
def polite_features(convo, corpus):
    word_pos_ratio, word_neg_ratio = sentiment_ratios(convo, corpus, word_level=True)
    sent_pos_ratio, sent_neg_ratio = sentiment_ratios(convo, corpus)
    
    feats = {
        'avg sol': avg_speech_overlap_len(convo),
        'cf-s': contrast_in_formality(convo, corpus),
        'cf-w': contrast_in_formality(convo, corpus, True),
        'psr-s': sent_pos_ratio,
        'psr-w': word_pos_ratio,
        'nsr-s': sent_neg_ratio,
        'nsr-w': word_neg_ratio
    }

    return feats


# word psych properties
def psych_features(convo, corpus):
    ratings_matrix = []
    for speaker in convo.iter_speakers():
        scores = speaker_psych_property_scores(speaker, convo, corpus)
        ratings_matrix.append(scores)
    
    vars = psych_property_score_variances(ratings_matrix)
    
    feats = {
        'aoa': vars[0],
        'cnc': vars[1],
        'fam': vars[2],
        'img': vars[3]
    }

    return feats


if __name__ == "__main__":
    main()