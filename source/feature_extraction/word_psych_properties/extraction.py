from convokit import TextParser
from nltk.stem import WordNetLemmatizer
import numpy as np
from mrc_psych_db import query_mrc_db, avg_ratings
from source.feature_extraction.utils.content_token import is_content_word, lemmatize_content_word
from source.feature_extraction.utils.collections import variance

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
        vars.append(round(variance(ratings_matrix[:, i])))
    
    return vars


# uses speaker vocabulary to calculate scores
def speaker_psych_property_scores(speaker, convo, corpus):
    # filter out utterances not included in convo/do not belong to speaker
    corpus = corpus.filter_utterances_by(lambda u: u.conversation_id == convo.id and u.speaker.id == speaker.id)

    # define parser and parse corpus
    parser = TextParser()
    corpus = parser.transform(corpus)

    # initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    words = []
    for utt in corpus.iter_utterances():
        
        first_word_not_found = True
        for tok_dict in [tok_dict for parsed_dict in utt.meta['parsed'] 
                         for tok_dict in parsed_dict['toks']]:
            
            tok = tok_dict['tok']

            if not tok.isalnum(): continue

            # exclude proper nouns
            is_proper_noun = tok_dict['tag'][0:3] == 'NNP'

            if is_proper_noun: 
                first_word_not_found = False
                continue

            if first_word_not_found:
                first_word_not_found = False
                is_content = is_content_word(tok_dict, parser, True)
            else:
                is_content = is_content_word(tok_dict, parser, False)
            
            if is_content:
                tok = lemmatize_content_word(tok_dict, lemmatizer)
            
            words.append(tok)
    
    ratings = query_mrc_db(words)

    return avg_ratings(ratings)