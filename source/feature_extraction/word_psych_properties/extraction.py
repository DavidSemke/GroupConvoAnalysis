from convokit import TextParser
from nltk.stem import WordNetLemmatizer
from mrc_psych_db import query_mrc_db
from source.feature_extraction.utils.content_token import is_content_word, lemmatize_content_word
import numpy as np

"""
aoa = age of acquisition
cnc = concreteness
fam = familiarity
img = imageability

"""

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


def avg_ratings(ratings):
    # cols and rows labels; each starts listing at col/row 0
    # 2 columns: propy_score_total, propy_word_total
    # 4 rows: aoa, cnc, fam, img
    psych_prop_matrix = np.zeros((4, 2), dtype=int)
    
    for rating in ratings:
        # index 0 is the word
        aoa = rating[1]
        cnc = rating[2]
        fam = rating[3]
        img = rating[4]

        props = [aoa, cnc, fam, img]
        for i in range(len(props)):
            
            if props[i] == '-': continue

            psych_prop_matrix[i][0] += int(props[i])
            psych_prop_matrix[i][1] += 1
    
    avgs = []
    for i in range(len(props)):
        avg = round(psych_prop_matrix[i][0]/psych_prop_matrix[i][1])
        avgs.append(avg)

    return avgs




