from convokit import TextParser
from nltk.stem import WordNetLemmatizer
from src.feature_extraction.word_psych_properties.mrc_psych_db import query_mrc_db, avg_ratings
from src.feature_extraction.utils.token import is_content_word, is_word, lemmatize_word

# uses speaker vocabulary to calculate scores
def speaker_psych_property_scores(speaker, convo, corpus):
    # filter out utterances not included in convo and do not belong to speaker
    corpus = corpus.filter_utterances_by(lambda u: u.conversation_id == convo.id and u.speaker.id == speaker.id)

    # define parser and parse corpus
    parser = TextParser()
    corpus = parser.transform(corpus)

    # initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    words = []
    for utt in corpus.iter_utterances():
        
        first_word_not_found = True
        for tok_dict in [
            tok_dict for parsed_dict in utt.meta['parsed'] 
            for tok_dict in parsed_dict['toks']
        ]:
            
            if not is_word(tok_dict): continue

            tok = tok_dict['tok']

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
                tok = lemmatize_word(tok_dict, lemmatizer)
            
            words.append(tok)
    
    ratings = query_mrc_db(words)

    return avg_ratings(ratings)