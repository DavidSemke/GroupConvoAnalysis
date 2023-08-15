import numpy as np
from convokit import TextParser, PolitenessStrategies
from nltk.stem import WordNetLemmatizer
from src.utils.timestamps import convert_to_secs
from src.utils.token import content_word, is_word
import src.constants as const


"""
A personality vector consists of feature values indicative of 
personality. It is meant to capture the big 5 personality traits,
    1) Extroversion
    2) Agreeableness
    3) Conscientiousness
    4) Neuroticism
    5) Openness

The personality vector here is constructed via LIWC (Linguistic 
Inquiry and Word Count)
"""


# Each row of returned matrix is a personality vector
# Personality vectors correspond to speakers
def personality_matrix(convo, corpus):
    corpus = corpus.filter_utterances_by(
        lambda u: u.conversation_id == convo.id
    )
    
    parser = TextParser()
    corpus = parser.transform(corpus)

    ps = PolitenessStrategies()
    corpus = ps.transform(corpus, markers=True)

    # initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    personality_matrix = [
        speaker_personality_vector(
            speaker, convo, parser, lemmatizer
        )
        for speaker in convo.iter_speakers()
    ]
    
    return np.array(personality_matrix)


def speaker_personality_vector(speaker, convo, parser, lemmatizer):

    # personality vector includes these features in this order:
    speaking_time = 0
    simple_words = 0
    first_pronouns_sing = 0
    third_pronouns = 0
    articles = 0
    neg_emotion_words = 0
    negations = 0
    
    # total_words not included in the personality vector
    total_words = 0
    for utt in convo.iter_utterances(
        lambda u: u.speaker.id == speaker.id
    ):
        
        speaking_time += convert_to_secs(utt.meta['Duration'])

        first_word = True
        for tok_dict in [
            tok_dict for parsed_dict in utt.meta['parsed'] 
            for tok_dict in parsed_dict['toks']
        ]:
            
            tok = tok_dict['tok']

            # include nt as word - short for 'not'
            if not (is_word(tok_dict) or tok == 'nt'): continue

            total_words += 1
            content = content_word(
                tok_dict, parser, lemmatizer, first_word
            )
            first_word = False
            
            if content:
                tok = content
            elif tok in const.negation_words:
                negations += 1
            elif tok in const.first_pronouns_sing:
                first_pronouns_sing += 1
            elif tok in const.third_pronouns:
                third_pronouns += 1
            elif tok in const.articles:
                articles += 1
            
            # length is calculated after lemmatization
            if len(tok) <= 6:
                simple_words += 1
            
        pol_dict = utt.meta['politeness_markers']
        negatives = pol_dict['politeness_markers_==HASNEGATIVE==']
        neg_emotion_words += len(negatives)

    p_vector = [
        speaking_time,
        simple_words/total_words,
        first_pronouns_sing/total_words,
        third_pronouns/total_words,
        articles/total_words,
        neg_emotion_words/total_words,
        negations/total_words
    ]

    return p_vector
    
            
            







    

