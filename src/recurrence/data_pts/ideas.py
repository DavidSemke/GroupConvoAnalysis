from convokit import TextParser
from nltk.stem import WordNetLemmatizer
from src.utils.token import is_word, idea_word

# returns (data_pts, vocab_words)
# index of a word is the position of the word in vocab_words
def idea_data_pts(convo, corpus):
    corpus = corpus.filter_utterances_by(
        lambda u: u.conversation_id == convo.id
    )

    parser = TextParser()
    corpus = parser.transform(corpus)

    lemmatizer = WordNetLemmatizer()

    data_pts = []
    vocab_words = []
    word_index_dict = {}
    index = 0
    
    for utt in corpus.iter_utterances():
        first_word = True
        
        for tok_dict in [
            tok_dict for parsed_dict in utt.meta['parsed'] 
            for tok_dict in parsed_dict['toks']
        ]:
            
            if not is_word(tok_dict): continue

            idea = idea_word(
                tok_dict, parser, lemmatizer, first_word
            )
            first_word = False
            
            if not idea: continue

            existing_index = word_index_dict.get(idea)

            if existing_index is not None:
                data_pts.append(existing_index)
            
            else:
                word_index_dict[idea] = index
                data_pts.append(index)
                vocab_words.append(idea)
                index += 1

    return data_pts, vocab_words