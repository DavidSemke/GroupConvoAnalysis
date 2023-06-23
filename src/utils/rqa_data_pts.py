from convokit import TextParser
from nltk.stem import WordNetLemmatizer
from src.utils.token import is_word, idea_word

def turn_taking_data_pts(convo):
    data_pts = []
    index_to_speaker = []
    speaker_to_index = {}
    new_idx = 0
    last_utt_speaker_id = None

    for utt in convo.iter_utterances():
        utt_speaker_id = utt.speaker.id

        if utt_speaker_id == last_utt_speaker_id: continue

        idx = speaker_to_index.get(utt_speaker_id)

        if idx is not None:
            data_pts.append(idx)
        
        else:
            speaker_to_index[utt_speaker_id] = new_idx
            data_pts.append(new_idx)
            index_to_speaker.append(utt_speaker_id)
            new_idx += 1
        
        last_utt_speaker_id = utt_speaker_id
    
    return data_pts, index_to_speaker


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


def letter_data_pts(convo):
    data_pts = []
    letters = [chr(i) for i in range(97, 123)]
    letter_to_index = {p[0]:p[1] for p in zip(letters, range(26))}

    for utt in convo.iter_utterances():
        words = utt.text.split()

        for word in words:
            
            if not is_word({'tok': word}): continue
            
            word = word.lower()
            data_pts.extend(
                [letter_to_index[letter] for letter in word]
            )
        
    return data_pts


def stress_data_pts(convo_stresses):
    data_pts = []
    position_to_index = {
        'U': 0, 'P': 1, 'S': 2,
        'UU': 3, 'UP': 4, 'US': 5,
        'PP': 6, 'PU': 7, 'PS': 8,
        'SS': 9, 'SU': 10, 'SP': 11
    }

    for utt_id in convo_stresses:
        utt_stresses = list(convo_stresses[utt_id].values())

        if len(utt_stresses) != 1:
            raise Exception(
                'Utterance does not have one stress pattern'
            )

        utt_stress = utt_stresses[0]
        positions = utt_stress.split('|')

        for pos in positions:
            data_pts.append(position_to_index[pos])
        
    return data_pts



        

