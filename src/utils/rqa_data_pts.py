from convokit import TextParser
from nltk.stem import WordNetLemmatizer
from src.utils.token import is_word, idea_word
from src.utils.timestamps import convert_to_secs

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


def stress_data_pts(stresses):
    data_pts = []
    position_to_index = {'U': 0, 'S': 1, 'P': 2}

    for utt_id in stresses:
        utt_stresses = list(stresses[utt_id].values())

        if len(utt_stresses) != 1:
            raise Exception(
                'Utterance does not have one stress pattern'
            )

        utt_stress = utt_stresses[0]
        positions = utt_stress.split('|')

        for pos in positions:
            for slot in pos:
                data_pts.append(position_to_index[slot])
        
    return data_pts


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


# data_pts are sampled once per sec
def speech_sampling_data_pts(convo, primes):
    speaker_ids = convo.get_speaker_ids()
    speaker_speech = {
        'primes': {
            sid:primes[i] for i, sid in enumerate(speaker_ids)
        },
        'utts': {sid:[] for sid in speaker_ids},
        'utt_periods': {}
    }
    utts = convo.get_chronological_utterance_list()

    # Organize utts into speaker bins in reverse chronological order
    # Utts that have been scanned are removed from their bin; utts
    # are scanned in chronological order, so removal of an utt always 
    # takes place at the end of a list (bin)
    for utt in reversed(utts):
        speaker_speech['utts'][utt.speaker.id].append(utt)


    for sid in speaker_ids:
        first_utt = speaker_speech['utts'][sid][-1]
        
        start_secs = convert_to_secs(first_utt.timestamp)
        end_secs = convert_to_secs(first_utt.meta['End'])

        speaker_speech['utt_periods'][sid] = (start_secs, end_secs)
    
    data_pts = []
    position_secs = 0

    # Keys are removed from dict speaker_utts when their corresponding
    # list value becomes empty, eventually leading to an empty dict
    while speaker_speech['utts']:
        sample = sample_position(position_secs, speaker_speech)
        position_secs += 1
        data_pts.append(sample)

    # trim data_pts by removing pause symbols (1's) from both ends of list
    data_pts = trim_pauses(data_pts)

    return data_pts, speaker_speech['primes']


def sample_position(pos, speaker_speech):
    # if a sample remains 1, that means no utterance was
    # occurring at that moment (which is position_secs)
    sample = 1

    for sid in tuple(speaker_speech['utts']):
        start_secs, end_secs = speaker_speech['utt_periods'][sid]
        first_loop = True
        pos_after_utt = False

        while first_loop or pos_after_utt:
            pos_after_utt = pos > end_secs
            first_loop = False
        
            # sampling occurs within utt period
            if start_secs <= pos <= end_secs:
                sample *= speaker_speech['primes'][sid]
            
            if not pos_after_utt: break

            # utt was completely scanned or skipped; remove it
            speaker_speech['utts'][sid].pop()

            # remove speaker if speaker has no more utts
            if not speaker_speech['utts'][sid]: 
                del speaker_speech['utts'][sid]
                break
                
            next_utt = speaker_speech['utts'][sid][-1]
            start_secs = convert_to_secs(next_utt.timestamp)
            end_secs = convert_to_secs(next_utt.meta['End'])
            speaker_speech['utt_periods'][sid] = (start_secs, end_secs)
        
    return sample


def trim_pauses(data_pts):
    trim_start = 0
    for pt in data_pts:
        if pt != 1: break 
        trim_start += 1
    
    trim_end = 0
    for pt in reversed(data_pts):
        if pt != 1: break 
        trim_end += 1
    
    data_pts = data_pts[trim_start:len(data_pts)-trim_end]

    return data_pts