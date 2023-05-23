from convokit import TextParser

# True if word in tok_dict is verb, noun, adjective, adverb 
def is_content_word(tok_dict, parser, started_sent):
    
    if not is_content_tag(tok_dict['tag']): return False

    tok = tok_dict['tok']
    vowels = ['a', 'e', 'i', 'o', 'u', 'y']
    contains_vowel = bool([letter for letter in tok if letter in vowels])
    is_acronym = tok.isupper()

    if not (contains_vowel or is_acronym): return False

    is_proper_noun = tok_dict['tag'] == "NNP" 

    # this check skips some words that are mistaken as
    # proper nouns simply due to sentence capitalization
    if started_sent and is_proper_noun:
        # get pos tag for word when not uppercased
        # skip unless lowercase word is a noun/adj/verb/adverb
        utt = parser.transform_utterance(tok.lower())
        tag = utt.meta['parsed'][0]['toks'][0]['tag']

        if not is_content_tag(tag): return False
        
    return True


def is_content_tag(tag):
    two_letter_tag = tag[0:2]
    one_letter_tag = two_letter_tag[0]
    is_content = (one_letter_tag == 'J' 
                  or one_letter_tag == 'N' 
                  or one_letter_tag == 'V'
                  or two_letter_tag == 'RB')
    return is_content


# an utterance is a content utterance if it contains a content word
def is_content_utterance(utt, parser):
    first_word_not_found = True
    for tok_dict in [tok_dict for parsed_dict in utt.meta['parsed'] 
                     for tok_dict in parsed_dict['toks']]:
        if first_word_not_found and tok_dict['tok'].isalnum():
            first_word_not_found = False
            is_content = is_content_word(tok_dict, parser, True)
        else:
            is_content = is_content_word(tok_dict, parser, False)
        
        if is_content: return True
    
    return False


def content_word_count(convo, corpus):
    # filter out utterances not included in convo
    corpus = corpus.filter_utterances_by(lambda u: u.conversation_id == convo.id)

    # parse corpus
    parser = TextParser()
    corpus = parser.transform(corpus)
    
    count = 0
    for utt in corpus.iter_utterances():
        
        first_word_not_found = True
        for tok_dict in [tok_dict for parsed_dict in utt.meta['parsed'] 
                        for tok_dict in parsed_dict['toks']]:
            
            if first_word_not_found and tok_dict['tok'].isalnum():
                first_word_not_found = False
                is_content = is_content_word(tok_dict, parser, True)
            else:
                is_content = is_content_word(tok_dict, parser, False)
        
            if is_content: 
                count+=1
    
    return count


def content_utterance_count(convo, corpus):
    # filter out utterances not included in convo
    corpus = corpus.filter_utterances_by(lambda u: u.conversation_id == convo.id)

    # parse corpus
    parser = TextParser()
    corpus = parser.transform(corpus)

    count = 0
    for utt in corpus.iter_utterances():
        if is_content_utterance(utt, parser): 
            count+=1
    
    return count
    







