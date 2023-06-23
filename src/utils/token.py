from convokit import TextParser
from nltk.stem import WordNetLemmatizer

def is_word(tok_dict):
    
    tok = tok_dict['tok']

    if not tok.isalnum(): return False

    vowels = ['a', 'e', 'i', 'o', 'u', 'y']
    contains_vowel = bool(
        [letter for letter in tok if letter in vowels]
    )
    is_acronym = tok.isupper()

    if not (contains_vowel or is_acronym): return False

    return True


# True if word in tok_dict is verb, noun, adjective, adverb 
def content_word(tok_dict, parser, lemmatizer, started_sent):
    
    if not is_word(tok_dict): return None
    
    if not is_content_tag(tok_dict['tag']): return None

    tok = tok_dict['tok']
    
    lemma = lemmatize_word(tok_dict, lemmatizer)
    func_verbs = ["be", "have", "do", "shall"]

    if lemma in func_verbs: return None

    is_proper_noun = tok_dict['tag'][0:3] == 'NNP'

    # this check skips some words that are mistaken as
    # proper nouns simply due to sentence capitalization
    if started_sent and is_proper_noun:
        # get pos tag for word when not uppercased
        # skip unless lowercase word is a noun/adj/verb/adverb
        utt = parser.transform_utterance(tok.lower())
        tag = utt.meta['parsed'][0]['toks'][0]['tag']

        if not is_content_tag(tag): return None
        
    return lemma


def is_content_tag(tag):
    two_letter_tag = tag[0:2]
    one_letter_tag = two_letter_tag[0]
    is_content = (one_letter_tag == 'J' 
                  or one_letter_tag == 'N' 
                  or one_letter_tag == 'V'
                  or two_letter_tag == 'RB')
    return is_content


def idea_word(tok_dict, parser, lemmatizer, started_sent):  
    
    # exclude adverbs; adverbs are not idea words
    is_adverb = tok_dict['tag'][0:2] == 'RB'
    
    if is_adverb: return None

    content = content_word(
        tok_dict, parser, lemmatizer, started_sent
    )
    
    return content
    
    
# an utterance is a content utterance if it contains a content word
def is_content_utterance(utt, parser, lemmatizer):
    first_word = True
    
    for tok_dict in [
        tok_dict for parsed_dict in utt.meta['parsed'] 
        for tok_dict in parsed_dict['toks']
    ]:
        
        if not is_word(tok_dict): continue

        content = content_word(
            tok_dict, parser, lemmatizer, first_word
        )
        first_word = False
        
        if content: return True
    
    return False


def is_idea_utterance(utt, parser, lemmatizer):
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
        
        if idea: return True
    
    return False


def content_word_count(convo, corpus):
    corpus = corpus.filter_utterances_by(
        lambda u: u.conversation_id == convo.id
    )

    parser = TextParser()
    corpus = parser.transform(corpus)

    lemmatizer = WordNetLemmatizer()
    
    count = 0

    for utt in corpus.iter_utterances():
        first_word = True
        
        for tok_dict in [
            tok_dict for parsed_dict in utt.meta['parsed'] 
            for tok_dict in parsed_dict['toks']
        ]:
            
            if not is_word(tok_dict): continue

            content = content_word(
                tok_dict, parser, lemmatizer, first_word
            )
            first_word = False
        
            if content: count+=1
    
    return count


def content_utterance_count(convo, corpus):
    # filter out utterances not included in convo
    corpus = corpus.filter_utterances_by(
        lambda u: u.conversation_id == convo.id
    )

    parser = TextParser()
    corpus = parser.transform(corpus)

    lemmatizer = WordNetLemmatizer()

    count = 0
    for utt in corpus.iter_utterances():
        if is_content_utterance(utt, parser, lemmatizer): 
            count+=1
    
    return count


def lemmatize_word(tok_dict, lemmatizer):
    tok = tok_dict['tok']
    
    # if proper noun, no need to lemmatize
    is_proper_noun = tok_dict['tag'] == "NNP"
    if is_proper_noun:
        return tok

    # if proper noun plural, remember capitalized letters
    # so they can be added back after lemmatization
    is_proper_noun_plural = tok_dict['tag'] == "NNPS"
    if is_proper_noun_plural:
        upper_indexes = [
            i for i in range(len(tok)) if tok[i].isupper()
        ]

    tok = tok.lower()
    
    # lemmatize token
    two_letter_tag = tok_dict['tag'][0:2]
    one_letter_tag = two_letter_tag[0]

    if two_letter_tag == 'RB':
        lemmatizing_tag = 'r'
    elif one_letter_tag == 'J':
        lemmatizing_tag = 'a'
    else:
        lemmatizing_tag = one_letter_tag.lower()
    
    lemma = lemmatizer.lemmatize(tok, lemmatizing_tag)

    if is_proper_noun_plural:
        l = lambda char, i: (
            char.upper() if i in upper_indexes else char
        )
        lemma_list = [l(lemma[i], i) for i in range(len(lemma))]
        
        lemma = "".join(lemma_list)

    return lemma