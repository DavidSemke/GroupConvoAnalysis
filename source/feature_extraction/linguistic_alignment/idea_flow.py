from convokit import TextParser
from nltk.stem import WordNetLemmatizer
from source.feature_extraction.utils.timestamps import add_timestamps
from source.feature_extraction.utils.collections import binary_search

"""
An idea flow is a string of utterances that contain a specific idea, where an idea is a(n)
    1. noun
    2. verb
    3. adjective
"""


def idea_flows(convo, corpus):
    # filter out utterances not included in convo
    corpus = corpus.filter_utterances_by(lambda u: u.conversation_id == convo.id)

    # parse corpus
    parser = TextParser()
    corpus = parser.transform(corpus)

    # initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # J = Adjective, N = Noun, V = Verb
    idea_flows = {"J": [], "N": [], "V": []}
    
    for utt in corpus.iter_utterances():

        expiration_tick(utt, idea_flows)

        for tok_dict in utt.meta['parsed'][0]['toks']:
            short_tag = tok_dict['tag'][0]
            is_idea = short_tag == 'J' or short_tag == 'N' or short_tag == 'V'
            
            if not is_idea: 
                continue
            
            tok = lemmatize_idea_word(tok_dict, short_tag, lemmatizer)
            
            handle_idea_existence(tok, utt, convo, idea_flows[short_tag])

    # get rid of idea flows that failed (never included more than 1 participant)
    successful_idea_flows = {"J": [], "N": [], "V": []}
    for key in successful_idea_flows:
        successful_idea_flows[key] = [idea_flow for idea_flow in idea_flows[key] 
                                      if idea_flow['total_participants'] > 1]

    return successful_idea_flows


def expiration_tick(utt, idea_flows):
    # only decrement toward expiration if the utts are from speakers
    # that did not introduce the idea
    for key in idea_flows:
        for idea_flow in idea_flows[key]:
            
            # if true, total_participants = 1
            if (idea_flow['utts_before_expiry'] > 0  
                and idea_flow['participant_ids'][0] != utt.speaker.id):
                
                idea_flow['utts_before_expiry'] -= 1


def lemmatize_idea_word(tok_dict, one_letter_pos_tag, lemmatizer):
    tok = tok_dict['tok']
    
    is_proper_noun = tok_dict['tag'] == "NNP"
    if is_proper_noun:
        return tok

    tok = tok.lower()
    
    # lemmatize token
    lemmatizing_tag = one_letter_pos_tag.lower()
    if lemmatizing_tag == 'j':
        lemmatizing_tag = 'a'
    return lemmatizer.lemmatize(tok, lemmatizing_tag)


def handle_idea_existence(tok, utt, convo, idea_flows_list):
    # idea_exists is true if idea is repeated
    idea_exists, index = binary_search(idea_flows_list, 'tok', tok)
    if idea_exists:
        idea_flow = idea_flows_list[index]
        expired = idea_flow['utts_before_expiry'] == 0

        if expired:
            # removes previous speaker who tried to introduce the idea
            # and adds the latest speaker that is trying to introduce it
            idea_flow['participant_ids'] = [utt.speaker.id]
            
        curr_speaker_is_participant = utt.speaker.id in idea_flow['participant_ids']
        
        if idea_flow['total_participants'] == 1 and curr_speaker_is_participant:
            idea_flow['time_spent'] = utt.meta['Duration']
            idea_flow['utt_ids'] = [utt.id]
            # reset expiry countdown since idea was reintroduced
            idea_flow['utts_before_expiry'] = len(convo.get_speaker_ids())
        elif curr_speaker_is_participant:
            idea_flow['time_spent'] = add_timestamps(idea_flow['time_spent'], utt.meta['Duration'])
            idea_flow['utt_ids'].append(utt.id)
        else:
            idea_flow['time_spent'] = add_timestamps(idea_flow['time_spent'], utt.meta['Duration'])
            idea_flow['utt_ids'].append(utt.id)
            idea_flow['participant_ids'].append(utt.speaker.id)
            idea_flow['total_participants'] += 1
            # idea flow established, -1 indicates it can no longer expire
            idea_flow['utts_before_expiry'] = -1
    
    else:
        # create idea flow 
        idea_flow = {
            "tok": tok,
            "time_spent": utt.meta['Duration'],
            "total_participants": 1,
            "participant_ids": [utt.speaker.id],
            "utt_ids": [utt.id],
            "utts_before_expiry": len(convo.get_speaker_ids())
        }

        # insert idea flow
        idea_flows_list.insert(index, idea_flow)