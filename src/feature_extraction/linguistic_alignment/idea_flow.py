from convokit import TextParser
from nltk.stem import WordNetLemmatizer
from src.utils.search import binary_search
from src.utils.token import is_word, idea_word


"""
An idea flow is a string of utterances that contain a specific idea, 
where an idea is a(n)
    1. noun
    2. verb
    3. adjective
"""

# If parameter include_time is False, idea flow property time_spent
# will not be calculated
def idea_flows(convo, corpus):
    # filter out utterances not included in convo
    corpus = corpus.filter_utterances_by(
        lambda u: u.conversation_id == convo.id
    )

    # define parser and parse corpus
    parser = TextParser()
    corpus = parser.transform(corpus)

    # initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # J = Adjective, N = Noun, V = Verb
    idea_flows = {"J": [], "N": [], "V": []}
    
    for utt in corpus.iter_utterances():
        
        expiration_tick(utt, idea_flows)
        
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

            handle_idea_existence(
                idea, utt, convo, idea_flows[tok_dict['tag'][0]]
            )

    # get rid of failed idea flows (only have 1 participant)
    successful_idea_flows = {"J": [], "N": [], "V": []}
    for key in successful_idea_flows:
        successful_idea_flows[key] = [
            idea_flow for idea_flow in idea_flows[key] 
            if idea_flow['total_participants'] > 1
        ]

    return successful_idea_flows


def expiration_tick(utt, idea_flows_dict):
    # only decrement toward expiration if the utts are from speakers
    # that did not introduce the idea
    for idea_flow in [
        idea_flow for key in idea_flows_dict 
        for idea_flow in idea_flows_dict[key]
    ]:
        # if first bool is true, total_participants = 1
        if (idea_flow['utts_before_expiry'] > 0  
            and idea_flow['participant_ids'][0] != utt.speaker.id):
            
            idea_flow['utts_before_expiry'] -= 1
    

def handle_idea_existence(tok, utt, convo, idea_flows_list):
    # idea_exists is true if idea is repeated
    idea_exists, index = binary_search(idea_flows_list, 'tok', tok)

    if idea_exists:
        idea_flow = idea_flows_list[index]
        expired = idea_flow['utts_before_expiry'] == 0

        if expired:
            # removes previous speaker who tried to introduce the idea
            # and adds the latest speaker trying to introduce it
            idea_flow['participant_ids'] = [utt.speaker.id]
            
        repeat_speaker = utt.speaker.id in idea_flow['participant_ids']
        
        if idea_flow['total_participants'] == 1 and repeat_speaker:
            idea_flow['utt_ids'] = [utt.id]
            # reset expiry countdown since idea was reintroduced
            idea_flow['utts_before_expiry'] = len(convo.get_speaker_ids())
        
        elif repeat_speaker:
            idea_flow['utt_ids'].append(utt.id)
        
        else:
            idea_flow['utt_ids'].append(utt.id)
            idea_flow['participant_ids'].append(utt.speaker.id)
            idea_flow['total_participants'] += 1
            # idea flow established, -1 indicates it can no longer 
            # expire
            idea_flow['utts_before_expiry'] = -1
    
    else:
        # create idea flow 
        idea_flow = {
            "tok": tok,
            "total_participants": 1,
            "participant_ids": [utt.speaker.id],
            "utt_ids": [utt.id],
            "utts_before_expiry": len(convo.get_speaker_ids())
        }

        # insert idea flow
        idea_flows_list.insert(index, idea_flow)