import random as rand


# Returns the first and last frame % of the convo's utterances,
# where utterances are in chronological order
def convo_frame(convo, frame):

    if not 0 < frame < 50:
        raise Exception(
            'Parameter frame must take a value in range (0, 50)'
        )

    utts = convo.get_chronological_utterance_list()
    total_utts = len(utts)
    frame_utt_count = round(frame/100 * total_utts)
    
    first = utts[:frame_utt_count]
    last = utts[total_utts - frame_utt_count:]

    return first, last


# Returns true if utterance
# 1) comes from speaker 1 and speaker 2 AND
# 2) is directed toward
    # a) the group as a whole (no one in particular)
    # b) either speaker1 or speaker2
# An utterance following an utterance spoken by the same speaker is 
# directed toward the group as a whole
def is_dyad_utterance(utt, speaker1, speaker2):
        
    if utt.speaker.id in (speaker1.id, speaker2.id):

        if not utt.reply_to: return True
        
        reply_sid = '.'.join(utt.reply_to.split('.')[:2])
                
        if reply_sid in (speaker1.id, speaker2.id): return True
            
    return False


# An utterance that follows an utterance is by default labelled as a 
# reply to the speaker of the previous utterance; since utterances are
# also 1 sentence long, utterances can be labelled such that a speaker
# replies to themself
# Strict dyad utterances do not include such utterances unless the 
# chain of self-replies starts with a reply to the other speaker of 
# the dyad
# Utterances that are not replies are excluded
def strict_dyad_utterances(convo, speaker1, speaker2):
    sid1 = speaker1.id
    sid2 = speaker2.id
    replied_to = None
    strict_dyad_utts = []
    
    for utt in convo.iter_utterances():
        sid = utt.speaker.id
        
        if not (sid in (sid1, sid2) and utt.reply_to): continue

        reply_sid = '.'.join(utt.reply_to.split('.')[:2])
        
        # if not a self-reply
        if sid != reply_sid:
            replied_to = reply_sid

        if replied_to in (sid1, sid2):
            strict_dyad_utts.append(utt)
    
    return strict_dyad_utts