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
def is_dyad_utterance(utt, speaker1, speaker2):
        
    if utt.speaker.id in (speaker1.id, speaker2.id):

        if not utt.reply_to: return True
        
        reply_sid = '.'.join(utt.reply_to.split('.')[:2])
                
        if reply_sid in (speaker1.id, speaker2.id): return True
            
    return False