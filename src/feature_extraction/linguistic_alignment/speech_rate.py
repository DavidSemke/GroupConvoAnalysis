import numpy as np
import prosodic as pro
from src.utils.timestamps import convert_to_secs
from src.utils.filter_utterances import convo_frame

def speaker_median_speech_rate(speaker, convo, frame):
    
    # get first frame % and last frame % of convo utterances
    frame_utts = list(convo_frame(convo, frame))
    medians = []
    for utt_period in frame_utts:
        
        # filter out utterances of other speakers
        utt_period = [utt for utt in utt_period if utt.speaker.id == speaker.id]

        rates = []
        for utt in utt_period:
            
            text = pro.Text(utt.text)
            syll_count = len(text.syllables())
            
            duration_timestamp = utt.meta['Duration']
            duration = convert_to_secs(duration_timestamp)

            rate = syll_count/duration
            rates.append(rate)
        
        if rates:
            medians.append(np.median(rates))

    return medians







