from re import sub as re_sub
from prosodic import Text
from random import shuffle as rand_shuffle
from src.utils.filter_utterances import is_dyad_utterance
import src.constants as const


# Utterances with a word length less than len_cutoff are ignored
# Divide by 0 if len_cutoff is too high
# Parameter utt_filter is a boolean func that takes an utterance as 
# an argument
# Returns percentage of utts that each meter managed to get the best 
# parse on and the stress sequences associated with the best parses 
# for each utterance
def convo_meter_affinity(
        convo, utt_percentage=50, utt_filter=None, len_cutoff=6
):
    utt_meter_counts = {m:0 for m in const.meters}
    utt_stresses = {}
    total_utts = 0
    
    utt_filter = utt_filter or (lambda _: True)
    utts = utterance_samples(
        convo, utt_percentage, utt_filter, len_cutoff
    )

    for utt in utts:
        meter_dict = {}

        # parse utt using each meter in constants.py
        for m in const.meters:
            t = Text(utt.text, meter=m)
            t.parse()
            best_ps = t.bestParses()

            if not (best_ps and best_ps[0]): continue

            p = best_ps[0]
            stress = p.str_stress()
            viols_dict = p.violations()
            viol_counts = list(viols_dict.values())
            viol_count = int(sum(viol_counts))
            viol_constraint_count = sum(
                [1 for vc in viol_counts if vc]
            )
            
            meter_dict[m] = {
                'vc': viol_count,
                'vcc': viol_constraint_count,
                'stress': stress
            }
        
        if not meter_dict: continue

        total_utts += 1
        utt_stresses[utt.id] = {}
        best_ms = best_meters(meter_dict)
        
        # Increment meter count and the associated stress pattern
        # for all meters that provided a best parse for the utt
        for m in best_ms:
            utt_meter_counts[m] += 1
            utt_stresses[utt.id][m] = meter_dict[m]['stress']

    utt_meter_percentages = {
        k: utt_meter_counts[k]*100/total_utts for k in utt_meter_counts
    }
    
    return utt_meter_percentages, utt_stresses


# Randomly choose utt_percentage % of utterances
def utterance_samples(
        convo, utt_percentage, utt_filter, len_cutoff
):
    long_utts = []
    
    for utt in convo.iter_utterances(utt_filter):
        text = re_sub(' ?[^ \w]', '', utt.text).strip()

        if len(text.split()) < len_cutoff: continue
        
        utt.text = text
        long_utts.append(utt)
    
    # randomize order of utts to allow random sampling
    rand_shuffle(long_utts)
    
    return long_utts[:round(len(long_utts)*utt_percentage/100)]

    
def best_meters(meter_dict):
    meter_triples = [
        (
            meter_dict[k]['vc'],
            meter_dict[k]['vcc'],  
            k
        )  
        for k in meter_dict
    ]

    meter_triples.sort()
    vc1, vcc1, _ = meter_triples[0]
    best = [
        t[2] for t in meter_triples 
        if t[0] == vc1 and t[1] == vcc1
    ]

    return best


def speaker_meter_affinity(
        speaker, convo, utt_percentage=50, utt_filter=None, 
        len_cutoff=6
):
    utt_filter = utt_filter or (lambda _: True)
    filter = lambda u: speaker.id == u.speaker.id and utt_filter(u)
    
    return convo_meter_affinity(
        convo, utt_percentage, filter, len_cutoff
    )


def dyad_meter_affinity(
        speaker1, speaker2, convo, utt_percentage=50, utt_filter=None, 
        len_cutoff=6
):
    utt_filter = utt_filter or (lambda _: True)
    filter = lambda u: (
        is_dyad_utterance(u, speaker1, speaker2) and utt_filter(u)
    )

    return convo_meter_affinity(
        convo, utt_percentage, filter, len_cutoff
    )


# Ensures that only the best meter is associated with each utt
def best_utterance_stresses(meter_affinity):
    utt_meter_percentages, utt_stresses = meter_affinity

    meter_order = sorted(
        utt_meter_percentages, key=utt_meter_percentages.get
    )

    for uid, meter_dict in utt_stresses.items():

        if len(list(meter_dict.keys())) == 1: continue
        
        best_meter = [m for m in meter_order if m in meter_dict][0]
        
        utt_stresses[uid] = {best_meter: meter_dict[best_meter]}
    
    return utt_stresses


def speaker_subset_best_stresses(
        speakers, convo, utt_percentage=50, utt_filter=None, 
        len_cutoff=6
):
    best_stresses = {}

    for speaker in speakers:
        meter_affinity = speaker_meter_affinity(
            speaker, convo, utt_percentage, utt_filter, len_cutoff
        )
        utt_stresses = best_utterance_stresses(meter_affinity)
        best_stresses[speaker.id] = utt_stresses
    
    return best_stresses


def convo_stresses(convo, speaker_stresses):
    convo_utt_stresses = {}

    for utt in convo.iter_utterances():
        sid = utt.speaker.id
        utt_stress = speaker_stresses[sid].get(utt.id)

        if not utt_stress: continue

        convo_utt_stresses[utt.id] = utt_stress
    
    return convo_utt_stresses