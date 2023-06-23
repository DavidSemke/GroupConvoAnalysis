import re
import prosodic as pro
import src.constants as const

# Utterances with a length less than len_cutoff are ignored
# Divide by 0 if len_cutoff is too high
# Returns percentage of utts that each meter managed to get the best parse on and the stress sequences associated with the best parses for each utterance
def speaker_meter_affinity(speaker, convo, len_cutoff=6):
    
    utt_meter_counts = {m:0 for m in const.meters}
    utt_stresses = {}
    total_utts = 0

    for utt in convo.iter_utterances(
            lambda u: u.speaker.id == speaker.id
    ):
        text = re.sub('[^ \w]', '', utt.text).strip()

        if len(text.split()) < len_cutoff: continue

        meter_dict = {}

        for m in const.meters:
            t = pro.Text(text, meter=m)
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
        
        for m in best_ms:
            utt_meter_counts[m] += 1
            utt_stresses
            utt_stresses[utt.id][m] = meter_dict[m]['stress']

    utt_meter_percentages = utt_meter_counts
    
    for k in utt_meter_percentages:
        utt_meter_percentages[k] /= total_utts/100
    
    return utt_meter_percentages, utt_stresses

        
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


# output from speaker_meter_affinity function is input here
def best_utterance_stresses(utt_meter_percentages, utt_stresses):

    best_meter = max(
        utt_meter_percentages, key=utt_meter_percentages.get
    )
    
    for utt_id, meter_dict in utt_stresses.items():

        if len(list(meter_dict.keys())) == 1: continue
        
        utt_stresses[utt_id] = {}
        utt_stresses[utt_id][best_meter] = meter_dict[best_meter]
    
    return utt_stresses


# utt_stresses_list is a list of all utt_stresses dicts
# (one utt_stresses dict per speaker of a convo)
# An utt_stresses dict is created using the utterances_stresses 
# function
def convo_stresses(convo, utt_stresses_list):
    speaker_utt_stresses = {}

    for utt_stresses in utt_stresses_list:
        key = list(utt_stresses.keys())[0]
        # utt ids have format int.str.int (e.g. 1.Pink.1)
        speaker_id = '.'.join(key.split('.')[:2])
        speaker_utt_stresses[speaker_id] = utt_stresses
    
    convo_utt_stresses = {}

    for utt in convo.iter_utterances():
        s_id = utt.speaker.id
        utt_stress = speaker_utt_stresses[s_id].get(utt.id)

        if not utt_stress: continue

        convo_utt_stresses[utt.id] = utt_stress
    
    return convo_utt_stresses
        
        


