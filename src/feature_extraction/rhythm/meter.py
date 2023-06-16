import re
import prosodic as pro
import src.constants as const

def speaker_meter_affinity(speaker, convo):
    utt_meter_counts = {}
    
    for m in const.meters:
        utt_meter_counts.setdefault(m, 0)

    total_utts = 0

    for utt in convo.iter_utterances(
        lambda u: u.speaker.id == speaker.id
    ):
        text = re.sub('[^ \w]', '', utt.text)

        # filter utterances unlikely to reveal meter affinity
        if len(text.split()) < 5: continue

        meter_dict = {}

        for m in const.meters:
            t = pro.Text(text, meter=m)
            t.parse()
            best_ps = t.bestParses()

            if not (best_ps and best_ps[0]): continue

            p = best_ps[0]

            viols_dict = p.violations()
            viol_counts = list(viols_dict.values())
            viol_count = int(sum(viol_counts))
            viol_constraint_count = sum(
                [1 for vc in viol_counts if vc]
            )
            
            meter_dict[m] = {
                'vc': viol_count,
                'vcc': viol_constraint_count
            }
        
        if not meter_dict: continue

        total_utts += 1
        
        best_ms = best_meters(meter_dict)
        
        for m in best_ms:
            utt_meter_counts[m] += 1

    utt_meter_percentages = utt_meter_counts

    for k in utt_meter_percentages:
        utt_meter_percentages[k] /= total_utts/100
    
    return utt_meter_percentages

        
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