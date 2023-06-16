from meter import speaker_meter_affinity
import src.constants as const

for convo in const.gap_convos:
    print()
    print(f'{convo.id.upper()} - SPEAKER METERS')
    print()

    for s in convo.iter_speakers():
        print(s.id, ':')
        affinity_vector = speaker_meter_affinity(s, convo)

        for m in const.meters:
            print('\t', m, ':', affinity_vector[m], '%')
        
        print()


# meter_dict = {
#     'a': {'stress': 0, 'vc': 2, 'vcc': 1},
#     'b': {'stress': 0, 'vc': 1, 'vcc': 2},
#     'c': {'stress': 0, 'vc': 4, 'vcc': 1},
#     'd': {'stress': 0, 'vc': 9, 'vcc': 2}
# }

# meter_triples = [
#         (
#             meter_dict[k]['vc'],
#             meter_dict[k]['vcc'],  
#             k
#         )  
#         for k in meter_dict
# ]

# print(meter_triples)