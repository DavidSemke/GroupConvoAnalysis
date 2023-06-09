from meter import speaker_meter_affinity
import src.constants as const

convo = const.gap_convos[0]

print()
print('SPEAKER METERS')
print()

for s in convo.iter_speakers():
    print(s.id, ':')
    dist = speaker_meter_affinity(s, convo)

    for m in const.meters:
        print('\t', m, ':', dist[m], '%')
    
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