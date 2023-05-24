from extraction import *
from source.constants import gap_corpus, convos

for convo in convos:
    print()
    print(convo.id.upper())
    print()

    print('Speaker psych prop scores:')

    for id in convo.get_speaker_ids():
        speaker = convo.get_speaker(id)
        print(f'\t{id}:', speaker_psych_property_scores(speaker, convo, gap_corpus))
    
    print()
