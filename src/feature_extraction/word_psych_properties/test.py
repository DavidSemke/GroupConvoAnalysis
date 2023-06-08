from src.feature_extraction.word_psych_properties.psych_property_scores import speaker_psych_property_scores
from extraction import *
from src.constants import gap_corpus, gap_convos

for convo in gap_convos:
    print()
    print(convo.id.upper())
    print()

    print('Speaker psych prop scores:')

    ratings_matrix = []
    for speaker in convo.iter_speakers():
        scores = speaker_psych_property_scores(speaker, convo, gap_corpus)
        ratings_matrix.append(scores)
        rounded_scores = [round(s) for s in scores] 
        print(f'\t{speaker.id}:', rounded_scores)
    
    print()
    print('Psych property variances:')
    
    vars = psych_property_score_variances(ratings_matrix)
    print(f'\tAOA:', vars[0])
    print(f'\tCNC:', vars[1])
    print(f'\tFAM:', vars[2])
    print(f'\tIMG:', vars[3])
    
    print()
    print('Contrast in personality (WCV):',
          constrast_in_personality(convo, gap_corpus)
    )
    print()