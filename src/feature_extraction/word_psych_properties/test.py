from extraction import *
from src.constants import gap_corpus, gap_convos


for convo in gap_convos:
    print()
    print(convo.id.upper())
    print()

#     print('Psych property variances:')
    
#     vars = psych_property_score_variances(convo, gap_corpus)
#     print(f'\tAOA:', vars[0])
#     print(f'\tCNC:', vars[1])
#     print(f'\tFAM:', vars[2])
#     print(f'\tIMG:', vars[3])
    
    print()
    print('Contrast in personality (WCV):',
          constrast_in_personality(convo, gap_corpus)
    )
    print()