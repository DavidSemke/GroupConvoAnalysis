from extraction import *
from source.constants import gap_corpus, convos

for convo in convos:
    print()
    print(convo.id.upper())
    print()
    print("Avg speech overlap len:", avg_speech_overlap_len(convo))
    print("Contrast in formality (WCV):", contrast_in_formality(convo, gap_corpus))
    print()




