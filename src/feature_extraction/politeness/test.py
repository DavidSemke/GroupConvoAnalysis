from extraction import *
from src.constants import gap_corpus, gap_convos

for convo in gap_convos:
    print()
    print(convo.id.upper())
    print()
    
    print("Speech overlap percentage:", speech_overlap_percentage(convo), "%")
    print("Contrast in formality, sentence level (WCV):", contrast_in_formality(convo, gap_corpus))
    print("Contrast in formality, word level (WCV):", contrast_in_formality(convo, gap_corpus, True))

    word_pos_ratio, word_neg_ratio = sentiment_ratios(convo, gap_corpus, word_level=True)
    sent_pos_ratio, sent_neg_ratio = sentiment_ratios(convo, gap_corpus)

    print()
    print("Positive sentiment ratio, sentence level:", sent_pos_ratio)
    print("Positive sentiment ratio, word level:", word_pos_ratio)
    print("Negative sentiment ratio, sentence level:", sent_neg_ratio)
    print("Negative sentiment ratio, word level:", word_neg_ratio)
    print()