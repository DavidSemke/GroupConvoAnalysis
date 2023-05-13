from sentiment import speaker_sentiment_matrix_given_convo
from source.constants import gap_corpus

convo = gap_corpus.get_conversation('1.Pink.1')
print(speaker_sentiment_matrix_given_convo(convo, gap_corpus))