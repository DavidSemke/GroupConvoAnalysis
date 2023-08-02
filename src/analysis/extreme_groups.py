from src.feature_extraction.dominance.extraction import *
from src.feature_extraction.linguistic_alignment.extraction import *
from src.feature_extraction.rhythm.extraction import *
from src.feature_extraction.word_psych_properties.extraction import *
from src.feature_extraction.linguistic_alignment.extraction import *
from src.utils.filter_convos import extreme_convo_groups
from src.constants import gap_corpus

"""
RQA features ignored here (recurrence folder explores)
"""

# It is assumed that the first argument of the feature_func is 
# the convo
# Parameter *args holds all feature_func arguments except the first
# one, which is the convo (since this varies)
# Parameter convo_filter takes a convo and returns True if convo is 
# passable else False
def extreme_feature_summary(
        corpus, feature_func, *args, convo_filter=None
):
    
    if not convo_filter:
        convo_filter = lambda _: True

    extremes = extreme_convo_groups(corpus, filter=convo_filter)
    groups = {'low': extremes[0], 'high': extremes[1]}

    for key in groups:
        print()
        print('GROUP', key.upper())

        scores = []

        for convo in groups[key]:
            print()
            print('Convo ID -', convo.id)

            score = feature_func(convo, *args)
            scores.append(score)

            print(feature_func.__name__, ':', score)

        print()
        print('Group Stats')
        print()
        print(feature_func.__name__, 'mean:', np.mean(scores))
        print(feature_func.__name__, 'var:', np.var(scores))
        print()


if __name__ == '__main__':
    # dominance
    # extreme_feature_summary(
    #     gap_corpus, speech_distribution_score, gap_corpus
    # )
    # lin alignment
    # extreme_feature_summary(
    #     gap_corpus, speech_rate_convergence, 10
    # )
    extreme_feature_summary(
        gap_corpus, dyad_exchange_distribution_score,
        convo_filter=lambda convo: len(convo.get_speaker_ids()) > 2 
    )