from convokit import PolitenessStrategies, TextParser
import numpy as np


def convo_sentiment_matrix(convo, corpus, word_level):
    # filter out utterances not included in convo
    corpus = corpus.filter_utterances_by(
        lambda u: u.conversation_id == convo.id
    )

    # parse corpus
    parser = TextParser()
    corpus = parser.transform(corpus)

    # detecting politeness
    ps = PolitenessStrategies()
    corpus = ps.transform(corpus, markers=True)

    sentiment_matrix = []
    ids = convo.get_speaker_ids()
    ids.sort()
    for s_id in ids:
        sentiment_vector = scoped_sentiment_vector(convo, s_id, word_level)
        sentiment_matrix.append(sentiment_vector)

    return np.array(sentiment_matrix)


# scoped means sentiment can be analyzed at sentence or word level
# assumes convo is politeness parsed
def scoped_sentiment_vector(convo, speaker_id, word_level):
    # there are 21 politeness fields
    sentiment_vector = np.zeros((21,), dtype=int)
    for utt in convo.iter_utterances(
        lambda u: u.speaker.id == speaker_id
    ):
        if word_level:
            p_counts = utt.meta['politeness_markers'].values()
            p_counts_vector = np.array(
                [len(markers) for markers in p_counts]
            )
        else:
            p_counts = utt.meta['politeness_strategies'].values()
            p_counts_vector = np.array(list(p_counts))

        sentiment_vector += p_counts_vector
    
    return sentiment_vector