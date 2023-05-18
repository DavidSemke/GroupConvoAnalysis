from convokit import PolitenessStrategies, TextParser
import numpy as np

def convo_sentiment_matrix(convo, corpus):
    # filter out utterances not included in convo
    corpus = corpus.filter_utterances_by(lambda u: u.conversation_id == convo.id)

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
        # there are 21 politeness fields
        sentiment_vector = np.zeros((21,), dtype=int)
        for utt in convo.get_chronological_utterance_list(lambda u: u.speaker.id == s_id):
            p_counts = utt.meta['politeness_strategies']
            p_counts_vector = np.array(list(p_counts.values()))
            sentiment_vector += p_counts_vector
        
        sentiment_matrix.append(sentiment_vector)

    return np.array(sentiment_matrix)