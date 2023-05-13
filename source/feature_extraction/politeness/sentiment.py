from convokit import PolitenessStrategies, TextParser
import numpy as np

def display_conversation_sentiment(convo):
    all_utts = convo.get_chronological_utterance_list(lambda u: str(u.meta['Sentiment']) != "nan")
    for utt in all_utts:
        print()
        print(utt.id, ", ", str(utt.meta['Sentiment']))
        print(utt.speaker.id, ":", utt.text) 
    print()


def speaker_sentiment_matrix_given_convo(convo, corpus):
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
        # there are 21 politeness fields, 22 if counting 'Sentiment'
        # initialize vector with 22 zeros
        sentiment_vector = np.zeros((22,), dtype=int)
        for utt in corpus.iter_utterances(lambda u: u.speaker.id == s_id):
            sentiment = utt.meta['Sentiment']
            politeness_counts = utt.meta['politeness_strategies']

            if sentiment == "Negative":
                sentiment_vector[0]-=1
            else:
                if sentiment == "Positive":
                    sentiment_vector[0]+=1

                # politeness is not counted if the sentiment was negative
                i = 1
                for k in politeness_counts:
                    sentiment_vector[i] += politeness_counts[k]
                    i+=1

        sentiment_matrix.append(sentiment_vector)

    return np.array(sentiment_matrix)