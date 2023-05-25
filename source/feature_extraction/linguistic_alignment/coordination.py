from convokit import Corpus, Coordination

corpus = Corpus('corpora/gap-corpus')
# utterance_end_index=100

coord = Coordination()
coord.fit(corpus)
corpus = coord.transform(corpus)

scores = coord.summarize(corpus, focus="targets").averages_by_speaker()
interlocutors = list(scores.keys())
score_max = 0
score_max_i = None
for i in interlocutors:
    if scores[i] > score_max:
        score_max = scores[i]
        score_max_i = i

# display interlocutor with max score (this person has the most influence on others in their group out of all groups)
print()
print(score_max_i.id, score_max)

convos = score_max_i.get_conversation_ids()
convo = corpus.get_conversation(convos[0])
all_utts = convo.get_chronological_utterance_list()

for utt in all_utts:
    print()
    print(utt.speaker.id + ": ", utt.text)

print()

