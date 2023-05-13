from convokit import Corpus, Utterance, Speaker
from idea_flow import idea_flows
from source.constants import gap_corpus

convo = gap_corpus.get_conversation('1.Pink.1')

# person = Speaker(id="0")

# utts = [
#     Utterance(id="0", conversation_id="0", text="Manchester United are looking to sign a larger forward for $90 million", speaker=person),
# ]


# corpus = Corpus(utterances=utts)
# convo = corpus.get_conversation('0')

flows = idea_flows(convo, gap_corpus)

print()
print(f"Total adjective idea flows: {len(flows['J'])}")
print(f"Total noun idea flows: {len(flows['N'])}")
print(f"Total verb idea flows: {len(flows['V'])}")

for key in flows:
    for flow in flows[key]:
        print(f"First {key} idea flow: {flow}")
        break

print()


