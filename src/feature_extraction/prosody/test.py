import prosodic as pro
from convokit import TextParser
from src.constants import *


# convo = convos[0]
# for speaker in convo.iter_speakers():

#     corpus = corpus.filter_utterances_by(lambda u: u.     conversation_id == convo.id and u.speaker.id == speaker.id)
    
#     for utt in corpus.iter_utterances():
        
#         parsed = pro.Text(utt.text)

# a = "clandestine operations always end poorly"
# b = "anyone who knew the details would understand the decision"
# c = "Mr Schloopendorf hates skiing"

# t = pro.Text(a)
# sylls = t.syllables()

# t.parse()

# print(t.scansion())

# print(t.report())

# for parse in t.bestParses():
#     print()
#     print(parse)
#     print()

# print(gap_corpus.get_utterance('1.Pink.1'))

print(convos[0].meta['Meeting Length in Minutes'] * 60)