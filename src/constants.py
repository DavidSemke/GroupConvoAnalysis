from convokit import Corpus

# GAP corpora constants
gap_corpus = Corpus('corpora/gap-corpus', utterance_end_index=890)
convos = [gap_corpus.get_conversation('1.Pink.1'),
          gap_corpus.get_conversation('12.Blue.1')]

