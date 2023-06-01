from convokit import Corpus

""" GAP Corpora Constants """

gap_corpus = Corpus('corpora/gap-corpus', utterance_end_index=890)

convos = [gap_corpus.get_conversation('1.Pink.1'),
          gap_corpus.get_conversation('12.Blue.1')]


""" Lexical Constants """

first_pronouns_sing = ['i','me','my','mine','myself']

articles = ['the','a','an','some']

third_pronouns = [
    'he','him','his','himself',
    'she','her','hers','herself',
    'it','its','itself',
    'they','them','their','theirs','themself','themselves'
]

negation_words = [
    'not','nt','no','never','neither','barely','hardly',
    'scarcely','seldom','rarely','nothing','none',
    'nobody','nowhere'
]


