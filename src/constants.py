from convokit import Corpus

# Shared Corpus Constants

speaker_colors = ['Pink', 'Blue', 'Green', 'Orange', 'Yellow']


# GAP Corpus Constants

gap_corpus = Corpus('corpora/gap-corpus', utterance_end_index=890)

gap_convos = [gap_corpus.get_conversation('1.Pink.1'),
          gap_corpus.get_conversation('12.Blue.1')]


# UGI Corpus Constants

ugi_corpus = Corpus('corpora/ugi-corpus/convokit_v2', utterance_end_index=100)

ugi_convos = [ugi_corpus.get_conversation('1.Blue.1')]


# Prosodic Constants

meters = [
    # 'iambic_pentameter',
    # 'iambic_pentameter2',
    # 'iambic_pentameter3',
    'kiparskyhanson_hopkins',
    # 'kiparskyhanson_shakespeare',
    # 'litlab',
    'meter_arto',
    'meter_ryan'
    # 'meter_ryan_categorical',
    # 'prose_rhythm_iambic',
    # 'prose_rhythm_iambic_inviolable',
    # 'prose_rhythm_iambic_violable',
    # 'strength_and_resolution',
    # 'strength_only'
]

# Lexical Constants

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


