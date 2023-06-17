from convokit import TextParser
from nltk.stem import WordNetLemmatizer
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import TaxicabMetric
from pyrqa.computation import RQAComputation, RPComputation
from pyrqa.image_generator import ImageGenerator
from src.utils.token import lemmatize_word, is_word, is_content_word


def turn_taking_rqa(data_pts, embed, rplot_path):
    time_series = TimeSeries(
        data_pts, embedding_dimension=embed, time_delay=1
    )
    settings = Settings(
        time_series, analysis_type=Classic, 
        neighbourhood=FixedRadius(0.1)
    )

    computation = RPComputation.create(settings)
    result = computation.run()
    ImageGenerator.save_recurrence_plot(
        result.recurrence_matrix_reverse, rplot_path
    )

    computation = RQAComputation.create(settings)
    result = computation.run()
    result.min_diagonal_line_length = 2
    result.min_vertical_line_length = 2
    result.min_white_vertical_line_length = 2

    return result


def turn_taking_data_pts(convo):
    data_pts = []
    index_to_speaker = []
    speaker_to_index = {}
    new_idx = 0
    last_utt_speaker_id = None

    for utt in convo.iter_utterances():
        utt_speaker_id = utt.speaker.id

        if utt_speaker_id != last_utt_speaker_id:
            idx = speaker_to_index.get(utt_speaker_id)

            if idx is not None:
                data_pts.append(idx)
            
            else:
                speaker_to_index[utt_speaker_id] = new_idx
                data_pts.append(new_idx)
                index_to_speaker.append(utt_speaker_id)
                new_idx += 1
            
            last_utt_speaker_id = utt_speaker_id
    
    return data_pts, index_to_speaker


def idea_rqa(data_pts, rplot_path):
    time_series = TimeSeries(
        data_pts, embedding_dimension=1, time_delay=1
    )
    settings = Settings(
        time_series, analysis_type=Classic, 
        neighbourhood=FixedRadius(0.1)
    )

    computation = RPComputation.create(settings)
    result = computation.run()
    ImageGenerator.save_recurrence_plot(
        result.recurrence_matrix_reverse, rplot_path
    )

    computation = RQAComputation.create(settings)
    result = computation.run()
    result.min_diagonal_line_length = 2
    result.min_vertical_line_length = 2
    result.min_white_vertical_line_length = 2

    return result


# returns (data_pts, vocab_words)
# index of a word is the position of the word in vocab_words
def idea_data_pts(convo, corpus):
    corpus = corpus.filter_utterances_by(
        lambda u: u.conversation_id == convo.id
    )

    parser = TextParser()
    corpus = parser.transform(corpus)

    lemmatizer = WordNetLemmatizer()

    banned_verb_ideas = ["be", "have", "do"]
    data_pts = []
    vocab_words = []
    word_index_dict = {}
    index = 0
    
    for utt in corpus.iter_utterances():
        first_word_not_found = True

        for tok_dict in [
            tok_dict for parsed_dict in utt.meta['parsed'] 
            for tok_dict in parsed_dict['toks']
        ]:
            
            if not is_word(tok_dict): continue
            
            # exclude adverbs
            is_adverb = tok_dict['tag'][0:2] == 'RB'
            
            if is_adverb:
                first_word_not_found = False
                is_idea = False

            elif first_word_not_found:
                # skip non-idea words mistaken for proper nouns
                # (due to capitalization)
                first_word_not_found = False
                is_idea = is_content_word(tok_dict, parser, True)

            else:
                is_idea = is_content_word(tok_dict, parser, False)
            
            if not is_idea: continue
            
            tok = lemmatize_word(tok_dict, lemmatizer)

            if tok in banned_verb_ideas: continue

            existing_index = word_index_dict.get(tok)

            if existing_index is not None:
                data_pts.append(existing_index)
            else:
                word_index_dict[tok] = index
                data_pts.append(index)
                vocab_words.append(tok)
                index += 1

    return data_pts, vocab_words