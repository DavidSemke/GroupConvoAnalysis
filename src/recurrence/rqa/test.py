from feature_rqa import *
from summarize_rqa import (
    plot_epoch_rqa_metric,
    epoch_rqa_summary,
    convo_group_rqa_det_summary,
    convo_group_rqa_lam_summary
)
from src.utils.filter_convos import extreme_convo_groups
from src.constants import gap_corpus, gap_convos


# Parameter feature_rqa_func is a non-helper function from 
# module feature_rqa
# See module summarize_rqa for an explanation of parameter metrics_func
def feature_rqa_test(
        title, verbose, metrics_func, feature_rqa_func, **kwargs
):
    print()
    print(f'{kwargs["convo"].id.upper()} - {title.upper()}')
    print()

    out = feature_rqa_func(**kwargs)

    if verbose:
        epoch_type = None

        if 'epoch_type' in kwargs:
            epoch_type = kwargs['epoch_type']

        epoch_rqa_summary(out, epoch_type, metrics_func)


def plot_turn_taking_determinism(convo):
    plot_epoch_rqa_metric(
        'Turn-Taking Determinism',
        'Determinism',
        'Epochs',
        turn_taking_rqa,
        convo, 'sliding',
        metric_func=lambda e: {
            'determinism': e.determinism
        }
    )


def epochless_turn_taking_determinism(convo):
    feature_rqa_test(
        'TURN-TAKING RQA', True, 
        lambda e: {
            'determinism': e.determinism,
            'avg diagonal line': e.average_diagonal_line,
            'longest diagonal line': e.longest_diagonal_line
        },
        turn_taking_rqa, convo=convo
    )


def extreme_group_turn_taking_determinism(corpus):
    filter = lambda convo: len(convo.get_speaker_ids()) > 2
    extremes = extreme_convo_groups(corpus, 0.5, filter)
    groups = {'low': extremes[0], 'high': extremes[1]}
    
    convo_group_rqa_det_summary(groups, turn_taking_rqa)


def extreme_group_speech_overlap_laminarity(corpus):
    extremes = extreme_convo_groups(corpus)
    groups = {'low': extremes[0], 'high': extremes[1]}

    convo_group_rqa_lam_summary(
        groups, simult_binary_speech_sampling_rqa
    )


def extreme_group_speech_pause_laminarity(corpus):
    extremes = extreme_convo_groups(corpus)
    groups = {'low': extremes[0], 'high': extremes[1]}

    convo_group_rqa_lam_summary(
        groups, binary_speech_sampling_rqa
    )


def extreme_group_convo_stress_determinism(corpus):
    extremes = extreme_convo_groups(corpus)
    groups = {'low': extremes[0], 'high': extremes[1]}

    convo_group_rqa_det_summary(
        groups, convo_stress_rqa
    )


if __name__ == '__main__':
    corpus = gap_corpus
    # convos = [
    #     corpus.get_conversation(convo_id) 
    #     for convo_id in corpus.get_conversation_ids()
    # ]

    # extreme_group_speech_pause_laminarity(corpus)
    extreme_group_convo_stress_determinism(corpus)