from feature_rqa import *
from summarize_rqa import epoch_rqa_summary
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

        epoch_rqa_summary(out, epoch_type, metrics_func )


if __name__ == '__main__':
    
    for convo in gap_convos:
        feature_rqa_test(
            'TURN-TAKING RQA', True, 
            lambda e: {
                'determinism': e.determinism,
                'avg diagonal line': e.average_diagonal_line,
                'longest diagonal line': e.longest_diagonal_line
            },
            turn_taking_rqa, convo=convo
        )