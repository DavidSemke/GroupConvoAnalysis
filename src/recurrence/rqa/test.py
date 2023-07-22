from src.recurrence.rqa.feature_rqa import *
from src.constants import gap_corpus, gap_convos


def feature_rqa_test(title, verbose, feature_rqa_func, **kwargs):
    print()
    print(f'{kwargs["convo"].id.upper()} - {title.upper()}')
    print()

    out = feature_rqa_func(**kwargs)

    if verbose:
        epoch_type = None

        if 'epoch_type' in kwargs:
            epoch_type = kwargs['epoch_type']

        rqa_det_summary(out, epoch_type)


# print out deterministic details of RQA output
def rqa_det_summary(rqa_output, epoch_type):
    
    if not epoch_type:

        for trial in rqa_output:
            for key, val in trial.items():
                
                if key == 'results': continue
                
                print(key, '=', val)

            print()
            print(trial['results'][0])
            print()

    elif epoch_type in ('frame', 'sliding'):
        
        for i, trial in enumerate(rqa_output):
            print(f'TRIAL {i}')
            print()

            for key, val in trial.items():
                
                if key == 'results': continue
                
                print(key, '=', val)
            
            print()

            for i, epoch in enumerate(trial['results']):
                print(f'\tEPOCH {i}')
                print()
                print('\tdeterminism =', 
                        epoch.determinism
                )
                print('\taverage_diagonal_line =', 
                    epoch.average_diagonal_line
                )
                print('\tlongest_diagonal_line =', 
                    epoch.longest_diagonal_line
                )
                print()

    else:
        raise Exception(
            'Parameter epoch_type can only take on values "frame"' 
            + 'and "sliding"'
        )
        
        
if __name__ == '__main__':
    
    for convo in gap_convos:
        # feature_rqa_test(
        #     'IDEA RQA', True, idea_rqa, 
        #     corpus=gap_corpus, convo=convo
        # )
        # feature_rqa_test(
        #     'LETTER STREAM RQA', True, letter_stream_rqa, 
        #     convo=convo
        # )
        feature_rqa_test(
            'TURN-TAKING RQA', True, turn_taking_rqa, 
            convo=convo, epoch_type='sliding'
        )
        # feature_rqa_test(
        #     'COMPLETE SPEECH SAMPLING RQA', True, 
        #     complete_speech_sampling_rqa, 
        #     convo=convo
        # )
        # feature_rqa_test(
        #     'BINARY SPEECH SAMPLING RQA', True, 
        #     binary_speech_sampling_rqa, 
        #     convo=convo
        # )
        # feature_rqa_test(
        #     'SIMULT BINARY SPEECH SAMPLING RQA', True, 
        #     simult_binary_speech_sampling_rqa, 
        #     convo=convo
        # )
        # feature_rqa_test(
        #     'CONVO STRESS RQA', True, convo_stress_rqa, 
        #     convo=convo, epoch_type=None
        # )
        # feature_rqa_test(
        #     'DYAD STRESS RQA', True, dyad_stress_rqa, 
        #     convo=convo, epoch_type=None
        # )