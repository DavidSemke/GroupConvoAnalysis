from src.recurrence.rqa.computation import *
from src.constants import gap_corpus, gap_convos

# delay and embed are both 1, so they are omitted from file names
def idea_rqa_test(verbose=False):
    
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - IDEA RQA')
        print()

        out = idea_rqa(gap_corpus, convo)
        
        if verbose:
            for res in out:
                print(res[0])


def letter_stream_rqa_test(verbose=False):
    
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - LETTER STREAM RQA')
        print()

        out = letter_stream_rqa(convo)
        
        if verbose:
            for res in out:
                print(res[0])


def turn_taking_rqa_test(verbose=False):
    
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - TURN-TAKING RQA')
        print()

        out = turn_taking_rqa(convo, verbose)
        
        if verbose:
            for res in out:
                print(res[0])


def complete_speech_sampling_rqa_test(verbose=False):
    
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - COMPLETE SPEECH SAMPLING RQA')
        print()

        out = complete_speech_sampling_rqa(convo, verbose)
        
        if verbose:
            for res in out:
                print(res[0])


def binary_speech_sampling_rqa_test(verbose=False):
    
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - BINARY SPEECH SAMPLING RQA')
        print()

        out = binary_speech_sampling_rqa(convo, verbose)
        
        if verbose:
            for res in out:
                print(res[0])


def simult_binary_speech_sampling_rqa_test(verbose=False):
    
    for convo in gap_convos:
        print()
        print(
            f'{convo.id.upper()} - SIMULT BINARY SPEECH SAMPLING RQA'
        )
        print()

        out = simult_binary_speech_sampling_rqa(convo, verbose)
        
        if verbose:
            for res in out:
                print(res[0])
    

def convo_stress_rqa_test(verbose=False):

    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - CONVO STRESS RQA')
        print()

        out = convo_stress_rqa(convo, plot=verbose)
        
        if verbose:
            for res in out:
                print(res[0])


def dyad_stress_rqa_test(verbose=False):

    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - DYAD STRESS RQA')
        print()

        out = dyad_stress_rqa(convo, plot=verbose)

        if verbose:
            for res in out:
                print(res[0])
            

if __name__ == '__main__':
    # idea_rqa_test()
    # letter_stream_rqa_test()
    # turn_taking_rqa_test(True)
    # convo_stress_rqa_test(True)
    # dyad_stress_rqa_test(True)
    # complete_speech_sampling_rqa_test(True)
    # binary_speech_sampling_rqa_test(True)
    simult_binary_speech_sampling_rqa_test(True)