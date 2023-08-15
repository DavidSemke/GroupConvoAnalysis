from src.recurrence.cross_rqa.computation import dyad_stress_crqa
from src.constants import gap_convos


def dyad_stress_crqa_test(verbose=False):
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - DYAD STRESS CRQA')
        print()

        out = dyad_stress_crqa(convo)
        
        if verbose:
            for res in out:
                print(res['results'][0])


if __name__ == '__main__':
    dyad_stress_crqa_test()