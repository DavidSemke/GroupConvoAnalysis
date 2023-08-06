from extraction import *
from src.constants import gap_convos, gap_corpus


def main():

    for convo in gap_convos:
        print()
        print(convo.id.upper())
        print()

        print(
            "Speech overlap percentage:", 
            speech_overlap_percentage(convo), "%"
        )
        print(
            'Speech distribution score:',
            speech_distribution_score(convo, gap_corpus)
        )

        lam, trial = speech_overlap_frame_lam(convo)
        frame = trial["frame"]
        print(f'Speech overlap frame LAM (frame = {frame}):', lam)

        lam, _ = speech_overlap_sliding_lam(convo)
        print(f'Speech overlap mean sliding LAM:', lam)

        vert_stats, _ = speech_overlap_vertical_stats(convo)
        print(
            'Speech overlap trapping time:',
            vert_stats['trapping_time']
        )
        print(
            'Speech overlap longest vertical line:',
            vert_stats['longest_vertical_line']
        )
        
        print()


if __name__ == '__main__':
    main()