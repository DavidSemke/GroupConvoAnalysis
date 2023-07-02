from extraction import *
from idea_flow import idea_flows
from src.constants import gap_corpus, gap_convos
from recurrence import *
from src.utils.rqa_data_pts import *

def main():

    for convo in gap_convos:
        print()
        print(convo.id.upper())
        print()
        
        idea_flows_dict = idea_flows(convo, gap_corpus)
        
        print("Idea flows:")
        for key in idea_flows_dict:
            for flow in idea_flows_dict[key]:
                print()
                print(flow['tok']) 
                for id in flow['utt_ids'][0:3]:
                    print(f"\t{id}")
                if len(flow['utt_ids']) > 3:
                    remaining = len(flow['utt_ids']) - 3
                    print("\t...and", remaining, "more")

        print()
        print('Median idea discussion time:', median_idea_discussion_time(idea_flows_dict), 'secs')
        print('Average idea participation percentage:', avg_idea_participation_percentage(convo, idea_flows_dict), '%')
        print('Idea distribution score:', idea_distribution_score(convo, idea_flows_dict))
        print('Speech rate convergence (frame = 10%):', speech_rate_convergence(convo, 10))
        print('Speech rate convergence (frame = 5%):', speech_rate_convergence(convo, 5))
        print('Coordination variances (to, from):', coordination_variances(convo, gap_corpus))
        print()


def idea_rqa_test():
    # for convo in gap_convos:
    convo = gap_convos[0]

    print()
    print(f'{convo.id.upper()} - IDEA RQA')
    print()

    data_pts, _ = idea_data_pts(convo, gap_corpus)
    rqa_res, rp_res = idea_rqa(
        data_pts, rf'recurrence_plots\rqa\ideas\rplot_{convo.id}.png'
    )

    print(rqa_res)
    print()


def turn_taking_rqa_test():
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - TURN-TAKING RQA')
        print()

        rplot_folder = r'recurrence_plots\rqa\turn-taking'

        data_pts, _ = turn_taking_data_pts(convo)

        for embed in (1, 2, 3):
            print(f'Embedding Dimn = {embed}:')
            print()

            rplot_path = rf'{rplot_folder}\rplot_{convo.id}_embed{embed}.png'
            
            rqa_res, rp_res = turn_taking_rqa(
                data_pts, embed, rplot_path
            )
            
            print(rqa_res.recurrence_points)
            print()


def letter_stream_rqa_test():
    for convo in gap_convos:
        print()
        print(f'{convo.id.upper()} - LETTER STREAM RQA')
        print()

        rplot_folder = r'recurrence_plots\rqa\letter_stream'

        data_pts = letter_data_pts(convo)

        for embed in (3, 4, 5):
            print(f'Embedding Dimn = {embed}:')
            print()

            rplot_path = rf'{rplot_folder}\rplot_{convo.id}_embed{embed}.png'
            
            rqa_res, rp_res = turn_taking_rqa(
                data_pts, embed, rplot_path
            )
            
            print(rqa_res)
            print()


if __name__ == "__main__":
    main()
    # idea_rqa_test()
    # turn_taking_rqa_test()
    # letter_stream_rqa_test()
    