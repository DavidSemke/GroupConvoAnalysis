from extraction import *
from idea_flow import idea_flows
from src.constants import gap_corpus, gap_convos
import matplotlib.pyplot as plt
from convokit import Corpus
from src.recurrence.data_pts.turn_taking import turn_taking_data_pts

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


def plot_turn_taking_epoch_determinism(convo):
    trials = turn_taking_rqa(convo, 'sliding')

    plt.title(f'Epoch Determinism - {convo.id}')
    plt.ylabel('Determinism')
    plt.xlabel('Epoch')

    for t in trials:
        dets = [epoch.determinism for epoch in t['results']]
        epochs = range(len(dets))
        label = f's={t["size"]},o={t["overlap"]},e={t["embed"]}'

        plt.plot(epochs, dets, label=label)

    plt.legend()
    plt.show()
    
    # diff, _ = turn_taking_frame_det_diff(convo)
    # print(diff)

    # diff, _ = turn_taking_sliding_det_diff(convo)
    # print(diff)


# get conversations with enough turn taking data
def turn_taking_convos(min_data_count=140):
    corpus = Corpus('corpora/gap-corpus')
    convo_ids = corpus.get_conversation_ids()
    convos = []

    for id in convo_ids:
        convo = corpus.get_conversation(id)
        data_pts, _ = turn_taking_data_pts(convo)

        if len(data_pts) < min_data_count: continue

        convos.append(convo)

    return convos


if __name__ == '__main__':
    # main()

    convos = turn_taking_convos()[:6]

    for convo in convos:
        plot_turn_taking_epoch_determinism(convo)
    