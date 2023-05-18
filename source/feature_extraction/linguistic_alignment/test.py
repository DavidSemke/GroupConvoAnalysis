from extraction import *
from idea_flow import idea_flows
from source.constants import gap_corpus, convos

for convo in convos:
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
    print()



