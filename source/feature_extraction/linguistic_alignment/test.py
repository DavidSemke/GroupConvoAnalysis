from extraction import avg_idea_participation_percentage, median_idea_discussion_time
from idea_flow import idea_flows
from source.constants import gap_corpus


convo = gap_corpus.get_conversation('1.Pink.1')
idea_flows_dict = idea_flows(convo, gap_corpus)

print()
for key in idea_flows_dict:
    for flow in idea_flows_dict[key]:
        print(flow['tok'])
        for id in flow['utt_ids']:
            print(f"\t{id}")

print()
print('Median idea discussion time:', median_idea_discussion_time(idea_flows_dict), 'secs')
print('Average idea participation percentage:', avg_idea_participation_percentage(convo, idea_flows_dict), '%')
print()


