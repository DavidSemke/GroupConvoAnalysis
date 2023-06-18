from src.ugi_dataset.line_processing import *
import src.constants as const

def convo_metadata(df, expert_ranking, transcripts_path, patts):
    convo_meta = {}

    grp_stats_dict = group_stats(df, expert_ranking)

    for group_id in grp_stats_dict:
        txt_path = (
            transcripts_path 
            + rf'\TeamID_{group_id}_transcript.txt'
        )

        with open(txt_path, 'r') as file:
            lines = file.readlines()
        
        lines = [l.strip() for l in lines if l.strip()]

        last_line = lines[-1]
        secs, _, _ = parse_line(last_line, patts)
        mins = round(float(secs) / 60, 2)

        line_index = 0

        while line_index < len(lines):

            elements = process_line_elements(lines, line_index, patts)

            if not elements:
                line_index += 1
                continue
        
            _, local_id, _ = elements

            color = const.speaker_colors[local_id-1]
            speaker_id = f'{group_id}.{color}'
            convo_id = f'{speaker_id}.1'

            break
        
        convo_meta[convo_id] = {
            'Group Number': group_id,
            'Meeting Size': grp_stats_dict[group_id]['Meeting Size'],
            'Meeting Length in Minutes': mins,
            'AGS': grp_stats_dict[group_id]['AGS']
        }
    
    return convo_meta


# returns a dict of dicts
# each inner dict has format {ags, meeting_size}
def group_stats(df, expert_ranking):
    # get cols reserved for group rankings
    only_groups_df = df.iloc[:, 18:33]

    # remove the '.1' appended to the end of each col name
    for col in only_groups_df:
        only_groups_df.rename(columns={col: col[:-2]}, inplace=True)

    # get cols reserved for individual rankings
    only_individs_df = df.iloc[:, 3:18]

    # fill in missing values of only_groups_df with values from 
    # only_individs_df
    only_groups_df[only_groups_df.isna()] = only_individs_df

    # group ids are in df but not in only_groups_df
    # therefore, df is used to get team id
    group_id = df.iloc[0, 0]
    first_group_index = 0
    last_group_index = 0
    more_groups = True
    group_stats_dict = {}

    while more_groups: 
        
        # determine the indexes of the first and last members
        # group_id is the same for all members
        for i in range(first_group_index, df.shape[0]):
            next_group_id = df.iloc[i, 0]
            
            if next_group_id != group_id:
                last_group_index = i-1
                break
        
        # ensure that last_group_index is updated for the 
        # last group
        if i == df.shape[0]-1:
            last_group_index = i
            more_groups = False
        
        # get rows corresponding to members of this group
        group_df = only_groups_df.iloc[first_group_index:last_group_index+1]
        g_ranking = []
        
        # rankings after group discussion are not necessarily
        # the same for each member
        # this loop computes the group rank for each item as 
        # the mode of individ ranks in the group
        # if more than one mode, mode that contributes the
        # least to the ags is chosen
        for j in range(group_df.shape[1]):
            item_col = group_df.iloc[:, j]            
            modes = list(item_col.mode())
            
            if len(modes) == 1:
                g_ranking.append(modes[0])
            
            else:
                mode = min(
                    modes, key=lambda x:abs(x-expert_ranking[j])
                )
                g_ranking.append(mode)
        
        ags = sum(
            [abs(g_ranking[j]-expert_ranking[j]) 
             for j in range(len(g_ranking))]
        )
        
        group_stats_dict[group_id] = {
            'AGS': ags, 
            'Meeting Size': last_group_index-first_group_index+1
        }
        
        group_id = next_group_id
        first_group_index = last_group_index+1
            
    return group_stats_dict