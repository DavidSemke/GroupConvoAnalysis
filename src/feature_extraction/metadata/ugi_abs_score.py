import pandas as pd
from src.constants import ugi_rankings_path 

ugi_expert_ranking = [15,4,6,8,13,11,12,1,3,9,14,2,10,7,5]

def ugi_rankings_df():
    # UGI dataset stores ranking info in an excel sheet
    df = pd.read_excel(ugi_rankings_path, skiprows=1).drop(
        ['Not Ranked', 'Not Ranked.1', 'Q3 Desc'], axis=1
    )

    return df    


def ugi_ags_all():

    df = ugi_rankings_df()

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
    ags_dict = {}

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
                mode = min(modes, key=lambda x:abs(x-ugi_expert_ranking[j]))
                g_ranking.append(mode)
        
        ags = sum(
            [abs(g_ranking[j]-ugi_expert_ranking[j]) for j in range(len(g_ranking))]
        )
        
        ags_dict[group_id] = ags
        
        group_id = next_group_id
        first_group_index = last_group_index+1
            
    return ags_dict


def ugi_ais_all():
    
    df = ugi_rankings_df()
    ais_dict = {}

    for i in range(df.shape[0]):
        ais = 0
        
        for j in range(len(ugi_expert_ranking)):
            i_rank = df.iloc[i, 3+j]
            
            # add abs diff between expert rank and individ rank
            if not pd.isna(i_rank):
                ais += abs(i_rank - ugi_expert_ranking[j])
        
        ais_dict[f'{df.iloc[i, 0]}-{df.iloc[i, 1]}'] = ais
    
    return ais_dict