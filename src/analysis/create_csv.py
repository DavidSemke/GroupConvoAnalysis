import csv
from src.feature_extraction.linguistic_alignment.idea_flow import idea_flows
from src.feature_extraction.dominance.extraction import *
from src.feature_extraction.linguistic_alignment.extraction import *
from src.feature_extraction.politeness.extraction import *
from src.feature_extraction.rhythm.extraction import *
from src.feature_extraction.word_psych_properties.extraction import *
from src.constants import gap_corpus


def main():
    corpus = gap_corpus
    corpus_id = 'gap'
    extraction_func_groups = {
        'dom': [dom_features],
        'align': [align_features],
        'polite': [polite_features]
        # 'rhythm': [rhythm_features],
        # 'psych': [psych_features]
    }

    csv_dump(corpus, corpus_id, extraction_func_groups)
    # chained_csv_dumps(corpus, corpus_id, extraction_func_groups)
   

def csv_dump(corpus, corpus_id, extraction_func_groups):
    
    for k, funcs in extraction_func_groups.items():
        obsn_matrix, fields = observation_matrix(corpus, funcs)

        with open(f'csv/{corpus_id}-{k}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            writer.writerows(obsn_matrix)
        
        print()
        print(k, 'features complete')
        print()


# Use this when it is unlikely that all conversations can be processed 
# without error (e.g. when using selenium package)
# Writes after extracting features from a convo
# Parameter start_convo_index is where to start given a list of convos
# from a corpus
def chained_csv_dumps(
    corpus, corpus_id, extraction_func_groups, start_convo_index=0
):
    
    for k, funcs in extraction_func_groups.items():
        convos = list(corpus.iter_conversations())[start_convo_index:]
        is_first = start_convo_index == 0

        for i, convo in enumerate(convos):
            obsn, fields = observation(convo, corpus, funcs)

            with open(f'csv/{corpus_id}-{k}.csv', 'a', newline='') as file:
                writer = csv.writer(file)

                if is_first:
                    writer.writerow(fields)
                    is_first = False
                
                writer.writerow(obsn)
            
            print('Convo', i, 'complete')
        
        print()
        print(k, 'features complete')
        print()


def observation_matrix(corpus, extraction_funcs):
    obsn_matrix = []

    for convo in corpus.iter_conversations():
        obsn, feat_names = observation(
            convo, corpus, extraction_funcs
        )
        
        obsn_matrix.append(obsn)
        print(f'Convo {convo.id} complete')
    
    return obsn_matrix, feat_names


def observation(convo, corpus, extraction_funcs):
    obsn = []
    feat_names = []
        
    for func in extraction_funcs:
        feats = func(convo, corpus)
        obsn.extend(list(feats.values()))
        feat_names += list(feats.keys())
    
    return obsn, feat_names


def dom_features(convo, corpus):
    vert_stats = speech_overlap_vertical_stats(convo)[0]

    feats = {
        'sd_score': speech_distribution_score(convo, corpus),
        'so-flam': speech_overlap_frame_lam(convo)[0],
        'so-slam': speech_overlap_sliding_lam(convo)[0],
        'so-avg_vert': vert_stats['trapping_time'],
        'so-longest_vert': vert_stats['longest_vertical_line'],
    }

    return feats
        

def align_features(convo, corpus):
    flows_dict = idea_flows(convo, corpus)
    coord_to, coord_from = coordination_variances(convo, corpus)
    diag_stats = turn_taking_diagonal_stats(convo)[0]
    
    feats = {
        'median_idt': median_idea_discussion_time_part(flows_dict),
        'avg_ip%' : avg_idea_participation_percentage_part(
            convo, flows_dict
        ),
        'id_score': idea_distribution_score_part(convo, flows_dict),
        'src-10': speech_rate_convergence(convo, 10),
        'coord_var-to': coord_to,
        'coord_var-from': coord_from,
        'tt-fdet': turn_taking_frame_det(convo)[0],
        'tt-sdet': turn_taking_sliding_det(convo)[0],
        'tt-avg_diag': diag_stats['average_diagonal_line'],
        'tt-longest_diag': diag_stats['longest_diagonal_line'],
    }

    return feats


def polite_features(convo, corpus):
    word_pos_ratio, word_neg_ratio = sentiment_ratios(
        convo, corpus, word_level=True
    )
    sent_pos_ratio, sent_neg_ratio = sentiment_ratios(convo, corpus)
    
    feats = {
        'cf-s': contrast_in_formality(convo, corpus),
        'cf-w': contrast_in_formality(convo, corpus, True),
        'pos_sr-s': sent_pos_ratio,
        'pos_sr-w': word_pos_ratio,
        'neg_sr-s': sent_neg_ratio,
        'neg_sr-w': word_neg_ratio
    }

    return feats


def rhythm_features(convo, _):
    sp_vert_stats = speech_pause_vertical_stats(convo)[0]
    cs_diag_stats = convo_stress_diagonal_stats(convo)[0]

    feats = {
        'cma': contrast_in_meter_affinity(convo),
        'sp-flam': speech_pause_frame_lam(convo)[0],
        'sp-slam': speech_pause_sliding_lam(convo)[0],
        'sp-avg_vert': sp_vert_stats['trapping_time'],
        'sp-longest_vert': sp_vert_stats['longest_vertical_line'],
        'cs-flam': convo_stress_frame_det(convo)[0],
        'cs-slam': convo_stress_sliding_det(convo)[0],
        'cs-avg_vert': cs_diag_stats['average_diagonal_line'],
        'cs-longest_vert': cs_diag_stats['longest_diagonal_line']
    }

    return feats


def psych_features(convo, corpus):
    vars = psych_property_score_variances(convo, corpus)
    
    feats = {
        'aoa': vars[0],
        'cnc': vars[1],
        'fam': vars[2],
        'img': vars[3],
        'cp': constrast_in_personality(convo, corpus)
    }

    return feats


if __name__ == "__main__":
    main()