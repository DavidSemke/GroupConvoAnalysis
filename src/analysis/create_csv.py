import csv
from src.feature_extraction.dominance.extraction import *
from src.feature_extraction.linguistic_alignment.idea_flow import idea_flows
from src.feature_extraction.linguistic_alignment.extraction import *
from src.feature_extraction.politeness.extraction import *
from src.feature_extraction.rhythm.meter import (
    speaker_meter_affinity,
    best_utterance_stresses,
    convo_stresses
)
from src.recurrence.data_pts.stresses import stress_data_pts
from src.feature_extraction.rhythm.extraction import *
from src.feature_extraction.word_psych_properties.extraction import *
import src.constants as const


# Make sure to comment out features not compatible with the UGI corpus 
# (any feature that relies on 'Duration'/'End' metadata fields)
def main():
    corpus = const.ugi_corpus
    corpus_id = 'ugi'
    extraction_func_groups = {
        # 'dom': [dom_features],
        # 'align': [align_features]
        # 'polite': [polite_features],
        'rhythm': [rhythm_features]
        # 'psych': [psych_features],
        # 'meta': [metadata_features]
    }

    # csv_dump(corpus, corpus_id, extraction_func_groups)
    chained_csv_dumps(corpus, corpus_id, extraction_func_groups)
   

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

    for i, convo in enumerate(corpus.iter_conversations()):
        obsn, feat_names = observation(
            convo, corpus, extraction_funcs
        )
        
        obsn_matrix.append(obsn)
        print(f'Convo', i, 'complete')
    
    return obsn_matrix, feat_names


def observation(convo, corpus, extraction_funcs):
    obsn = []
    feat_names = []
        
    for func in extraction_funcs:
        feats = func(convo, corpus)
        obsn.extend(list(feats.values()))
        feat_names += list(feats.keys())
    
    return obsn, feat_names


def metadata_features(convo, _):
    feats = {
        'id': convo.id,
        'size': convo.meta['Meeting Size'],
        'total_mins': convo.meta['Meeting Length in Minutes'],
        'ags': convo.meta['AGS']
    }
    
    return feats


def dom_features(convo, corpus):
    vert_stats = speech_overlap_vertical_stats(convo)[0]

    feats = {
        'sd_score': speech_distribution_score(convo, corpus),
        'so-flam': speech_overlap_frame_lam(convo)[0],
        'so-slam': speech_overlap_sliding_lam(convo)[0],
        'so-avg_vert': vert_stats['trapping_time'],
        'so-longest_vert': vert_stats['longest_vertical_line']
    }

    return feats
        

def align_features(convo, corpus):
    flows_dict = idea_flows(convo, corpus)
    coord_to, coord_from = coordination_variances(convo, corpus)
    diag_stats = turn_taking_diagonal_stats(convo)[0]
    
    feats = {
        'avg_ip%' : avg_idea_participation_percentage_part(
            convo, flows_dict
        ),
        'id_score': idea_distribution_score_part(convo, flows_dict),
        # 'src-10': speech_rate_convergence(convo, 10),
        'coord_var-to': coord_to,
        'coord_var-from': coord_from,
        'tt-fdet': turn_taking_frame_det(convo)[0],
        'tt-sdet': turn_taking_sliding_det(convo)[0],
        'tt-avg_diag': diag_stats['average_diagonal_line'],
        'tt-longest_diag': diag_stats['longest_diagonal_line']
    }

    return feats


def polite_features(convo, corpus):
    word_pos_ratio, word_neg_ratio = sentiment_ratios(
        convo, corpus, word_level=True
    )
    sent_pos_ratio, sent_neg_ratio = sentiment_ratios(convo, corpus)
    
    feats = {
        'cf-s': sentiment_variance(convo, corpus),
        'cf-w': sentiment_variance(convo, corpus, True),
        'pos_sr-s': sent_pos_ratio,
        'pos_sr-w': word_pos_ratio,
        'neg_sr-s': sent_neg_ratio,
        'neg_sr-w': word_neg_ratio
    }

    return feats


def rhythm_features(convo, _):
    meter_affinities = {
        speaker.id: speaker_meter_affinity(speaker, convo)
        for speaker in convo.iter_speakers()
    }

    # get stress data points for stress rqa
    best_stresses = {}
    
    for sid, affinity in meter_affinities.items():
        utt_stresses = best_utterance_stresses(affinity)
        best_stresses[sid] = utt_stresses

    stresses = convo_stresses(convo, best_stresses)
    data_pts = stress_data_pts(stresses)

    # extract features
    cs_diag_stats = convo_stress_diagonal_stats(convo, data_pts)[0]
    # sp_vert_stats = speech_pause_vertical_stats(convo)[0]

    feats = {
        # 'sp-flam': speech_pause_frame_lam(convo)[0],
        # 'sp-slam': speech_pause_sliding_lam(convo)[0],
        # 'sp-avg_vert': sp_vert_stats['trapping_time'],
        # 'sp-longest_vert': sp_vert_stats['longest_vertical_line'],
        'cs-flam': convo_stress_frame_det(convo, data_pts)[0],
        'cs-slam': convo_stress_sliding_det(convo, data_pts)[0],
        'cs-avg_vert': cs_diag_stats['average_diagonal_line'],
        'cs-longest_vert': cs_diag_stats['longest_diagonal_line']
    }

    # get affinity matrix for meter_affinity_variances func
    affinity_matrix = [
        list(affinity[0].values())
        for affinity in meter_affinities.values()
    ]

    vars, wcv = meter_affinity_variances_part(affinity_matrix)

    for i, meter in enumerate(const.meters):
        key = meter + '_var'
        feats[key] = vars[i]
    
    feats['meter_wcv'] = wcv

    return feats


def psych_features(convo, corpus):
    pps_vars, pps_wcv = psych_property_score_variances(convo, corpus)
    pt_vars, pt_wcv = personality_trait_variances(convo, corpus)
    
    feats = {
        'aoa_var': pps_vars[0],
        'cnc_var': pps_vars[1],
        'fam_var': pps_vars[2],
        'img_var': pps_vars[3],
        'pps_wcv': pps_wcv,
        'total_w_var': pt_vars[0],
        'simple_w_var': pt_vars[1],
        '1p_pron_sing_var': pt_vars[2],
        '3p_pron_var': pt_vars[3],
        'art_var': pt_vars[4],
        'neg_w_var': pt_vars[5],
        'neg_var': pt_vars[6],
        'pt_wcv': pt_wcv
    }

    return feats


if __name__ == "__main__":
    main()