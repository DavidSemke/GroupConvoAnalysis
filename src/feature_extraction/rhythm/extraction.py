from src.feature_extraction.rhythm.meter import speaker_meter_affinity
from src.utils.stats import within_cluster_variance

def contrast_in_meter_affinity(convo):
    affinity_matrix = []

    for speaker in convo.iter_speakers():
        affinity, _ = speaker_meter_affinity(speaker, convo)

        affinity_matrix.append(
            list(affinity.values())
        )
            
    return within_cluster_variance(affinity_matrix)