import numpy as np
from meter import speaker_meter_affinity
from src.feature_extraction.rhythm.meter import *
from src.utils.stats import within_cluster_variance
import src.constants as const


def main():
    for convo in const.gap_convos:

        print()
        print(f'{convo.id.upper()} - SPEAKER METER VARIANCES')
        print()

        print('Order of meters:', const.meters)
        print()

        affinity_matrix = []
        
        for speaker in convo.iter_speakers():
            affinity, _ = speaker_meter_affinity(speaker, convo)

            affinity_matrix.append(
                list(affinity.values())
            )
        
        affinity_matrix = np.array(affinity_matrix)
        vars = []

        for i in range(len(affinity_matrix[0])):
            vars.append(round(np.var(affinity_matrix[:, i]), 2))

        print('\tVARS:', vars)
        print('\tWCV:', within_cluster_variance(affinity_matrix))


if __name__ == '__main__':
    main()