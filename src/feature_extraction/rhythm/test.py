from extraction import *
import src.constants as const


def main():
    print()

    for convo in const.gap_convos:

        # print()
        # print(f'{convo.id.upper()} - SPEAKER METER VARIANCES')
        # print()

        # print('Order of meters:', const.meters)
        # print()

        # vars, wcv = meter_affinity_variances(convo)

        # print('\tVARS:', vars)
        # print('\tWCV:', wcv)

        print(f'{convo.id}- Meter Affinities')

        meter_affinities = [
            list(speaker_meter_affinity(speaker, convo)[0].values())
            for speaker in convo.iter_speakers()
        ]

        for affinity in meter_affinities:
            print(f'\t{affinity}')
        
        print()

        
if __name__ == '__main__':
    main()