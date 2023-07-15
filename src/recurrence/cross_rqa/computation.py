from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.computation import RQAComputation, RPComputation
from pyrqa.image_generator import ImageGenerator
from pyrqa.analysis_type import Cross
from itertools import combinations
from src.recurrence.data_pts.stresses import stress_data_pts
from src.feature_extraction.rhythm.meter import *

def dyad_stress_crqa(convo, embeds=(2,4,6)):
    speakers = list(convo.iter_speakers())
    speaker_pairs = list(combinations(speakers, 2))
    results = []

    for pair in speaker_pairs:
        data_pts = {}
        s1, s2 = pair

        for speaker in pair:
            filter = lambda u: u.speaker.id == speaker.id
            meter_affinity = dyad_meter_affinity(
                s1, s2, convo, filter
            )
            stresses = best_utterance_stresses(meter_affinity)
            data = stress_data_pts(stresses)
            data_pts[speaker.id] = data
    
        rplot_folder = r'recurrence_plots\cross_rqa\dyad_stress'

        delay = 1

        for embed in embeds:
            
            rplot_path = (
                rf'{rplot_folder}\rplot_{convo.id}_{s1.id}-{s2.id}'
                + f'_delay{delay}_embed{embed}.png'
            )

            out = crqa(
                data_pts[s1.id], data_pts[s2.id], delay, embed, 
                rplot_path=rplot_path
            )
            results.append(
                {'delay': delay, 'embed': embed, 'results': out}
            )
            
    return results


def crqa(
        data_pts_x, data_pts_y, delay=1, embed=1, radius=0.1, 
        rplot_path=None
):
    time_series_x = TimeSeries(
        data_pts_x, embedding_dimension=embed, time_delay=delay
    )
    time_series_y = TimeSeries(
        data_pts_y, embedding_dimension=embed, time_delay=delay
    )
    time_series = (time_series_x, time_series_y)
    
    settings = Settings(
        time_series, analysis_type=Cross,
        neighbourhood=FixedRadius(radius), theiler_corrector=0
    )

    results = []

    computation = RQAComputation.create(settings)
    rqa_result = computation.run()
    results.append(rqa_result)

    if rplot_path:
        computation = RPComputation.create(settings)
        rp_result = computation.run()
        ImageGenerator.save_recurrence_plot(
            rp_result.recurrence_matrix_reverse, rplot_path
        )
        results.append(rp_result)

    return results