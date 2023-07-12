from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.computation import RQAComputation, RPComputation
from pyrqa.image_generator import ImageGenerator

def rqa(data_pts, delay=1, embed=1, radius=0.1, rplot_path=None):
    time_series = TimeSeries(
        data_pts, embedding_dimension=embed, time_delay=delay
    )
    settings = Settings(
        time_series, analysis_type=Classic, 
        neighbourhood=FixedRadius(radius)
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