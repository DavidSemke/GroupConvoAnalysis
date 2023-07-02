from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.computation import RQAComputation, RPComputation
from pyrqa.image_generator import ImageGenerator
from pyrqa.analysis_type import Cross

def stress_crqa(data_pts_x, data_pts_y, embed, rplot_path):
    time_series_x = TimeSeries(
        data_pts_x, embedding_dimension=embed, time_delay=1
    )
    time_series_y = TimeSeries(
        data_pts_y, embedding_dimension=embed, time_delay=1
    )
    time_series = (time_series_x, time_series_y)
    
    settings = Settings(
        time_series, analysis_type=Cross,
        neighbourhood=FixedRadius(0.1), theiler_corrector=0
    )
    computation = RPComputation.create(settings)
    rp_result = computation.run()
    ImageGenerator.save_recurrence_plot(
        rp_result.recurrence_matrix_reverse, rplot_path
    )
    computation = RQAComputation.create(settings)
    rqa_result = computation.run()

    return rqa_result, rp_result


def stress_rqa(data_pts, embed, rplot_path):
    time_series = TimeSeries(
        data_pts, embedding_dimension=embed, time_delay=1
    )
    settings = Settings(
        time_series, analysis_type=Classic, 
        neighbourhood=FixedRadius(0.1)
    )
    computation = RPComputation.create(settings)
    rp_result = computation.run()
    ImageGenerator.save_recurrence_plot(
        rp_result.recurrence_matrix_reverse, rplot_path
    )
    computation = RQAComputation.create(settings)
    rqa_result = computation.run()

    return rqa_result, rp_result