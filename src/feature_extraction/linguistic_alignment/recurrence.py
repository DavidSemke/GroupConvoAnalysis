from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.computation import RQAComputation, RPComputation
from pyrqa.image_generator import ImageGenerator


def turn_taking_rqa(data_pts, embed, rplot_path):
    time_series = TimeSeries(
        data_pts, embedding_dimension=embed, time_delay=1
    )
    settings = Settings(
        time_series, analysis_type=Classic, 
        neighbourhood=FixedRadius(0.1)
    )

    computation = RPComputation.create(settings)
    result = computation.run()
    ImageGenerator.save_recurrence_plot(
        result.recurrence_matrix_reverse, rplot_path
    )

    computation = RQAComputation.create(settings)
    result = computation.run()
    result.min_diagonal_line_length = 2
    result.min_vertical_line_length = 2
    result.min_white_vertical_line_length = 2

    return result


def idea_rqa(data_pts, rplot_path):
    time_series = TimeSeries(
        data_pts, embedding_dimension=1, time_delay=1
    )
    settings = Settings(
        time_series, analysis_type=Classic, 
        neighbourhood=FixedRadius(0.1)
    )

    computation = RPComputation.create(settings)
    result = computation.run()
    ImageGenerator.save_recurrence_plot(
        result.recurrence_matrix_reverse, rplot_path
    )

    computation = RQAComputation.create(settings)
    result = computation.run()
    result.min_diagonal_line_length = 2
    result.min_vertical_line_length = 2
    result.min_white_vertical_line_length = 2

    return result


def letter_stream_rqa(data_pts, embed, rplot_path):
    time_series = TimeSeries(
        data_pts, embedding_dimension=embed, time_delay=1
    )
    settings = Settings(
        time_series, analysis_type=Classic, 
        neighbourhood=FixedRadius(0.1)
    )

    computation = RPComputation.create(settings)
    result = computation.run()
    ImageGenerator.save_recurrence_plot(
        result.recurrence_matrix_reverse, rplot_path
    )

    computation = RQAComputation.create(settings)
    result = computation.run()
    result.min_diagonal_line_length = 2
    result.min_vertical_line_length = 2
    result.min_white_vertical_line_length = 2

    return result