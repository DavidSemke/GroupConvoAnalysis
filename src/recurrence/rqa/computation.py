from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.computation import RQAComputation, RPComputation
from pyrqa.image_generator import ImageGenerator


# Makes the first and last frame % of the data_pts epochs and 
# returns the epochs' RQA metric values
# All epochs use the same delay and embedding dim
def frame_epochs(data_pts, frame, delay, embed):
    
    if not 0 < frame < 50:
        raise Exception(
            'Parameter frame must take a value in range (0, 50)'
        )
    
    frame_data_count = round(frame/100 * len(data_pts))

    early_epoch = data_pts[:frame_data_count]
    late_epoch = data_pts[len(data_pts) - frame_data_count:]

    results = []

    for epoch in (early_epoch, late_epoch):
        out = rqa(epoch, delay, embed)
        results.append(out[0])
    
    return results


# Returns the RQA metric values of adjacent epochs, where parameters
# size and overlap correspond to epoch size and inter-epoch overlap
# All epochs use the same delay and embedding dim
# Parameter position determines how to deal with excess data pts;
    # left - first epoch starts at index 0
    # right - last epoch ends at index -1
    # center - remove excess from left and right
def sliding_epochs(
        data_pts, size, overlap, delay=1, embed=1, position='left'
):

    # this ensures at least 2 epochs
    if not 0 < size <= (len(data_pts) + overlap) // 2:
        raise Exception(
            'Parameter size must take a value in range'
            + ' (0, (len(data_pts) + overlap) // 2]'
        )
    
    if not 0 <= overlap < size:
        raise Exception(
            'Parameter overlap must take a value in range [0, size)'
        )
    
    data_pts = fit_epochs(data_pts, size, overlap, position)
    results = []
    lower_bound = 0
    upper_bound = size

    while upper_bound <= len(data_pts):
        epoch = data_pts[lower_bound:upper_bound]
        out = rqa(epoch, delay, embed)
        results.append(out[0])

        lower_bound = upper_bound - overlap
        upper_bound = lower_bound + size
    
    return results


def fit_epochs(data_pts, size, overlap, position):
    # get excess data count that will be excluded from epochs
    data_count = len(data_pts)
    leftovers = size - overlap
    epoch_count = (data_count - overlap) // leftovers
    epoch_data_count = leftovers * epoch_count + overlap
    excess_data_count = data_count - epoch_data_count

    # remove excess data
    if position == 'left':
        data_pts = data_pts[:len(data_pts) - excess_data_count]
    
    elif position == 'right':
        data_pts = data_pts[excess_data_count:]
    
    elif position == 'center':
        half = excess_data_count // 2

        if excess_data_count % 2 == 0:
            data_pts = data_pts[half : len(data_pts)-half]
        else:
            data_pts = data_pts[half : len(data_pts)-half-1]
    
    else:
        raise Exception(
            'Parameter position must be "left", "right", or "center"'
        )

    return data_pts


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