from src.recurrence.rqa.computation import rqa

# Makes the first and last frame % of the data_pts epochs and 
# returns the epochs' RQA metric values
# All epochs use the same delay and embedding dim
def frame_epochs(data_pts, frame, delay, embed):
    if not 0 < frame < 50:
        raise Exception(
            'Parameter frame must take a value in range (0, 50)'
        )
    
    half_frame_data_count = round(frame/100 * len(data_pts))

    early_epoch = data_pts[:half_frame_data_count]
    late_epoch = data_pts[len(data_pts) - half_frame_data_count:]

    results = []

    for epoch in (early_epoch, late_epoch):
        res = rqa(epoch, delay, embed)
        results.append(res)
    
    return results


# Returns the RQA metric values of adjacent epochs, where parameters
# size and overlap correspond to epoch size and inter-epoch overlap
# All epochs use the same delay and embedding dim
# Parameter position determines how to deal with excess data pts ( );
# if another epoch will not fit,  
def adjacent_epochs(
        data_pts, size, overlap, delay=1, embed=1, position='left'
):
    if not 0 < size < len(data_pts)//2:
        raise Exception(
            'Parameter size must take a value in range'
            + ' (0, len(data_pts)//2]'
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
        res = rqa(epoch, delay, embed)
        results.append(res)

        lower_bound = upper_bound - overlap
        upper_bound = lower_bound + size
    
    return results


def fit_epochs(data_pts, size, overlap, position):
    # get excess data that will be excluded in epochs
    epoch_count = len(data_pts) // (size - overlap)

    if len(data_pts) % (size - overlap) < overlap:
        epoch_count -= 1
    
    epoch_data_count = size * epoch_count - overlap * (epoch_count - 1)
    excess_data_count = len(data_pts) - epoch_data_count

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




    


    

