# Stress position indexes:
    # U = 0 (not prominent)
    # S = 1 (semi-prominent)
    # P = 2 (prominent)
def stress_data_pts(stresses):
    data_pts = []
    position_to_index = {'U': 0, 'S': 1, 'P': 2}

    for stress in stresses.values():
        positions = stress.split('|')

        for pos in positions:
            for slot in pos:
                data_pts.append(position_to_index[slot])
        
    return data_pts