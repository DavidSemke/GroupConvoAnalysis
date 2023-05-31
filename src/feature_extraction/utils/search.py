# returns (True, index) if item found at index
# returns (False, index) if item not found; insert at index
def binary_search(dict_list, compare_key, val_to_find):
    low = 0
    high = len(dict_list) - 1
    mid = 0
 
    while low <= high:
        mid = (high + low) // 2

        # if x is greater, ignore left half
        if dict_list[mid][compare_key] < val_to_find:
            low = mid + 1
        # if x is smaller, ignore right half
        elif dict_list[mid][compare_key] > val_to_find:
            high = mid - 1
        # means x is present at mid
        else:
            return (True, mid)
 
    # element was not present
    # insert at low
    return (False, low)







