import torch


def match_sub_list(lst, check_func):
    """
    Find the longest consecutive subsequence satisfying the given predicate.

    Args:
        lst (list): Input list to search through
        check_func (callable): Predicate function that returns True/False for each element

    Returns:
        tuple: (start_index, end_index) of the longest matching subsequence
               Returns (-1, -1) if no matching subsequence is found
    """
    longest_start, longest_end = -1, -1
    max_length = 0
    current_start = -1

    for idx, item in enumerate(lst):
        if check_func(item):
            # Start a new sequence or continue the current one
            if current_start == -1:
                current_start = idx

            # Update longest subsequence if current sequence is longer
            current_length = idx - current_start + 1
            if current_length > max_length:
                max_length = current_length
                longest_start = current_start
                longest_end = idx
        else:
            # Reset current sequence
            current_start = -1

    return longest_start, longest_end
