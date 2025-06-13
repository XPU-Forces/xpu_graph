import torch
import torch_mlu
from torch import fx, nn

from ...utils.check_ops import get_shape


def find_longest_same_shape_sequence(inputs, start, end, mini_len, compare_input=False):
    """
    在[start, end]范围内找到shape相同的最长连续子序列
    Args:
        inputs: 输入序列
        start: 搜索起始位置
        end: 搜索结束位置
        mini_len: 最小长度要求
    Returns:
        (best_start, best_end): 最长相同shape子序列的起始和结束位置
    """
    best_start = start
    best_end = start
    best_length = 1

    for i in range(start, end + 1):
        if compare_input:
            current_shape = get_shape(inputs[i].args[0])
        else:
            current_shape = get_shape(inputs[i])
        current_start = i
        current_end = i

        for j in range(i + 1, end + 1):
            if compare_input:
                next_shape = get_shape(inputs[j].args[0])
            else:
                next_shape = get_shape(inputs[j])
            if next_shape == current_shape:
                current_end = j
            else:
                break

        current_length = current_end - current_start + 1

        if current_length > best_length:
            best_start = current_start
            best_end = current_end
            best_length = current_length

        if current_length == end - i + 1:
            break

    return best_start, best_end


def find_longest_same_param(inputs, start, end, mini_len):
    best_start = start
    best_end = start
    best_length = 1

    for i in range(start, end + 1):
        current_param = inputs[i].args[1:]
        current_start = i
        current_end = i

        for j in range(i + 1, end + 1):
            if inputs[j].args[1:] == current_param:
                current_end = j
            else:
                break

        current_length = current_end - current_start + 1

        if current_length > best_length:
            best_start = current_start
            best_end = current_end
            best_length = current_length

        if current_length == end - i + 1:
            break
    return best_start, best_end


def find_longest_same_input(inputs, start, end, mini_len):
    best_start = start
    best_end = start
    best_length = 1

    for i in range(start, end + 1):
        current_param = inputs[i].args[0]
        current_start = i
        current_end = i

        for j in range(i + 1, end + 1):
            if inputs[j].args[0] == current_param:
                current_end = j
            else:
                break

        current_length = current_end - current_start + 1

        if current_length > best_length:
            best_start = current_start
            best_end = current_end
            best_length = current_length

        if current_length == end - i + 1:
            break
    return best_start, best_end
