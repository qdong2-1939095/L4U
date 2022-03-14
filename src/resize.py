
def resize_image(left, right, top, bottom, width, height):
    mid = [(left + right) / 2, (top + bottom) / 2]
    result = [0, width - 1, 0, height - 1]
    scale = [2 * min(mid[0], width - mid[0]), 2 * min(mid[1], height - mid[1])]
    # process in parallel direction first
    if scale[0] / scale[1] < width / height:
        if mid[0] < (width - mid[0]):
            result[1] = mid[0] * 2
        else:
            result[0] = width - (width - mid[0]) * 2
        # then cut vertically
        result_half_height = scale[0] / width * height / 2
        result[2] = mid[1] - result_half_height
        result[3] = mid[1] + result_half_height

    else: # process in vertical direction first
        if mid[1] < (height - mid[1]):
            result[3] = mid[1] * 2
        else:
            result[2] = height - (height - mid[1]) * 2
        # then cut in parallel
        result_half_weight = scale[1] / height * width / 2
        result[0] = mid[0] - result_half_weight
        result[1] = mid[0] + result_half_weight
    return result
