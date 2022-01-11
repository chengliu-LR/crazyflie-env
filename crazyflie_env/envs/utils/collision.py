import numpy as np

def point_to_segment_dist(end_point_1, end_point_2, point):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)
    """

    x1, y1 = end_point_1
    x2, y2 = end_point_2
    x3, y3 = point

    px = x2 - x1
    py = y2 - y1

    # if a line segment is a point
    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))
    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))
