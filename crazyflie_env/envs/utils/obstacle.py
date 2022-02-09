import numpy as np

class Obstacle():
    """
    Static rectangular obstacle.
    E.g. boxes, walls.
    """
    def __init__(self, centroid, wx, wy, angle=0):
        """
        param centroid: (tuple or list) centroid of the obstacle
        param wx: height of the obstacle (>=0)
        param wy: width of the obstacle (>=0)
        param angle: anti-clockwise rotation from the x-axis
        """
        self.centroid = centroid
        self.wx = wx
        self.wy = wy
        self.angle = angle

    
    def _get_points(self, centroid):
        """
        return: A wall ((x1, y1, x1', y1'))
                Or a box: ((x1,y1,x1',y1'), (x2,y2,x2',y2'), (x3,y3,x3',y3'), (x4,y4,x4',y4'))
        """
        wx_cos = self.wx * np.cos(self.angle)
        wx_sin = self.wx * np.sin(self.angle)
        wy_cos = self.wy * np.cos(self.angle)
        wy_sin = self.wy * np.sin(self.angle)

        BR_x = centroid[0] + 0.5*(wx_cos + wy_sin) # BR bottom-right
        BR_y = centroid[1] + 0.5*(wx_sin - wy_cos)
        BL_x = centroid[0] - 0.5*(wx_cos - wy_sin)
        BL_y = centroid[1] - 0.5*(wx_sin + wy_cos)
        TL_x = centroid[0] - 0.5*(wx_cos + wy_sin)
        TL_y = centroid[1] - 0.5*(wx_sin - wy_cos)
        TR_x = centroid[0] + 0.5*(wx_cos - wy_sin)
        TR_y = centroid[1] + 0.5*(wx_sin + wy_cos)

        seg_bottom = (BL_x, BL_y, BR_x, BR_y)
        seg_left = (BL_x, BL_y, TL_x, TL_y)

        if self.wy == 0: # if no height
            return (seg_bottom,)
        elif self.wx == 0: # if no width
            return (seg_left,)
        else: #if rectangle
            seg_top = (TL_x, TL_y, TR_x, TR_y)
            seg_right = (BR_x, BR_y, TR_x, TR_y)
            return (seg_bottom, seg_top, seg_left, seg_right)