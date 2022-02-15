"""
State to describe state of your robot.
"""
class FullState():
    def __init__(self, px, py, vf, radius, gx, gy, orientation, ranger_reflections):
        self.px = px
        self.py = py
        self.vf = vf
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.orientation = orientation

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)

        self.ranger_reflections = ranger_reflections

        self.state_tuple = (self.px, self.py, self.vf, self.radius, self.gx, self.gy, self.orientation,
                            self.ranger_reflections[0], self.ranger_reflections[1],
                            self.ranger_reflections[2], self.ranger_reflections[3])

    def __add__(self, other):
        return other + self.state_tuple

    def __str__(self):
        return ' '.join([str(x) for x in self.state_tuple])

    def __len__(self):
        return len(self.state_tuple)


class ObservableState():
    def __init__(self, goal_distance, orientation, ranger_reflections):
        self.goal_distance = goal_distance
        self.orientation = orientation
        self.ranger_reflections = ranger_reflections

        self.state_tuple = (self.goal_distance, self.orientation,
                            self.ranger_reflections[0], self.ranger_reflections[1],
                            self.ranger_reflections[2], self.ranger_reflections[3])
    
    def __add__(self, other):
        return other + self.state_tuple

    def __str__(self):
        return ' '.join([str(x) for x in self.state_tuple])

    def __len__(self):
        return len(self.state_tuple)
