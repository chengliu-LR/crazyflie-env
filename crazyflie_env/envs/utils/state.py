# JointState = ObservableState + FullState
class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

        self.state_tuple = (self.px, self.py, self.vx, self.vy, self.radius)

    def __add__(self, other):
        return other + self.state_tuple

    def __str__(self):
        return ' '.join([str(x) for x in self.state_tuple])
    
    def __len__(self):
        return len(self.state_tuple)


class FullState():
    def __init__(self, px, py, vx, vy, radius, gx, gy):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

        self.state_tuple = (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy)

    def __add__(self, other):
        return other + self.state_tuple

    def __str__(self):
        return ' '.join([str(x) for x in self.state_tuple])
    
    def __len__(self):
        return len(self.state_tuple)


class JointState():
    def __init__(self, self_state, obstacle_states):
        assert isinstance(self_state, FullState)
        for obstacle_state in obstacle_states:
            assert isinstance(obstacle_state, ObservableState)
        
        self.self.state = self_state
        self.obstacle_states = obstacle_states