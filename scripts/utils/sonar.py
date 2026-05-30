## Unused for 3 sonar 5 bin
import numpy as np

MAX_SONAR = 1.5
MIN_SONAR = 0.05

THRESH_CLOSE = 0.4
THRESH_MEDIUM = 0.8

SENSOR_KEYS = [
    "left_0", "left_1", "left_2",
    "front",
    "right_0", "right_1", "right_2"
]

def discretize_distance(d):
    if d <= THRESH_CLOSE:
        return 0
    elif d <= THRESH_MEDIUM:
        return 1
    else:
        return 2

class SonarDiscretizer:
    def __init__(self, keys=None):
        self.keys = keys if keys is not None else SENSOR_KEYS

    def normalize(self, raw):
        if raw is None:
            return MAX_SONAR
        return float(np.clip(raw, MIN_SONAR, MAX_SONAR))

    def process(self, raw_sonar_dict):
        state = []
        for k in self.keys:
            d = self.normalize(raw_sonar_dict.get(k, MAX_SONAR))
            state.append(discretize_distance(d))
        return tuple(state)

    def is_crash(self, raw_sonar_dict, crash_threshold=0.25):
        front = raw_sonar_dict.get("front", MAX_SONAR)
        front = self.normalize(front)
        return front < crash_threshold
