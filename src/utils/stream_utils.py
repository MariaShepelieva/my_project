import torch
import numpy as np
from collections import deque, Counter

class GazeLabelAggregator:
    def __init__(self, window_size=30, fps=30):
        self.fps = fps
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, pred):
        self.buffer.append(pred)

    def get_dominant_label(self):
        if not self.buffer:
            return None
        counter = Counter(self.buffer)
        most_common = counter.most_common(1)[0]
        return most_common[0], most_common[1] / len(self.buffer)