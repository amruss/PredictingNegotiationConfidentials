import torch
import numpy as np

class DataPoint(object):
    def __init__(self, messages, p1_weights, p2_weights, items):
        self.messages = messages
        self.p1_weights = p1_weights
        self.p2_weights = p2_weights
        self.items = items
        self.target = []


class Message(object):
    def __init__(self, text, p1, index, word_tensor):
        self.text = text
        self.p1 = p1
        self.index = index
        self.word_tensor = word_tensor