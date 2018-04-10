class DataPoint(object):
    def __init__(self, messages, p1_weights, p2_weights, items):
        self.messages = messages
        self.p1_weights = p1_weights
        self.p2_weights = p2_weights
        self.items = items



class Message(object):
    def __init__(self, text, p1, index):
        self.text = text
        self.p1 = p1
        self.index = index