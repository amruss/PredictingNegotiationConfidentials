from data_classes import *
import torch
import numpy as np

item_indeces = [0, 2, 4]
weight_indeces = [1, 3, 5]
them = "THEM:"
you = "YOU:"

def process_file(data_file_name, num_lines=None):
    lines = []
    with open(data_file_name, 'r') as f:
        for line in f:
            lines.append(line.strip())
    data_points = []
    iterator = iter(lines)

    old = next(iterator)
    for line in iterator:
        new = line
        point = process_line(new, old)
        if point != None:
            data_points.append(point)
        old = new

    return data_points


def process_line(p1, p2):
    p1_reward_text, p1_text, p1_inputs, p1_outputs = seperate_line(p1)
    p2_reward_text, p2_text, p2_inputs, p2_outputs = seperate_line(p2)

    check = check_lines(p1_inputs, p2_inputs, p1_text[0], p2_text[0])

    if not check:
        return None

    items = [p1_inputs[i] for i in item_indeces]
    p1_weights = [p1_inputs[i] for i in weight_indeces]
    p2_weights = [p2_inputs[i] for i in weight_indeces]

    indx = 0
    messages = []
    for text in p1_text:
        if you in text:
            p1 = True
            message_text = text.split(you)[1]
            message = Message(message_text, p1, indx)
        elif them in text:
            p1 = False
            message_text = text.split(them)[1]
            message = Message(message_text, p1, indx)
        messages.append(message)
        indx += 1

    data_point = DataPoint(messages, p1_weights, p2_weights, items)
    return data_point



def check_lines(p1_inputs, p2_inputs, p1_message, p2_message):
    p1_items = [p1_inputs[i] for i in item_indeces]
    p2_items = [p2_inputs[i] for i in item_indeces]

    if p1_items != p2_items:
       return False

    if (them not in p1_message) and (them not in p2_message):
        return False

    if (them in p1_message) and (them in p2_message):
        return False

    if (them in p1_message):
        p1 = p1_message.split(them)[1]
        p2 = p2_message.split(you)[1]
    else:
        p2 = p2_message.split(them)[1]
        p1 = p1_message.split(you)[1]

    if p1 != p2:
        return False

    return True


def seperate_line(p1):
    p1_split = p1.split()
    p1_text = p1.split("<eos>")
    p1_text[0] = p1_text[0][12:]

    reward_text = p1_text[-1][1:9]
    p1_inputs = map(int, p1_split[:6])
    p1_outputs = map(int, p1_split[-6:])
    p1_text = p1_text[:-2]

    return reward_text, p1_text, p1_inputs, p1_outputs


if __name__ == "__main__":
    data_file_name = "data.txt"
    num_lines = 1000
    points = process_file(data_file_name, num_lines)
