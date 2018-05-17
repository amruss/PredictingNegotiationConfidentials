# data processing functions

from data_classes import *
import torch
import numpy as np
import gzip
from sklearn.feature_extraction.text import CountVectorizer
import re

# Constants
item_indeces = [0, 2, 4]
weight_indeces = [1, 3, 5]
nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
nums_string = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
them = "THEM:"
you = "YOU:"
BATCH_SIZE = 1
LONGEST_MESSAGE_CHAR = 269
LONGEST_MESSAGE_WORD = 67 #TODO: CHNAGE TO 70
MAX_LENGTH = LONGEST_MESSAGE_WORD
TARGET_LENGTH = 3
word_map = {}

def process_file(data_file_name, num_lines=None):
    file = open(data_file_name, 'r')
    vec = CountVectorizer()
    x = vec.fit_transform(file).toarray()
    file.close()
    word_map = vec.vocabulary_

    lines = []
    with open(data_file_name, 'r') as f:
        for line in f:
            lines.append(line.strip())
    data_points = []
    iterator = iter(lines)

    old = next(iterator)
    for line in iterator:
        new = line
        point = process_line(new, old, word_map)
        if point != None:
            get_target(point)
            data_points.append(point)
        old = new
    return data_points, word_map


def process_line(p1, p2, map):
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
            word_list = message_text.split()
            w_list = get_word_list(word_list, map)
            message = Message(word_list, p1, indx, w_list)
        elif them in text:
            p1 = False
            message_text = text.split(them)[1]
            word_list = message_text.split()
            w_list = get_word_list(word_list, map)
            message = Message(word_list, p1, indx, w_list)
        messages.append(message)
        indx += 1

    data_point = DataPoint(messages, p1_weights, p2_weights, items)
    return data_point


def get_selection(line):
    m = re.search('disagree', line)
    n = re.search('no agreement', line)
    d = re.search('disconnect', line)
    if m == None and n==None and d==None:
        s = line.split("<eos>")[-2].split("<selection>")[1]
        item1 = int(s[7])
        item2 = int(s[15])
        item3 = int(s[23])
        if item1 not in nums:
            print "SOMETHING WRONG"
            print item1
        if item2 not in nums:
            print "SOMETHING WRONG"
            print item2
        if item3 not in nums:
            print "SOMETHING WRONG"
            print item3
        return [item1, item2, item3]
    else:
        return [0, 0, 0]


def get_reward(line):
    p = line.split("<eos>")[-1]
    reward = p[8]
    if reward not in nums_string:
        return 0
    return int(reward)

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

def tensorfy(data_point):
    inp = torch.LongTensor(BATCH_SIZE, MAX_LENGTH)
    target = torch.LongTensor(BATCH_SIZE, MAX_LENGTH)

def get_target(point):
    tar = torch.LongTensor(BATCH_SIZE, 33)
    agenda = point.p2_weights
    book_index = agenda[0]
    hat_index = 11 + agenda[1]
    ball_index = 22 + agenda[2]
    array = [0]*33
    array[book_index] = 1
    array[hat_index] =  1
    array[ball_index] = 1
    tensor = torch.from_numpy(np.array(array)).long()
    tar[0] = tensor
    point.target = tar


def get_word_list(word_list, map):
    indexes = []
    for w in range(len(word_list)):
        try:
            indexes.append(map[word_list[w]])
        except:
            continue
    if len(indexes) < MAX_LENGTH:
        padding = [0]
        indexes.extend(padding * (MAX_LENGTH - len(indexes)))
    if len(indexes) > MAX_LENGTH:
        indexes = indexes[:MAX_LENGTH]

    return indexes


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
    points, word_map = process_file(data_file_name, num_lines)
