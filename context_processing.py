from data_classes import *
import torch
import numpy as np
import gzip
from sklearn.feature_extraction.text import CountVectorizer
import re

item_indeces = [0, 2, 4]
weight_indeces = [1, 3, 5]
nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
nums_string = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
them = "THEM:"
you = "YOU:"
BATCH_SIZE = 1
LONGEST_MESSAGE_CHAR = 269
LONGEST_MESSAGE_WORD = 70
MAX_LENGTH = LONGEST_MESSAGE_WORD
TARGET_LENGTH = 3
NUM_CLASSES = 6

word_map = {}

def process_file(data_file_name):
    vocabulary = get_vocabulary(data_file_name)

    x_s = []
    y_s = []
    with open(data_file_name, 'r') as f:
        for line in f:
            x, y = process_line(line.strip(), vocabulary)
            x_s.append(x)
            y_s.append(y)
    return x_s, y_s, vocabulary

def get_vocabulary(filename):
    vocab = {}
    i = 11
    max = 0
    with open(filename, 'r') as f:
        for line in f:
            words = line.strip().split(" ")
            for word in words:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
    return vocab

def process_line(line, map):
    book_num = int(line[0])
    hat_num = int(line[2])
    ball_num = int(line[4])

    book_offer = line[6]
    if book_offer == "-":
        book_offer = 9
    else:
        book_offer = int(book_offer)

    hat_offer = line[8]
    if hat_offer == "-":
        hat_offer = 9
    else:
        hat_offer = int(hat_offer)

    ball_offer = line[10]
    if ball_offer == "-":
        ball_offer = 9
    else:
        ball_offer = int(ball_offer)

    message = line[12:]

    word_list = word_tensor(message, map, [book_num, hat_num, ball_num])
    target = get_target(book_offer, hat_offer, ball_offer)
    return word_list, target

def get_word_list(string, vocabulary):
    words = string.split(" ")
    word_list = []
    for word in words:
        word_list.append(vocabulary[word])
    return word_list

def get_target(book_offer, hat_offer, ball_offer):
    # target = torch.LongTensor(BATCH_SIZE, TARGET_LENGTH, NUM_CLASSES)
    tar = torch.LongTensor(BATCH_SIZE, 30)
    book_index = book_offer
    hat_index = 10 + hat_offer
    ball_index = 20 + ball_offer
    array = [0]*30
    array[book_index] = 1
    array[hat_index] =  1
    array[ball_index] = 1
    tensor = torch.from_numpy(np.array(array)).long()
    tar[0] = tensor


    # book_array = [0]*NUM_CLASSES
    # book_array[book_offer] = 1
    # book_tensor = torch.from_numpy(np.array(book_array)).long()
    #
    # hat_array = [0]*NUM_CLASSES
    # hat_array[hat_offer] = 1
    # hat_tensor = torch.from_numpy(np.array(hat_array)).long()
    #
    # ball_array = [0]*NUM_CLASSES
    # ball_array[ball_offer] = 1
    # ball_tensor = torch.from_numpy(np.array(ball_array)).long()
    #
    # target[0][0] = book_tensor
    # target[0][1] = hat_tensor
    # target[0][2] = ball_tensor

    return tar


def word_tensor(string, map, amounts):
    word_list = string.split(" ")
    # tensor = torch.zeros(len(MAX_LENGTH)).long()
    indexes = amounts
    for w in range(len(word_list)):
        try:
            indexes.append(map[word_list[w]])
        except:
            continue
    if len(indexes) < MAX_LENGTH:
        padding = [10]
        indexes.extend(padding * (MAX_LENGTH - len(indexes)))


    inp = torch.LongTensor(BATCH_SIZE, MAX_LENGTH)
    ipt = torch.from_numpy(np.array(indexes))
    ipt = ipt.float()

    inp[0] = ipt #TODO: change for batch size

    return inp


if __name__ == "__main__":
    data_file_name = "context.txt"
    num_lines = 1000
    x_s, y_s, vocab = process_file(data_file_name)
