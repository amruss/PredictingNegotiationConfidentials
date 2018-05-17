import argparse
import os

import torch.autograd
from tqdm import tqdm
from context_processing import *
from rnn import *
import random

argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str, default="tester_model")
argparser.add_argument('--model', type=str, default="None")
argparser.add_argument('--data_file', type=str, default="data.txt")
argparser.add_argument('--layers', type=int, default=2)
argparser.add_argument('--batch_size', type=int, default=1)
argparser.add_argument('--learning_rate', type=float, default=0.0005)
argparser.add_argument('--iterations', type=int, default=1000)
args = argparser.parse_args()
torch.manual_seed(1)


# Trains the model for one batch of data points
def train_batch(model, inp, target, batch_size, criterion, optimizer):
    hidden = model.init_hidden(batch_size)
    model.zero_grad()
    loss = 0
    output, hidden = model(inp, hidden)
    o = output.view(batch_size, -1)
    t = target.float()
    loss += criterion(o, t)
    loss.backward()
    optimizer.step()
    return loss.data[0]


# Run training for all iterations
def train(model, x_s, y_s, training_epochs, criterion, optimizer, batch_size, test_xs, test_ys, rev_map, print_progress=True):
    for epoch in range(1, training_epochs + 1):
        loss_avg = 0
        for i in range(len(x_s)):
            if len(x_s[i][0]) != MAX_LENGTH:
                print len(x_s[i])
            tar = Variable(y_s[i])
            inp = Variable(x_s[i])
            loss = train_batch(model, inp, tar, batch_size, criterion, optimizer)
            loss_avg += loss

        if print_progress and (epoch % 10 == 0):
            print "--------------------------------"
            print('Average Loss for epoch ' + str(epoch) +" is " + str(loss_avg))
            print('AVERAGE TEST LOSS')
            print(test(model, test_xs, test_ys, batch_size, criterion, optimizer, rev_map))

    print("Saving...")
    save(args.filename, model)


# Test the model
def test(model, x_s, y_s, batch_size, criterion, optimizer, rev_map):
    l = 0
    correct = []
    for i in range(len(x_s)):
        target = Variable(y_s[i])
        inp = Variable(x_s[i])

        hidden = model.init_hidden(batch_size)
        model.zero_grad()
        loss = 0
        output, hidden = model(inp, hidden)
        o = output.view(batch_size, -1)
        t = target.float()
        loss += criterion(o, t)
        l += loss.data[0]

        y = y_s[i].squeeze(0).numpy().tolist()
        y_boo = y[:10]
        y_ha = y[10:20]
        y_bal = y[20:]
        y_books = y_boo.index(max(y_boo))
        y_hats = y_ha.index(max(y_ha))
        y_balls = y_bal.index(max(y_bal))

        o = output.squeeze(0).data.numpy().tolist()
        boo = o[:10]
        ha = o[10:20]
        bal = o[20:]
        pred_books = boo.index(max(boo))
        pred_hats = ha.index(max(ha))
        pred_balls = bal.index(max(bal))

        number_correct = 0

        if (pred_books == y_books):
            number_correct += 1

        if (pred_hats == y_hats):
            number_correct += 1

        if (pred_balls == y_balls):
            number_correct += 1

        correct.append(number_correct)

    correct_avg = sum(correct) / float(len(correct))
    print "Average Number Correct: " + str(correct_avg)

    string = ""
    index = random.randint(0, len(x_s) -1)
    x = x_s[index].squeeze(0)
    y = y_s[index].squeeze(0)


    book = str(x[0])
    hat = str(x[1])
    ball = str(x[2])
    words = x[3:]
    for word in words:
        if word in rev_map:
            string = string + " " + rev_map[int(word)]

    target = Variable(y_s[index])
    inp = Variable(x_s[index])
    hidden = model.init_hidden(batch_size)
    model.zero_grad()
    loss = 0
    output, hidden = model(inp, hidden)

    o = output.squeeze(0).data.numpy().tolist()
    boo = o[:10]
    ha = o[10:20]
    bal = o[20:]
    pred_books = str(boo.index(max(boo)))
    pred_hats = str(ha.index(max(ha)))
    pred_balls = str(bal.index(max(bal)))


    if pred_books == "9":
        pred_books = "-"
    if pred_hats == "9":
        pred_hats = "-"
    if pred_balls == "9":
        pred_balls = "-"

    print "Starting Books: " + book + " Starting Hats: " + hat + " Starting Balls: " + ball
    print "String: " + string
    print "Output: "
    print "Predicted Books: " + pred_books + " Predicted Hats: " + pred_hats + " Predicted Balls: " + pred_balls


    return float(l) / len(x_s)


# Save the model
def save(filename, model):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)


if __name__ == "__main__":
    x_s, y_s, map = process_file("data/context.txt")
    rev_map = {v: k for k, v in map.iteritems()}
    training_x = x_s[:200]
    training_y = y_s[:200]
    test_x = x_s[200:]
    test_y = y_s[200:]

    if len(x_s) != len(y_s):
        print "SOMETHING WRONG"

    input_size = MAX_LENGTH
    hidden_size = int(input_size)
    output_size = 30

    if args.model == "None":
        model = RNN(input_size, hidden_size, output_size, args.layers)
    else:
        model = torch.load(args.model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MultiLabelSoftMarginLoss()

    # Train the Model
    try:
        train(model, x_s, y_s, args.iterations, criterion, optimizer, args.batch_size, test_x, test_y, rev_map)
    except KeyboardInterrupt:
        print("Saving before quit...")
        save(args.filename, model)