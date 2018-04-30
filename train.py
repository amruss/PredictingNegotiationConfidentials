import argparse
import os

import torch.autograd
from tqdm import tqdm
from data_processing import *
from rnn import *

argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str, default="tester_model")
argparser.add_argument('--model', type=str, default="None")
argparser.add_argument('--data_file', type=str, default="data.txt")
argparser.add_argument('--layers', type=int, default=2)
argparser.add_argument('--batch_size', type=int, default=1)
argparser.add_argument('--learning_rate', type=float, default=0.0005)
argparser.add_argument('--iterations', type=int, default=60)

args = argparser.parse_args()
torch.manual_seed(1)

def train_batch(model, inp, target, batch_size, criterion, optimizer):
    hidden = model.init_hidden(batch_size)
    model.zero_grad()
    loss = 0
    for c in range(inp.data.shape[1]):
        output, hidden = model(inp[:,c], hidden)
        loss += criterion(output.view(batch_size, -1), target[:,c])
    loss.backward()
    optimizer.step()
    return loss.data[0] / c

def train(model, train_dataset, training_epochs, criterion, optimizer, batch_size, print_progress=True):
    for epoch in tqdm(range(1, training_epochs + 1)):
        loss_avg = 0
        for point in train_dataset:
            tar = Variable(point.target)
            for message in point.messages:
                inp = Variable(message.word_tensor)
                loss = train_batch(model, inp, tar, batch_size, criterion, optimizer)
            loss_avg += loss
        loss_avg = float(loss_avg) / len(train_dataset)

        if print_progress:
            print('Average Loss for epoch ' + str(epoch) +" is " + str(loss_avg))

    print("Saving...")
    save(args.filename, model)

def save(filename, model):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)

if __name__ == "__main__":

    #TODO: sperate into train/ test
    training_data, word_map = process_file(args.data_file)

    input_size = len(word_map.keys())
    hidden_size = int(input_size)
    output_size = 3

    #TODO
    if args.model == "None":
        model = RNN(input_size, hidden_size, output_size, args.layers)
    else:
        model = torch.load(args.model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train the Model
    try:
        train(model, training_data, args.iterations, criterion, optimizer, args.batch_size)
    except KeyboardInterrupt:
        print("Saving before quit...")
        save(args.filename, model)