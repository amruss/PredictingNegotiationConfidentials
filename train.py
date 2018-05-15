import argparse
import os

import torch.autograd
from tqdm import tqdm
from data_processing import *
from rnn import *
import random

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

#INPUT_LENGTH = MAX_LENGTH #+ 3 #p1_weight length
CONTEXT_INPUT_LENGTH = 70 #LONGEST_MESSAGE_WORD + 3
INPUT_LENGTH = 30 + 3

def train_batch(model, inp, target, batch_size, criterion, optimizer, hidden):
    model.zero_grad()
    loss = 0
    output, hidden = model(inp, hidden)
    o = output.view(batch_size, -1)
    t = target.float()
    loss += criterion(o, t)
    loss.backward()
    optimizer.step()
    return loss.data[0]

def train(model, train_dataset, training_epochs, criterion, optimizer, batch_size, val, test, context_model, print_progress=True):
    for epoch in range(1, training_epochs + 1):
        print "EPCOH " + str(epoch)
        loss_avg = 0
        for point in tqdm(train_dataset):
            tar = Variable(point.target)
            hidden = model.init_hidden(batch_size)
            context_hidden = context_model.init_hidden(BATCH_SIZE)
            l = 0
            for message in point.messages:
                model.zero_grad() #TODO: retrain with other vocab
                message_tensor = Variable(tensorfy(point.items + message.word_tensor, CONTEXT_INPUT_LENGTH))
                context_tensor, context_hidden = context_model(message_tensor, context_hidden)
                #input = point.p1_weights + message.word_tensor
                #inp = Variable(tensorfy(input))
                p1_weights = tensorfy(point.p1_weights, 3).float()
                model_input = torch.cat((p1_weights, context_tensor.data), 1)
                inp = Variable(model_input)
                loss = train_batch(model, inp, tar, batch_size, criterion, optimizer, hidden)
                l += loss
            loss_avg += l / len(point.messages)
        loss_avg = float(loss_avg) / len(train_dataset)

        if print_progress:
            print('Average Loss for epoch ' + str(epoch) +" is " + str(loss_avg))
            validate_with_context(model, val, criterion, context_model)

    print("Saving...")
    save(args.filename, model)

def validate_with_context(model, validation, criterion, context_model):
    index = random.randint(0, len(validation) -1)
    point = validation[index]
    tar = Variable(point.target)
    print "TARGET: " + str(get_numbers(tar))
    print "Items: " + str(point.items)
    print "P1_WEIGHTS: " + str(point.p1_weights)
    print "P2_WEIGHTS: " + str(point.p2_weights)
    hidden = model.init_hidden(BATCH_SIZE)
    context_hidden = context_model.init_hidden(BATCH_SIZE)
    model.zero_grad()
    for message in point.messages:
        message_tensor = Variable(tensorfy(point.items + message.word_tensor, CONTEXT_INPUT_LENGTH))
        context_tensor, context_hidden = context_model(message_tensor, context_hidden)
        inp = Variable(context_tensor.data)
        # input = point.p1_weights + message.word_tensor
        # inp = Variable(tensorfy(input))
        # loss = train_batch(model, inp, tar, BATCH_SIZE, criterion, optimizer, )
        loss = 0
        output, hidden = model(inp, hidden)
        o = output.view(BATCH_SIZE, -1)
        t = tar.float()
        loss += criterion(o, t)
        l = loss.data[0]
        print "MESSAGE: " + str(message.text)
        print "Predicted: " + str(get_numbers(output))

def validate(model, validation, criterion):
    index = random.randint(0, len(validation) -1)
    point = validation[index]
    tar = Variable(point.target)
    print "TARGET: " + str(get_numbers(tar))
    print "Items: " + str(point.items)
    print "P1_WEIGHTS: " + str(point.p1_weights)
    print "P2_WEIGHTS: " + str(point.p2_weights)
    hidden = model.init_hidden(BATCH_SIZE)
    model.zero_grad()
    for message in point.messages:
        input = point.p1_weights + message.word_tensor
        inp = Variable(tensorfy(input))
        # loss = train_batch(model, inp, tar, BATCH_SIZE, criterion, optimizer, )
        loss = 0
        output, hidden = model(inp, hidden)
        o = output.view(BATCH_SIZE, -1)
        t = tar.float()
        loss += criterion(o, t)
        l = loss.data[0]
        print "MESSAGE: " + str(message.text)
        print "Predicted: " + str(get_numbers(output))

def get_numbers(output):
    o = output.squeeze(0).data.numpy().tolist()
    boo = o[:11]
    ha = o[11:22]
    bal = o[22:]
    pred_books = str(boo.index(max(boo)))
    pred_hats = str(ha.index(max(ha)))
    pred_balls = str(bal.index(max(bal)))
    return pred_books, pred_hats, pred_balls

def tensorfy(list, length):
    inp = torch.LongTensor(BATCH_SIZE, length) #CONTEXT_INPUT_LENGTH
    ipt = torch.from_numpy(np.array(list))
    ipt = ipt.long()

    inp[0] = ipt #TODO: change for batch size
    return inp


def save(filename, model):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)

if __name__ == "__main__":

    #TODO: sperate into train/ test
    training_data, word_map = process_file(args.data_file)
    # length = 5916

    training = training_data[:4000]
    validation = training_data[4000:4900]
    test = training_data[4900:]

    input_size = INPUT_LENGTH
    hidden_size = int(input_size)
    output_size = 33

    #TODO
    if args.model == "None":
        model = RNN(input_size, hidden_size, output_size, args.layers)
    else:
        model = torch.load(args.model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MultiLabelSoftMarginLoss()


    FILENAME = 'context_model_5000_iterations_70words.pt'
    context_model = torch.load(FILENAME)
    del args.filename

    # Train the Model
    try:
        train(model, training, args.iterations, criterion, optimizer, args.batch_size, validation, test, context_model)
    except KeyboardInterrupt:
        print("Saving before quit...")
        save(args.filename, model)