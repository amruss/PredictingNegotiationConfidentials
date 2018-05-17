# Script for running logistic regressor experiments

from data_processing import *
from sklearn import linear_model

FILENAME = 'models/context_model_5000_iterations_70words_dataVocab.pt'
context_model = torch.load(FILENAME)
context_hidden = context_model.init_hidden(1)


data_file_name = "data/data.txt"
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

X = []
Y =[]

for line in iterator:
    new = line
    p1_reward_text, p1_text, p1_inputs, p1_outputs = seperate_line(new)
    p2_reward_text, p2_text, p2_inputs, p2_outputs = seperate_line(old)
    check = check_lines(p1_inputs, p2_inputs, p1_text[0], p2_text[0])
    # nltk.sentiment.util.demo_vader_instance()

    point = process_line(new, old, word_map)

    if check:
        items = [p1_inputs[i] for i in item_indeces]
        p1_weights = [p1_inputs[i] for i in weight_indeces]
        p2_weights = [p2_inputs[i] for i in weight_indeces]
        p1_got = get_selection(new)
        p2_got = get_selection(old)
        p1_reward = get_reward(new)
        p2_reward = get_reward(old)

        X.append(p1_weights)
        Y.append(p2_reward)

        X.append(p2_weights)
        Y.append(p1_reward)

    old = new

#
x_train = X[:5000]
x_test  = X[5000:8000]
x_val = X[8000:]

y_train = Y[:5000]
y_test = Y[5000:8000]
Y_val = Y[8000:]

lm = linear_model.LogisticRegression(C=1e-1)
lm.fit(x_train, y_train)

print "Dev Score: "
print(lm.score(x_test, y_test))


print "Val Score: "
print(lm.score(x_val, Y_val))

