import unicodedata
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import os
import torch.optim as optim
import learningData

letters = string.ascii_letters + ".,:'"


def toAscii(s):
    return ''.join(
        char for char in unicodedata.normalize('NFD', s)
        if unicodedata.category(char) != "Mn" and char in letters
    )


def lines(file):
    f = open(file, encoding="utf-8").read().strip().split("\n")
    return [toAscii(l) for l in f]


def charToIndex(char):
    return letters.find(char)


def charToTensor(char):
    ret = torch.zeros(1, len(letters))
    ret[0][charToIndex(char)] = 1
    return ret


def sentenceToTensor(sentence):
    sentence = toAscii(sentence.strip())
    ret = torch.zeros(len(sentence), 1, len(letters))
    for i, char in enumerate(sentence):
        ret[i][0][charToIndex(char)] = 1
    return ret


class Net(nn.Module):
    def __init__(self, input, hiddens, output):
        super(Net, self).__init__()
        self.hiddens = hiddens
        self.hid = nn.Linear(input + hiddens, hiddens)
        self.out = nn.Linear(input + hiddens, output)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        x = torch.cat((x, hidden), 1)
        new_hidden = self.hid(x)
        output = self.out(x)
        return output, new_hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hiddens))


def idFromOutput(out):
    _, i = out.data.topk(1)
    return i.item()


def getTrainData():
    id = random.choice(range(0, len(learningData.data)))
    sentence = random.choice(learningData.data[id])
    sentence_tensor = Variable(sentenceToTensor(sentence))
    id_tensor = Variable(torch.LongTensor([id]))
    return id, sentence, id_tensor, sentence_tensor


model = Net(len(letters), 128, len(learningData.data))
criterion = nn.NLLLoss()
if os.path.isfile('nn.pt'):
    model = torch.load('nn.pt')


def train(id_tensor, sentence_tensor):
    global output
    hidden = model.initHidden()
    model.zero_grad()
    for i in range(sentence_tensor.size()[0]):
        output, hidden = model(sentence_tensor[i], hidden)
    result = getResult(output)
    loss = criterion(model.logsoftmax(output), id_tensor)
    loss.backward()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()
    return output, loss, result


def getResult(output):
    ret = {}
    var = output.data[0]
    for i in var.size():
        for j in range(0, i):
            val = var[j].item()
            if val > 10:
                ret[j] = val
    ordered = {k: v for k, v in sorted(ret.items(), key=lambda item: item[1])}
    result = (-1, 0)
    if len(ordered) != 0:
        result = list(ordered.items())[-1]
    return result


def startTrain():
    sumavg = 0
    for i in range(1, 1000000):
        id, sentence, id_tensor, sentence_tensor = getTrainData()
        output, loss, result = train(id_tensor, sentence_tensor)
        sumavg = sumavg + loss.item()
        if i % 1000 == 0:
            print(sumavg / 1000)
            sumavg = 0
            torch.save(model, 'nn.pt')


def getAction(inputText):
    sentence_tensor = sentenceToTensor(inputText)
    global output
    hidden = model.initHidden()
    model.zero_grad()
    for i in range(sentence_tensor.size()[0]):
        output, hidden = model(sentence_tensor[i], hidden)
    result = getResult(output)
    return result



# startTrain()
