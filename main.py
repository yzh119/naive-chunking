import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import utils
import training_monitor.logger as tmlog
from loader import *
from models.bilstm import *
import argparse
import json
import time
import logging

"""
parser = argparse.ArgumentParser("Chunking")
parser.add_argument()

args = parser.parse_args()
"""

n_epochs = 100
corpus = utils.Corpus()

logger = tmlog.Logger('LOG')
train_batch = Batches('data/train.txt.gz', corpus.token_dict, corpus.pos_tag_dict, corpus.label_dict)
test_batch = Batches('data/test.txt.gz', corpus.token_dict, corpus.pos_tag_dict, corpus.label_dict)

model = BiLSTM(corpus, 200)
model.cuda()
model.load_pretrained('pretrained/glove.6B.50d.txt')
optim = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)

for epoch in range(n_epochs):
    print "Epoch {}: ".format(epoch),
    model.train()
    aver_loss = 0
    for idx_batch, (data, tags, label) in enumerate(train_batch):
        data = Variable(torch.cuda.LongTensor(data))
        tags = Variable(torch.cuda.LongTensor(tags))
        label = Variable(torch.cuda.LongTensor(label))
        loss = model(data, tags, label)
        optim.zero_grad()
        loss.backward()
        optim.step()
        logger.add_scalar('loss_' + str(epoch), loss.data[0], idx_batch)
        aver_loss += loss.data[0]

    print "loss: {} ".format(aver_loss),

    model.eval()

    hit = 0
    tot = 0
    for data, tags, label in test_batch:
        data = Variable(torch.cuda.LongTensor(data), volatile=True)
        tags = Variable(torch.cuda.LongTensor(tags), volatile=True)
        label = Variable(torch.cuda.LongTensor(label), volatile=True)

        for pred, truth in zip(model.predict(data, tags, label).view(-1).data,
                               label.view(-1).data):
            if truth > 0:
                tot += 1
                if pred == truth:
                    hit += 1

    print "acc: {}".format(hit * 1.0 / tot)