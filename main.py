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
torch.manual_seed(1111)

n_epochs = 100
corpus = utils.Corpus()

logger = tmlog.Logger('LOG')
train_batch = Batches('data/train.txt.gz', corpus.token_dict, corpus.pos_tag_dict, corpus.label_dict, batch_size=16)
test_batch = Batches('data/test.txt.gz', corpus.token_dict, corpus.pos_tag_dict, corpus.label_dict, batch_size=128)

model = BiLSTM(corpus, 200)
model.cuda()
model.load_pretrained('pretrained/glove.6B.50d.txt')
lr = 1e-3

aver_acc = 0

for epoch in range(n_epochs):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr)
    print "Epoch {}: ".format(epoch),
    model.train()
    aver_loss = 0
    for idx_batch, (data, tags, lbls, lengths) in enumerate(train_batch):
        data = Variable(torch.cuda.LongTensor(data))
        tags = Variable(torch.cuda.LongTensor(tags))
        lbls = Variable(torch.cuda.LongTensor(lbls))
        loss = model(data, tags, lbls, lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.add_scalar('loss_' + str(epoch), loss.data[0], idx_batch)
        aver_loss += loss.data[0]

    print "loss: {} ".format(aver_loss),

    model.eval()

    hit = 0
    tot = 1
    for data, tags, lbls, lengths in test_batch:
        data = Variable(torch.cuda.LongTensor(data), volatile=True)
        tags = Variable(torch.cuda.LongTensor(tags), volatile=True)
        lbls = Variable(torch.cuda.LongTensor(lbls), volatile=True)
        preds = model.predict(data, tags)
        for pred_lbl, truth_lbl in zip(preds.view(-1).data, lbls.view(-1).data):
            if truth_lbl > 0:
                tot += 1
                if pred_lbl == truth_lbl:
                    hit += 1

    lr = lr * 0.9 + 1e-5
    print "accuracy: {}".format(hit * 1.0 / tot)

with open('output.txt', 'w') as f:
    inverted_token_dict = {v: k for k, v in corpus.token_dict.items()}
    inverted_pos_tag_dict = {v: k for k, v in corpus.pos_tag_dict.items()}
    inverted_lbl_dict = {v: k for k, v in corpus.label_dict.items()}

    end = True
    for data, tags, lbls, lengths in test_batch:
        data = Variable(torch.cuda.LongTensor(data), volatile=True)
        tags = Variable(torch.cuda.LongTensor(tags), volatile=True)
        lbls = Variable(torch.cuda.LongTensor(lbls), volatile=True)
        preds = model.predict(data, tags)
        for token, tag, pred_lbl, truth_lbl in zip(data.view(-1).data, tags.view(-1).data, preds.view(-1).data, lbls.view(-1).data):
            if truth_lbl > 0:
                f.write('{} {} {} {}\n'.format(inverted_token_dict[token],
                                               inverted_pos_tag_dict[tag],
                                               inverted_lbl_dict[truth_lbl],
                                               inverted_lbl_dict[pred_lbl]))
                end = True
            else:
                if end:
                    f.write('\n')
                    end = False