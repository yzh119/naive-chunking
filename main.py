import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import utils
#import training_monitor.logger as tmlog
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

#logger = tmlog.Logger('LOG')
train_batch = Batches('data/train.txt.gz', corpus.token_dict, corpus.pos_tag_dict, corpus.label_dict, batch_size=16)
test_batch = Batches('data/test.txt.gz', corpus.token_dict, corpus.pos_tag_dict, corpus.label_dict, batch_size=128)

model = BiLSTM(corpus, 200)
model.cuda()
# model.load_pretrained('pretrained/glove.6B.50d.txt')
lr = 1e-3

aver_acc = 0

for epoch in range(n_epochs):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr)
    print "Epoch {}: ".format(epoch),
    model.train()
    aver_loss = 0
    for idx_batch, (data, tags, lbl_bios, lbl_types, lengths) in enumerate(train_batch):
        #data = Variable(torch.cuda.LongTensor(data))
        tags = Variable(torch.cuda.LongTensor(tags))
        lbl_bios = Variable(torch.cuda.LongTensor(lbl_bios))
        lbl_types = Variable(torch.cuda.LongTensor(lbl_types))
        loss = model(data, tags, lbl_bios, lbl_types, lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#        logger.add_scalar('loss_' + str(epoch), loss.data[0], idx_batch)
        aver_loss += loss.data[0]

    print "loss: {} ".format(aver_loss),

    model.eval()

    hit_bio = 0
    hit = 0
    tot = 0
    last_type = 0
    for data, tags, lbl_bios, lbl_types, lengths in test_batch:
        #data = Variable(torch.cuda.LongTensor(data), volatile=True)
        tags = Variable(torch.cuda.LongTensor(tags), volatile=True)
        lbl_bios = Variable(torch.cuda.LongTensor(lbl_bios), volatile=True)
        lbl_types = Variable(torch.cuda.LongTensor(lbl_types), volatile=True)
        pred_bios, pred_types = model.predict(data, tags)

        for pred_lbl_bio, pred_lbl_type, truth_lbl_bio, truth_lbl_type in zip(pred_bios.view(-1).data, pred_types.view(-1).data,
                                                                              lbl_bios.view(-1).data, lbl_types.view(-1).data):
            if truth_lbl_bio > 0:
                tot += 1
                if pred_lbl_bio == 3:
                    last_type = pred_lbl_type
                pred_lbl_type = last_type
                if truth_lbl_bio == pred_lbl_bio:
                    hit_bio += 1
                    if truth_lbl_type == pred_lbl_type or truth_lbl_bio == 1:
                        hit += 1

    lr = lr * 0.95
    print "acc: {}, acc_bio: {}".format(hit * 1.0 / tot, hit_bio * 1.0 / tot)

with open('output_bio.txt', 'w') as f:
    inverted_token_dict = {v: k for k, v in corpus.token_dict.items()}
    inverted_pos_tag_dict = {v: k for k, v in corpus.pos_tag_dict.items()}
    inverted_lbl_dict = {v: k for k, v in corpus.label_dict.items()}

    end = True
    for data, tags, lbl_bios, lbl_types, lengths in test_batch:
        #data = Variable(torch.cuda.LongTensor(data), volatile=True)
        tags = Variable(torch.cuda.LongTensor(tags), volatile=True)
        lbl_bios = Variable(torch.cuda.LongTensor(lbl_bios), volatile=True)
        lbl_types = Variable(torch.cuda.LongTensor(lbl_types), volatile=True)
        pred_bios, pred_types = model.predict(data, tags)

        for token, tag, pred_lbl_bio, pred_lbl_type, truth_lbl_bio, truth_lbl_type in \
                zip(data.view(-1).data, tags.view(-1).data, pred_bios.view(-1).data, pred_types.view(-1).data,
                    lbl_bios.view(-1).data, lbl_types.view(-1).data):
            def convert_lbl(bio, type):
                prefix = ''
                suffix = '-NP'
                if bio == 3:
                    prefix = 'B'
                elif bio == 2:
                    prefix = 'I'
                elif bio == 1:
                    prefix = 'O'
                return prefix + suffix

            if truth_lbl_bio > 0:
                """
                print convert_lbl(truth_lbl_bio, truth_lbl_type), \
                    convert_lbl(pred_lbl_bio, pred_lbl_type)
                """
                f.write('{} {} {} {}\n'.format(inverted_token_dict[token],
                                               inverted_pos_tag_dict[tag],
                                               convert_lbl(truth_lbl_bio, truth_lbl_type),
                                               convert_lbl(pred_lbl_bio, pred_lbl_type)))
                end = True
            else:
                if end:
                    f.write('\n')
                    end = False

with open('output.txt', 'w') as f:
    inverted_token_dict = {v: k for k, v in corpus.token_dict.items()}
    inverted_pos_tag_dict = {v: k for k, v in corpus.pos_tag_dict.items()}
    inverted_lbl_dict = {v: k for k, v in corpus.label_dict.items()}

    end = True
    for data, tags, lbl_bios, lbl_types, lengths in test_batch:
        #data = Variable(torch.cuda.LongTensor(data), volatile=True)
        tags = Variable(torch.cuda.LongTensor(tags), volatile=True)
        lbl_bios = Variable(torch.cuda.LongTensor(lbl_bios), volatile=True)
        lbl_types = Variable(torch.cuda.LongTensor(lbl_types), volatile=True)
        pred_bios, pred_types = model.predict(data, tags)

        for token, tag, pred_lbl_bio, pred_lbl_type, truth_lbl_bio, truth_lbl_type in \
                zip(data.view(-1).data, tags.view(-1).data, pred_bios.view(-1).data, pred_types.view(-1).data,
                    lbl_bios.view(-1).data, lbl_types.view(-1).data):
            def convert_lbl(bio, type):
                prefix = ''
                suffix = ''
                if bio == 3:
                    prefix = 'B'
                elif bio == 2:
                    prefix = 'I'
                elif bio == 1:
                    prefix = 'O'

                if type > 0 and bio != 1:
                    suffix = '-' + inverted_lbl_dict[type]
                return prefix + suffix

            if truth_lbl_bio > 0:
                """
                print convert_lbl(truth_lbl_bio, truth_lbl_type), \
                    convert_lbl(pred_lbl_bio, pred_lbl_type)
                """
                f.write('{} {} {} {}\n'.format(inverted_token_dict[token],
                                               inverted_pos_tag_dict[tag],
                                               convert_lbl(truth_lbl_bio, truth_lbl_type),
                                               convert_lbl(pred_lbl_bio, pred_lbl_type)))
                end = True
            else:
                if end:
                    f.write('\n')
                    end = False