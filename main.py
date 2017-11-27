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


parser = argparse.ArgumentParser("Chunking")
parser.add_argument('--pretrained', '-p', type=str, default='senna')
parser.add_argument('--device', '-d', type=int, default=0)
args = parser.parse_args()

pretrained_type = args.pretrained

torch.manual_seed(1111)

n_epochs = 30
corpus = utils.Corpus()

#logger = tmlog.Logger('LOG')
train_batch = Batches('data/train.txt.gz', corpus.token_dict, corpus.pos_tag_dict, corpus.label_dict, batch_size=16)
test_batch = Batches('data/test.txt.gz', corpus.token_dict, corpus.pos_tag_dict, corpus.label_dict, batch_size=128)

model = BiLSTM(corpus, 300, args.device)

if pretrained_type == 'glove':
    model.load_pretrained('pretrained/glove.6B.50d.txt')
elif pretrained_type == 'senna':
    model.load_pretrained('pretrained/embeddings.txt', 'pretrained/words.lst')

model.cuda(args.device)

lr = 1e-3

aver_acc = 0

for epoch in range(n_epochs):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr)
    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1)
    print "Epoch {}: ".format(epoch),
    model.train()
    aver_loss = 0
    for idx_batch, (data, tags, lbl_bios, lbl_types, lengths) in enumerate(train_batch):
        data = Variable(torch.LongTensor(data)).cuda(args.device)
        tags = Variable(torch.LongTensor(tags)).cuda(args.device)
        lbl_bios = Variable(torch.LongTensor(lbl_bios)).cuda(args.device)
        lbl_types = Variable(torch.LongTensor(lbl_types)).cuda(args.device)
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
        data = Variable(torch.LongTensor(data), volatile=True).cuda(args.device)
        tags = Variable(torch.LongTensor(tags), volatile=True).cuda(args.device)
        lbl_bios = Variable(torch.LongTensor(lbl_bios), volatile=True).cuda(args.device)
        lbl_types = Variable(torch.LongTensor(lbl_types), volatile=True).cuda(args.device)
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
        data = Variable(torch.LongTensor(data), volatile=True).cuda(args.device)
        tags = Variable(torch.LongTensor(tags), volatile=True).cuda(args.device)
        lbl_bios = Variable(torch.LongTensor(lbl_bios), volatile=True).cuda(args.device)
        lbl_types = Variable(torch.LongTensor(lbl_types), volatile=True).cuda(args.device)
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
        data = Variable(torch.LongTensor(data), volatile=True).cuda(args.device)
        tags = Variable(torch.LongTensor(tags), volatile=True).cuda(args.device)
        lbl_bios = Variable(torch.LongTensor(lbl_bios), volatile=True).cuda(args.device)
        lbl_types = Variable(torch.LongTensor(lbl_types), volatile=True).cuda(args.device)
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