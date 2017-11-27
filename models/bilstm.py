import torch
import re
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

pretrained_dict = {}
dim = 50
dim_pos = 50


def mean_embed(seq):
    cnt = 0
    embed = torch.FloatTensor(dim)
    for token in seq:
        if len(token) == 0:
            continue
        if token not in pretrained_dict:
            continue
        cnt += 1.0
        embed += torch.FloatTensor(pretrained_dict[token])
    if cnt == 0:
        return embed
    return embed / cnt


class BiLSTM(nn.Module):
    def __init__(self, corpus, n_hidden, device_id):
        super(BiLSTM, self).__init__()
        self.corpus = corpus
        self.weight_type = Variable(torch.ones(len(self.corpus.label_dict) + 1).cuda(device_id))
        self.weight_type[0] = 0
        self.weight_bio = Variable(torch.ones(3 + 1).cuda(device_id))
        self.weight_bio[0] = 0
        self.word_embed = nn.Embedding(len(corpus.token_dict) + 1, dim)
        # self.word_embed.weight.requires_grad = False
        # self.chara_embed = nn.RNN(128, 50)
        self.pos_embed = nn.Embedding(len(corpus.pos_tag_dict) + 1, dim_pos)
        self.lstm = nn.LSTM(input_size=dim + dim_pos
                            , hidden_size=n_hidden, bidirectional=True, num_layers=3,
                            batch_first=True)
        self.fc_bio = nn.Linear(2 * n_hidden, 3 + 1)
        self.fc_type = nn.Linear(2 * n_hidden, len(corpus.label_dict) + 1)
        print corpus.label_dict
        self.drop = nn.Dropout(0.7)

    def load_pretrained(self, path, vocab_path=None):
        pretrained = torch.FloatTensor(len(self.corpus.token_dict) + 1, dim)
        nn.init.constant(self.word_embed.weight, 0)
        if vocab_path == None:
            with open(path, 'r') as f:
                for line in f:
                    row = line.split()
                    token = row[0]
                    embed = [float(x) for x in row[1:]]
                    pretrained_dict[token] = embed
        else:
            token_lst = []
            with open(vocab_path, 'r') as f:
                for line in f:
                    token = line.strip()
                    token_lst.append(token)

            with open(path, 'r') as f:
                for token, line in zip(token_lst, f):
                    row = line.split()
                    embed = [float(x) for x in row]
                    pretrained_dict[token] = embed

        cnt = 0
        hit = 0
        for k, v in tqdm(self.corpus.token_dict.items()):
            cnt += 1
            if k not in pretrained_dict:
                components = re.split('\_|\-|\/|\\|\;|\,|\.|\*|\n|\:|\&|\%', k)
                pretrained[v] = mean_embed(components)
            else:
                hit += 1
                pretrained[v] = torch.FloatTensor(pretrained_dict[k])
        print hit * 1.0 / cnt
        self.word_embed.weight.data.copy_(pretrained)

    def forward(self, data, tags, lbl_bios, lbl_types, lengths):
        word_input = self.word_embed(data)
        tag_input = self.pos_embed(tags)
        input = torch.cat([word_input, tag_input], -1)
        packed = pack_padded_sequence(input, lengths, batch_first=True)
        hiddens = self.lstm(packed)[0]
        out, _ = pad_packed_sequence(hiddens, batch_first=True)
        out_type = self.fc_type(self.drop(out))
        out_bio = self.fc_bio(self.drop(out))
        return (F.nll_loss(F.log_softmax(out_type).view(-1, len(self.corpus.label_dict) + 1), lbl_types.view(-1),
                           weight=self.weight_type) +
                F.nll_loss(F.log_softmax(out_bio).view(-1, 3 + 1), lbl_bios.view(-1), weight=self.weight_bio))

    def predict(self, data, tags):
        word_input = self.word_embed(data)
        tag_input = self.pos_embed(tags)
        input = self.drop(torch.cat([word_input, tag_input], -1))
        hiddens = self.lstm(input)
        out_bio = self.fc_bio(hiddens[0])
        out_type = self.fc_type(hiddens[0])
        return out_bio.max(dim=-1)[1].cpu(), out_type.max(dim=-1)[1].cpu()
