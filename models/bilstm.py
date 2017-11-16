import torch
import re
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

pretrained_dict = {}

def mean_embed(seq):
    cnt = 0
    embed = torch.cuda.FloatTensor(50)
    for token in seq:
        if len(token) == 0:
            continue
        if token not in pretrained_dict:
            continue
        cnt += 1.0
        embed += torch.cuda.FloatTensor(pretrained_dict[token])
    if cnt == 0:
        return embed
    return embed / cnt


class BiLSTM(nn.Module):
    def __init__(self, corpus, n_hidden):
        super(BiLSTM, self).__init__()
        self.corpus = corpus
        self.weight = torch.ones(len(self.corpus.label_dict) + 1).cuda()
        self.weight[0] = 0
        self.word_embed = nn.Embedding(len(corpus.token_dict) + 1, 50)
        self.word_embed.weight.requires_grad = False
        # self.chara_embed = nn.RNN(128, 50)
        self.pos_embed = nn.Embedding(len(corpus.pos_tag_dict) + 1, 25)
        self.lstm = nn.LSTM(input_size=50 + 25, hidden_size=n_hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * n_hidden, len(corpus.label_dict) + 1)
        self.drop = nn.Dropout(0.5)

    def load_pretrained(self, path):
        nn.init.constant(self.word_embed.weight, 0)
        with open(path, 'r') as f:
            for line in f:
                row = line.split()
                token = row[0]
                embed = [float(x) for x in row[1:]]
                pretrained_dict[token] = embed
        cnt = 0
        for k, v in tqdm(self.corpus.token_dict.items()):
            if k not in pretrained_dict:
                cnt += 1
                components = re.split('\_|\-|\/|\\|\;|\,|\.|\*|\n|\:|\&|\%', k)
                self.word_embed.weight[v] = mean_embed(components)
            else:
                self.word_embed.weight[v] = torch.cuda.FloatTensor(pretrained_dict[k])

    def forward(self, data, tags, label):
        word_input = self.word_embed(data)
        tag_input = self.pos_embed(tags)
        input = torch.cat([word_input, tag_input], -1)
        hiddens = self.lstm(input)
        out = self.fc(self.drop(hiddens[0]))
        return F.nll_loss(F.log_softmax(out).view(-1, len(self.corpus.label_dict) + 1), label.view(-1), weight=self.weight)

    def predict(self, data, tags, label):
        word_input = self.word_embed(data)
        tag_input = self.pos_embed(tags)
        input = torch.cat([word_input, tag_input], -1)
        hiddens = self.lstm(input)
        out = self.fc(self.drop(hiddens[0]))
        return out.max(dim=-1)[1].cpu()