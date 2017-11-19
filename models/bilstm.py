import torch
import re
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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
        self.weight_type = Variable(torch.ones(len(self.corpus.label_dict) + 1).cuda())
        self.weight_type[0] = 0
        self.weight_bio = Variable(torch.ones(3 + 1).cuda())
        self.weight_bio[0] = 0
        self.word_embed = nn.Embedding(len(corpus.token_dict) + 1, 50)
        self.word_embed.weight.requires_grad = False
        #self.chara_embed = nn.RNN(128, 50)
        self.pos_embed = nn.Embedding(len(corpus.pos_tag_dict) + 1, 10)
        self.lstm = nn.LSTM(input_size=50+10, hidden_size=n_hidden, bidirectional=True, num_layers=1, batch_first=True)
        self.fc_bio = nn.Linear(2 * n_hidden, 3 + 1)
        self.fc_type = nn.Linear(2 * n_hidden, len(corpus.label_dict) + 1)
        self.drop = nn.Dropout(0.5)

    def load_pretrained(self, path):
        pretrained = torch.cuda.FloatTensor(len(self.corpus.token_dict) + 1, 50)
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
                pretrained[v] = mean_embed(components)
            else:
                pass
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
        return (F.nll_loss(F.log_softmax(out_type).view(-1, len(self.corpus.label_dict) + 1), lbl_types.view(-1), weight=self.weight_type) +
                F.nll_loss(F.log_softmax(out_bio).view(-1, 3 + 1), lbl_bios.view(-1), weight=self.weight_bio))


    def predict(self, data, tags):
        word_input = self.word_embed(data)
        tag_input = self.pos_embed(tags)
        input = torch.cat([word_input, tag_input], -1)
        hiddens = self.lstm(input)
        out_bio = self.fc_bio(hiddens[0])
        out_type = self.fc_type(hiddens[0])
        return out_bio.max(dim=-1)[1].cpu(), out_type.max(dim=-1)[1].cpu()
