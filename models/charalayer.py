import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CharaEmbedLayer(nn.Module):
    def __init__(self, inverted_vocab, n_chara=128, n_dim=30):
        super(CharaEmbedLayer, self).__init__()
        self.inverted_vocab = inverted_vocab
        self.inverted_vocab[0] = ''
        self.n_chara = n_chara
        self.n_dim = n_dim
        self.chara_embed = nn.Embedding(n_chara, n_dim, padding_idx=0)
        self.conv_5 = nn.Conv1d(1, 1, 5, padding=4)
        self.conv_4 = nn.Conv1d(1, 1, 4, padding=3)
        self.conv_3 = nn.Conv1d(1, 1, 3, padding=2)
        self.fc = nn.Linear(3 * n_dim, n_dim)

    """
    Input format:
    size: batch_size * length
    Ouptut format:
    size: batch_size * length * n_dim
    """

    def forward(self, input):
        len_max = -1
        batch_size = len(input)
        seq_length = len(input[0])
        tokens = [[self.inverted_vocab[token] for token in line] for line in input]
        for line in tokens:
            row_max = max([len(token) for token in line])
            if row_max > len_max:
                len_max = row_max

        input_array = []
        for line in tokens:
            row_array = []
            for token in line:
                token_array = []
                for chara in token:
                    token_array.append(ord(chara))
                token_array.extend([0] * (len_max - len(token_array)))
                row_array.append(token_array)
            input_array.append(row_array)

        """
        input_var
        size: batch_size, length, max_chara_len
        """
        input_var = Variable(torch.LongTensor(input_array).cuda())
        """
        input_emb
        size: batch_size * length, max_chara_len, n_dim 
        """
        input_emb = self.chara_embed(input_var.view(-1, len_max))
        out_5 = self.conv_5(input_emb.transpose(-1, -2).contiguous().view(-1, 1, len_max)).max(dim=-1)[0].view(batch_size, seq_length, self.n_dim)
        out_4 = self.conv_4(input_emb.transpose(-1, -2).contiguous().view(-1, 1, len_max)).max(dim=-1)[0].view(batch_size, seq_length, self.n_dim)
        out_3 = self.conv_3(input_emb.transpose(-1, -2).contiguous().view(-1, 1, len_max)).max(dim=-1)[0].view(batch_size, seq_length, self.n_dim)
        out = self.fc(torch.cat([out_3, out_4, out_5], dim=-1))
        return out

if __name__ == '__main__':
    chara_embed = CharaEmbedLayer({1: 'what', 2: 'the', 3: 'fuck'}).cuda()
    print chara_embed([[1, 0, 0], [2, 1, 0], [3, 2, 1]])