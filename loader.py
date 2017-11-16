import gzip
import random


class Batches(object):
    def __init__(self, path, token_dict, pos_tag_dict, label_dict, batch_size=32):
        self.batch_size = batch_size
        self.data = []
        with gzip.open(path, 'r') as f:
            current_sts_triples = []
            for line in f:
                triple = line.strip().split()
                if len(triple) == 0:
                    self.data.append(current_sts_triples)
                    current_sts_triples = []
                else:
                    token, pos_tag, lbl = triple
                    token = token.lower()
                    current_sts_triples.append((token_dict[token], pos_tag_dict[pos_tag], label_dict[lbl]))

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        random.shuffle(self.data)
        for i in range(0, len(self.data), self.batch_size):
            max_len = -1
            for row in self.data[i: i + self.batch_size]:
                if len(row) > max_len:
                    max_len = len(row)
            tokens = []
            pos_tags = []
            lbls = []
            for row in self.data[i: i + self.batch_size]:
                row_tokens = []
                row_pos_tags = []
                row_lbls = []
                for token, pos_tag, lbl in row:
                    row_tokens.append(token)
                    row_pos_tags.append(pos_tag)
                    row_lbls.append(lbl)
                row_tokens.extend([0] * (max_len - len(row_tokens)))
                row_pos_tags.extend([0] * (max_len - len(row_pos_tags)))
                row_lbls.extend([0] * (max_len - len(row_lbls)))
                tokens.append(row_tokens)
                pos_tags.append(row_pos_tags)
                lbls.append(row_lbls)
            yield tokens, pos_tags, lbls