import gzip
from sets import Set


class Corpus(object):
    def __init__(self):
        cnt = 0
        token_set = Set([])
        pos_tag_set = Set([])
        label_set = Set([])
        with gzip.open('data/train.txt.gz', 'r') as f:
            for line in f:
                triple = line.strip().split()
                if len(triple) == 0:
                    continue
                token, pos_tag, label = triple
                token = token.lower()
                token_set.add(token)
                pos_tag_set.add(pos_tag)
                label_set.add(label)

        with gzip.open('data/test.txt.gz', 'r') as f:
            for line in f:
                triple = line.strip().split()
                if len(triple) == 0:
                    continue
                token, pos_tag, label = triple
                token = token.lower()
                token_set.add(token)
                pos_tag_set.add(pos_tag)
                label_set.add(label)

        self.tokens = {e:i for i, e in enumerate(list(token_set), 1)}
        self.pos_tags = {e:i for i, e in enumerate(list(pos_tag_set), 1)}
        self.labels = {e:i for i, e in enumerate(list(label_set), 1)}

        self.tokens['</s>'] = 0
        self.pos_tags['</s>'] = 0
        self.pos_tags['</s>'] = 0

    @property
    def token_dict(self):
        return self.tokens

    @property
    def pos_tag_dict(self):
        return self.pos_tags

    @property
    def label_dict(self):
        return self.labels