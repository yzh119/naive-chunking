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
                    if len(lbl) == 1:
                        current_sts_triples.append((token_dict[token], pos_tag_dict[pos_tag], 1, 0))
                    elif lbl[0] == 'I':
                        current_sts_triples.append((token_dict[token], pos_tag_dict[pos_tag], 2, label_dict[lbl[2:]]))
                    elif lbl[0] == 'B':
                        current_sts_triples.append((token_dict[token], pos_tag_dict[pos_tag], 3, label_dict[lbl[2:]]))
                    else:
                        assert False, "CUNT"

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        random.shuffle(self.data)
        for i in range(0, len(self.data), self.batch_size):
            batch = sorted(self.data[i: i + self.batch_size], key=lambda lst: len(lst), reverse=True)
            max_len = len(batch[0])
            tokens = []
            pos_tags = []
            lbl_bios = []
            lbl_types = []
            lengths = []
            for row in batch:
                row_tokens = []
                row_pos_tags = []
                row_lbl_bios = []
                row_lbl_types = []
                for token, pos_tag, lbl_bio, lbl_type in row:
                    row_tokens.append(token)
                    row_pos_tags.append(pos_tag)
                    row_lbl_bios.append(lbl_bio)
                    row_lbl_types.append(lbl_type)
                lengths.append(len(row_tokens))
                len_padding = max_len - lengths[-1]
                row_tokens.extend([0] * len_padding)
                row_pos_tags.extend([0] * len_padding)
                row_lbl_bios.extend([0] * len_padding)
                row_lbl_types.extend([0] * len_padding)
                tokens.append(row_tokens)
                pos_tags.append(row_pos_tags)
                lbl_bios.append(row_lbl_bios)
                lbl_types.append(row_lbl_types)

            yield tokens, pos_tags, lbl_bios, lbl_types, lengths