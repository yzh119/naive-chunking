# naive-chunking
Evaluate the embedding's performance on CONLL-2000 chunking dataset.

## Usage

Using GloVe Embedding:

    bash data/download.sh
    bash pretrained/download.sh
    python main.py
    bash eval_label.sh
    bash eval_segment.sh

## Result

|              | acc   | fb1   |
|--------------|-------|-------|
| no pretrain  | 94.69 | 91.39 |
| GloVe        | 95.67 | 92.92 |
| Senna        | 96.37 | 94.08 |


