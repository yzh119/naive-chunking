# naive-chunking
Evaluate the embedding's performance on CONLL-2000 chunking dataset.


## Usage

Using GLoVe Embedding:

    bash data/download.sh
    bash pretrained/download.sh
    python main.py
    bash eval_label.sh
    bash eval_segment.sh

## Result

Without pretrained embedding:

|              | acc   | fb1   |
|--------------|-------|-------|
| segmentation | 97.11 | 94.64 |
| labeling     | 95.46 | 92.50 |

Using GLoVe Embedding:

Using Senna Embedding:

