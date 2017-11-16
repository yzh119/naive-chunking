#!/usr/bin/env bash
if [ ! -f $1/glove.6B.zip ] && [ ! -f $1/glove.6B.50d.txt ]; then
    wget http://nlp.stanford.edu/data/glove.6B.zip -P $1
fi
if [ ! -f $1/glove.6B.50d.txt ]; then
    unzip -u $1/glove.6B.zip
fi
ln -s $1/glove.6B.50d.txt .