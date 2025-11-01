#!/bin/bash
python src/data_preprocess.py --src train.tags.en-de.en --tgt train.tags.en-de.de --out_dir data
python src/train.py --data_dir data --epochs 5 --batch_size 32
