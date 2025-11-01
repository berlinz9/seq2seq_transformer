# Seq2Seq Transformer (Encoder-Decoder) for IWSLT en-de

Structure:
- src/model.py         : Encoder-Decoder Transformer implementation (pure PyTorch)
- src/data_preprocess.py : Clean parallel files, build vocab, produce train/val json
- src/data.py          : Dataset and collate_fn
- src/train.py         : Training + evaluation (BLEU if sacrebleu installed)
- scripts/run.sh       : Example run script
- requirements.txt     : python requirements

Usage (example):
1. Put your `train.tags.en-de.en` and `train.tags.en-de.de` files in the project root.
2. Install dependencies: `pip install -r requirements.txt`
3. Run preprocessing + train: `bash scripts/run.sh`

Notes:
- data_preprocess.py uses a simple whitespace tokenizer by default and builds a shared vocab.
- You can adapt tokenizer to SentencePiece for better performance.
