import re, os, json, argparse, xml.etree.ElementTree as ET
from collections import Counter

def clean_line(line):
    line = line.strip()
    line = re.sub(r'<[^>]+>', '', line)
    line = re.sub(r'\s+', ' ', line)
    return line.strip()

def read_plain(src_path, tgt_path):
    #读取训练集平行文本文件（去标签、去空行）
    src_lines, tgt_lines = [], []
    with open(src_path, 'r', encoding='utf-8') as f:
        for l in f:
            text = clean_line(l)
            if text:
                src_lines.append(text)
    with open(tgt_path, 'r', encoding='utf-8') as f:
        for l in f:
            text = clean_line(l)
            if text:
                tgt_lines.append(text)
    n = min(len(src_lines), len(tgt_lines))
    return src_lines[:n], tgt_lines[:n]

def read_xml_pair(en_xml, de_xml):
    #读取xml文件对，提取<seg>标签内的平行句子
    def extract(xml_file):
        root = ET.parse(xml_file).getroot()
        segs = [clean_line(seg.text or '') for seg in root.iter('seg') if seg.text]
        return segs
    en_lines = extract(en_xml)
    de_lines = extract(de_xml)
    n = min(len(en_lines), len(de_lines))
    return en_lines[:n], de_lines[:n]

def build_vocab(lines, vocab_size=30000, min_freq=2, special_tokens=None):
    #根据训练文本构建词表
    if special_tokens is None:
        special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
    counter = Counter()
    for s in lines:
        tokens = s.strip().split()
        counter.update(tokens)
    vocab_tokens = [tok for tok, freq in counter.most_common(vocab_size) if freq >= min_freq]
    stoi = {tok: i+len(special_tokens) for i, tok in enumerate(vocab_tokens)}
    for i, tok in enumerate(special_tokens):
        stoi[tok] = i
    itos = {i: tok for tok, i in stoi.items()}
    return stoi, itos

def tokenize_lines(lines, stoi, bos='<bos>', eos='<eos>', unk='<unk>'):
    #将句子转为token id序列
    tokenized = []
    for s in lines:
        toks = s.strip().split()
        ids = [stoi.get(t, stoi.get(unk)) for t in toks]
        ids = [stoi[bos]] + ids + [stoi[eos]]
        tokenized.append(ids)
    return tokenized

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def process_xml_dir(xml_dir, stoi, out_dir):
    #处理验证集与测试集
    xml_files = sorted([f for f in os.listdir(xml_dir) if f.endswith('.xml')])
    paired = {}
    for f in xml_files:
        base = f.replace('.en.xml','').replace('.de.xml','')
        paired.setdefault(base, {'en': None, 'de': None})
        if f.endswith('.en.xml'):
            paired[base]['en'] = os.path.join(xml_dir, f)
        elif f.endswith('.de.xml'):
            paired[base]['de'] = os.path.join(xml_dir, f)
    for name, paths in paired.items():
        if not paths['en'] or not paths['de']:
            continue
        src_lines, tgt_lines = read_xml_pair(paths['en'], paths['de'])
        tokenized_src = tokenize_lines(src_lines, stoi)
        tokenized_tgt = tokenize_lines(tgt_lines, stoi)
        dataset = {'src': tokenized_src, 'tgt': tokenized_tgt}

        if "dev2010" in name:
            save_path = os.path.join(out_dir, 'val.json')
            print(f"Processed validation set: {name} ({len(src_lines)} sentences)")
        else:
            save_path = os.path.join(out_dir, f"{name}.json")
            print(f"Processed test set: {name} ({len(src_lines)} sentences)")

        save_json(dataset, save_path)

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    # Step 1: 使用训练集构建词表
    src_lines, tgt_lines = read_plain(args.train_src, args.train_tgt)
    all_lines = src_lines + tgt_lines
    stoi, itos = build_vocab(all_lines, vocab_size=args.vocab_size, min_freq=args.min_freq)
    save_json(stoi, os.path.join(args.out_dir, 'vocab_stoi.json'))
    save_json(itos, os.path.join(args.out_dir, 'vocab_itos.json'))
    # Step 2: 训练集token化
    tokenized_src = tokenize_lines(src_lines, stoi)
    tokenized_tgt = tokenize_lines(tgt_lines, stoi)
    train = {'src': tokenized_src, 'tgt': tokenized_tgt}
    save_json(train, os.path.join(args.out_dir, 'train.json'))
    print(f"Saved train.json with {len(train['src'])} samples.")

    #Step 3: 处理xml的dev/test
    if args.xml_dir and os.path.isdir(args.xml_dir):
        process_xml_dir(args.xml_dir, stoi, args.out_dir)
    else:
        print("No XML directory found or path invalid; skipped dev/test processing.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_src', required=True, help='train src .en file')
    parser.add_argument('--train_tgt', required=True, help='train tgt .de file')
    parser.add_argument('--xml_dir', help='directory containing IWSLT XML test/dev files')
    parser.add_argument('--out_dir', default='dataset', help='output dir')
    parser.add_argument('--vocab_size', type=int, default=30000)
    parser.add_argument('--min_freq', type=int, default=2)
    args = parser.parse_args()
    main(args)