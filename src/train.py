import argparse, os, time, math, json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn, optim
from model import Seq2SeqTransformer
from data import TranslationDataset, collate_fn
from utils import set_seed, save_checkpoint, load_vocab, idxs_to_sentence

try:
    import sacrebleu
    HAVE_SACREBLEU = True
except:
    HAVE_SACREBLEU = False

def compute_bleu(references, candidates):
    if HAVE_SACREBLEU:
        return sacrebleu.corpus_bleu(candidates, [references]).score
    else:
        # fallback: simple n-gram precision (very rough)
        return 0.0

def train_epoch(model, dataloader, optimizer, device, pad_idx=0):
    model.train()
    total_loss = 0.0
    crit = nn.CrossEntropyLoss(ignore_index=pad_idx)
    for batch in dataloader:
        src = batch['src'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        optimizer.zero_grad()
        logits, *_ = model(src, tgt_input, src_pad_idx=pad_idx, tgt_pad_idx=pad_idx)
        # logits: (B, T, V)
        loss = crit(logits.view(-1, logits.size(-1)), tgt_output.contiguous().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, itos, pad_idx=0):
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=pad_idx)
    total_loss = 0.0
    refs = []
    hyps = []
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            logits, *_ = model(src, tgt_input, src_pad_idx=pad_idx, tgt_pad_idx=pad_idx)
            loss = crit(logits.view(-1, logits.size(-1)), tgt_output.contiguous().view(-1))
            total_loss += loss.item()
            # greedy decode for BLEU (simple)
            preds = logits.argmax(dim=-1).cpu().tolist()
            for p in preds:
                hyps.append(idxs_to_sentence(p, itos))
            # refs: use tgt_output (skip pad and stop at eos)
            for r in tgt_output.cpu().tolist():
                refs.append(idxs_to_sentence(r, itos))
    bleu = compute_bleu(refs, hyps)
    return total_loss / len(dataloader), bleu

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data', help='dir with train.json val.json and vocab')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    set_seed(42)
    device = args.device
    # load vocab and datasets
    stoi = load_vocab(os.path.join(args.data_dir, 'vocab_stoi.json'))
    itos = load_vocab(os.path.join(args.data_dir, 'vocab_itos.json'))
    vocab_size = len(stoi)
    train_ds = TranslationDataset(os.path.join(args.data_dir, 'train.json'))
    val_ds = TranslationDataset(os.path.join(args.data_dir, 'val.json'))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn)
    model = Seq2SeqTransformer(vocab_size, vocab_size, d_model=args.d_model, N=args.layers, heads=args.heads, d_ff=args.d_ff).to(device)
    print('Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    best_bleu = 0.0
    train_log = []
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, bleu = evaluate(model, val_loader, device, itos)
        t1 = time.time()
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, BLEU={bleu:.2f}, time={t1-t0:.1f}s")
        train_log.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'bleu': bleu, 'time': round(t1-t0, 1)})
        if bleu > best_bleu:
            best_bleu = bleu
            save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(args.data_dir, 'best.pth'))
    log_path = os.path.join(args.data_dir, 'train_log.csv')
    pd.DataFrame(train_log).to_csv(log_path, index=False)
    print(f"Saved training log to {log_path}")
    #绘制loss曲线
    plt.figure()
    plt.plot([x['epoch'] for x in train_log], [x['train_loss'] for x in train_log], label='Train Loss')
    plt.plot([x['epoch'] for x in train_log], [x['val_loss'] for x in train_log], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.data_dir, 'loss_curve.png'))
    #绘制bleu曲线
    plt.figure()
    plt.plot([x['epoch'] for x in train_log], [x['bleu'] for x in train_log], 'o-', color='orange', label='BLEU')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.title('BLEU vs Epoch')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.data_dir, 'bleu_curve.png'))
    print("Saved loss_curve.png and bleu_curve.png")