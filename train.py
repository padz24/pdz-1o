"""Training PDZ-1O di CPU — multitask, dari nol.
- Batch diambil seimbang antar task (multitasking).
- CPU only (numpy). Atur n_embd/block_size kecil jika RAM/CPU pas-pasan.
"""
import os, json, random, gc
import numpy as np
from config import PDZ1oConfig
from tokenizer import PDZ1oTokenizer
from model import PDZ1oModel, cross_entropy, AdamW
from dataset_self import seed_samples

def encode_batch(rows, tok, block_size):
    xs, ys, tasks = [], [], []
    for r in rows:
        ids = tok.encode(r["text"])[:block_size + 1]
        if len(ids) < 3:
            continue
        # LM shift: x=ids[:-1], y=ids[1:], pad ke block_size
        x = ids[:-1][:block_size]
        y = ids[1:][:block_size]
        pad = block_size - len(x)
        if pad > 0:
            # pad dengan id <eos>? gunakan 0 aman (byte 0) + mask via -100? sederhana: pad 0 dan mask
            x = x + [0] * pad
            y = y + [-100] + [0] * (pad - 1) if pad >= 1 else y
        xs.append(x); ys.append(y); tasks.append(r.get("task", "chat"))
    return np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64), tasks

def train(cfg: PDZ1oConfig, data_rows, save_dir="checkpoints", resume="", tok=None):
    os.makedirs(save_dir, exist_ok=True)
    if tok is None:
        from tokenizer import PDZ1oTokenizer
        tok = PDZ1oTokenizer()
    if resume and os.path.exists(resume):
        model = PDZ1oModel.load(resume)
        print(f"[pdz-1o] resume dari {resume}")
    else:
        model = PDZ1oModel(vocab_size=tok.vocab_size, n_layer=cfg.n_layer,
                           n_head=cfg.n_head, n_embd=cfg.n_embd,
                           block_size=cfg.block_size, seed=cfg.seed)
    opt = AdamW(model.parameters(), lr=cfg.lr, b1=cfg.betas[0], b2=cfg.betas[1], eps=cfg.eps)
    rng = random.Random(cfg.seed)
    # kelompokkan per task untuk sampling seimbang (multitasking)
    by_task = {}
    for r in data_rows:
        by_task.setdefault(r.get("task", "chat"), []).append(r)
    tasks = list(by_task.keys())
    print(f"[pdz-1o] tasks: {tasks} | params: {sum(p.data.size for p in model.parameters())} | CPU numpy", flush=True)
    for step in range(1, cfg.max_steps + 1):
        # sample seimbang: ambil 1 sampel per task round-robin hingga batch penuh
        batch = []
        for i in range(cfg.batch_size):
            t = tasks[(step + i) % len(tasks)]
            batch.append(rng.choice(by_task[t]))
        X, Y, _ = encode_batch(batch, tok, cfg.block_size)
        # mask pad (-100)
        mask = (Y != -100)
        Yc = np.where(mask, Y, 0)
        logits = model.forward(X)  # (B,T,V)
        B, T, V = logits.data.shape
        # flatten valid positions
        lv = logits.reshape(B * T, V)
        yv = Yc.reshape(-1)
        mv = mask.reshape(-1)
        idx = np.where(mv)[0]
        if len(idx) == 0:
            continue
        # gather valid: buat node baru via operasi yang mendukung grad?
        # Sederhana: hitung loss hanya pada posisi valid dengan slice manual + grad scatter.
        # Implementasi: loop kecil (B*T <= ~1024, murah di CPU).
        from model import Tensor
        # ambil logits valid sebagai view ber-grad:
        # gunakan trick: logits_valid = sum_i onehot(i)*logits_flat[i] -> gunakan embedding-like gather
        # Di sini implementasi gather ber-grad eksplisit:
        flat = lv  # Tensor (N,V)
        sel_data = flat.data[idx]  # (M,V)
        sel = Tensor(sel_data, (flat,), "gather")
        def make_bw(flat=flat, idx=idx):
            def _bw():
                if flat.requires_grad:
                    g = np.zeros_like(flat.data)
                    g[idx] += sel.grad
                    flat.grad = g if flat.grad is None else flat.grad + g
            return _bw
        sel._backward = make_bw()
        loss = cross_entropy(sel, yv[idx])
        opt.zero_grad()
        loss.backward()
        # clip grad
        for p in model.parameters():
            if p.grad is not None:
                p.grad = np.clip(p.grad, -1.0, 1.0)
        opt.step()
        lval = float(loss.data)
        # bebaskan graph autograd (hindari siklus referensi menumpuk -> OOM)
        del loss, sel, lv, logits, X, Y, Yc, yv, mv, idx, mask
        gc.collect()
        if step % cfg.log_every == 0 or step == 1:
            print(f"step {step}/{cfg.max_steps} loss={lval:.4f}", flush=True)
        if step % cfg.save_every == 0:
            p = os.path.join(save_dir, f"pdz-1o-step{step}.npz")
            model.save(p)
            print(f"  saved {p}", flush=True)
    final = os.path.join(save_dir, "pdz-1o-final.npz")
    model.save(final)
    print(f"[pdz-1o] selesai -> {final}")
    return model, tok

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--embd", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--block", type=int, default=320)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--seed_n", type=int, default=800)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--tok", type=str, default="byte", choices=["byte", "bpe"])
    ap.add_argument("--bpe_path", type=str, default="data/bpe2.json")
    a = ap.parse_args()
    from config import PDZ1oConfig as _C
    if a.tok == "bpe":
        from tokenizer_bpe import BPETokenizer as _T
        _tok0 = _T.load(a.bpe_path)
    else:
        from tokenizer import PDZ1oTokenizer as _T
        _tok0 = _T()
    import numpy as _np
    cfg = _C(vocab_size=_tok0.vocab_size, n_embd=a.embd, n_layer=a.layers,
             n_head=a.heads, block_size=a.block, batch_size=a.batch,
             max_steps=a.steps, lr=a.lr)
    if a.data and os.path.exists(a.data):
        from dataset_self import load_jsonl
        rows = load_jsonl(a.data)
    elif a.tok == "bpe":
        from dataset_big import gen_big
        rows = gen_big(8000)
    else:
        rows = seed_samples(a.seed_n)
    # peringatan truncasi: pastikan block muat sampel penuh
    _tok = _tok0
    _lens = [len(_tok.encode(r["text"])) for r in rows]
    _cover = sum(1 for l in _lens if l <= a.block + 1)
    print(f"[data] n={len(rows)} len min/max/mean={min(_lens)}/{max(_lens)}/{sum(_lens)/len(_lens):.1f} covered_by_block={_cover}/{len(rows)}")
    train(cfg, rows, resume=a.resume, save_dir=a.save_dir, tok=_tok0)
