#!/bin/bash
# Alur end-to-end PDZ-1O di CPU (full: block 320 agar 1 sampel penuh terlihat model)
set -e
cd "$(dirname "$0")"
echo "== 1. dataset seed (aturan lokal, lalu self-synthetic) =="
python3 dataset_self.py
echo "== 2. training dari NOL di CPU (multitask) =="
python3 train.py --steps 1000 --embd 96 --layers 3 --block 320 --batch 8 --seed_n 800
echo "== 2b. evaluasi kemajuan =="
python3 eval.py --ckpt checkpoints/pdz-1o-final.npz
echo "== 3. self-learning: belajar dari kesalahan + novelty =="
python3 -c "
from tokenizer import PDZ1oTokenizer
from model import PDZ1oModel
from self_learn import self_learn_loop
tok = PDZ1oTokenizer()
model = PDZ1oModel.load('checkpoints/pdz-1o-final.npz')
self_learn_loop(model, tok, rounds=2, n_per_round=10)
"
echo "== 4. demo inference (chatml + think + tool + url) =="
python3 inference.py --ckpt checkpoints/pdz-1o-selflearn.npz --prompt "hitung 7*6 pakai tool dan bernalar langkah demi langkah" || \
python3 inference.py --ckpt checkpoints/pdz-1o-final.npz --prompt "hitung 7*6 pakai tool dan bernalar langkah demi langkah"
