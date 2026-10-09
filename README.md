# PDZ-1O — model + project folder `pdz-1o`

Model **dilatih dari NOL** (tanpa pretrained / model pihak ketiga), murni **NumPy di CPU**
(tanpa GPU, tanpa torch/transformers). Dataset **dibuat oleh model itu sendiri**
(bootstrapping self-synthetic).

Kemampuan:
- **Multitasking**: satu backbone untuk `chat`, `reasoning`, `tool_call`,
  `function_call`, `url_context`, `self_correct`, `novelty` (via `<|task:...|>`).
- **Belajar dari kesalahan sendiri**: loop self-learn + verifier → pasangan
  `self_correct` → fine-tune ulang (mirip STaR/ReST sederhana).
- **Memperbaiki kesalahan yang belum pernah diketahui**: koreksi generik
  (`<think>` ulang + verifier + saran tool) + fallback `<rID>` untuk token asing.
- **Menemukan hal baru**: deteksi pola jawaban benar yang belum pernah ada →
  `data/novel.jsonl` untuk dipecahkan/dilatih lagi.
- **ChatML**: `<|im_start|>role ... <|im_end|>`.
- **Tools call / function call**: `<tool_call>{"name":..., "arguments":{...}}</tool_call>`
  dieksekusi di `tools.py` (`hitung`, `waktu`, `eksekusi_kode`; tambah fungsi = tambah tool).
- **URL konteks**: tulis `<url>https://...</url>` → difetch (urllib) → disuntik
  sebagai `<tool_response>` sebelum penalaran.
- **Penalaran / thinking**: selalu `<think>...</think>` sebelum jawaban akhir.

## Struktur

```
pdz-1o/
  config.py        konfigurasi model & training
  tokenizer.py     tokenizer byte-level + ChatML (dari nol, tanpa download)
  model.py         transformer decoder-only + autograd NumPy (dari nol, CPU)
  tools.py         registry tool/function + eksekutor + fetch URL
  dataset_self.py  generator dataset mandiri (seed aturan + generate oleh model)
  train.py         training multitask CPU (block 320 = 1 sampel penuh)
  eval.py          evaluasi kuantitatif: think_fmt, tool_valid, math_acc
  self_learn.py    loop belajar-dari-kesalahan + novelty discovery
  inference.py     CLI ChatML + think + tool + url
  run.sh           alur end-to-end
  data/            seed.jsonl, self_corrections.jsonl, novel.jsonl
  checkpoints/     *.npz
```

## Cara jalan (CPU)

```bash
cd pdz-1o
pip install -r requirements.txt   # hanya numpy
bash run.sh
# atau langkah per langkah:
python3 dataset_self.py                              # 800 sampel, semua muat di block 320
python3 train.py --steps 1000 --embd 96 --layers 3 --block 320 --batch 8 --seed_n 800
python3 eval.py --ckpt checkpoints/pdz-1o-final.npz  # bukti kemajuan angka
python3 inference.py --ckpt checkpoints/pdz-1o-final.npz --prompt "hitung 7*6 pakai tool"
python3 inference.py --ckpt checkpoints/pdz-1o-final.npz --interactive
```

> Catatan jujur: ini model **kecil** (~464K param) agar muat di CPU.
> Yang dikuasai: format ChatML + `<think>` + `tool_call` JSON valid.
> Aritmetika langsung masih template-retrieval (rapuh di luar distribusi);
> untuk hitungan tepat gunakan jalur tool (`hitung`) — eksekusi Python selalu eksak.
> Untuk kualitas bahasa setara model besar tetap butuh data + compute jauh lebih besar.

## Hasil full-run CPU (10 soal, greedy, `eval.py`)

| checkpoint | think_fmt | chatml_isi | tool_valid | math_acc | loss |
|---|---|---|---|---|---|
| acak (awal) | 0/10 | 0/10 | 0/2 | 0/6 | ~6.2 |
| 100-step (pilot) | 0/10 | 10/10 | 0/2 | 0/6 | 1.61 |
| 1000-step (**champion**) | 9/10 | 9/10 | 2/2 | 3/6 | 0.03 |

Contoh nyata champion (`hitung 8*7 pakai tool`):
`<think>Perlu komputasi tepat -> panggil tool hitung.</think>`
`<tool_call>{"name": "hitung", "arguments": {"ekspresi": "5+5"}}</tool_call>`
→ dieksekusi: `{'hasil': 10}` (format benar; grounding angka masih bertahap).
Self-correct hapal benar kata-per-kata:
`Koreksi: jawaban benar 6, bukan 8. Saya belajar dari kesalahan prioritas operator.`
Eksperimen fine-tune: LR 1e-3/150-step → kolaps "46" (dibuang);
drill-copy LR 3e-4/60-step → math tetap 3/6 tapi chat regresi (dibuang).
Self-learn loop: 12 koreksi + 8 novelty terkumpul, dataset tumbuh oleh model sendiri.
Bobot pasca self-learn langsung diuji dan menurun (think 2/10 — fine-tune buta pada output
berisik merusak model kecil) sehingga **champion v1 tetap `checkpoints/champion.npz`**
(1000-step). Pelajaran: self-training butuh filter verifier ketat + LR kecil.

## Belajar otonom dari website (`web_learn.py`, `--web`)
Tanpa dataset manual: 12 situs (dummyjson ×5, w3schools ×6, python docs)
difetch + cache + ekstrak defensif (inline code, div w3-code, filter nav/footer)
-> 33 sampel ChatML + indeks kata->URL + discovery URL berantai.
`inference.py --web`: soal faktual/bingung -> retrieval (rarity+stemming,
fallback `products/search?q=`) -> grounding neuro-simbolik (jawaban faktual
disalin dari sumber + `<sumber>`, trace `<think>` jujur).
Bukti live: "carikan produk phone..." -> "Produk Apple iPhone Charger harganya
19.99. <sumber>...search?q=phone</sumber>"; "contoh for loop python" ->
`for x in fruits: print(x)` dari halaman yang tepat + sumber.
Batas jujur: grounding generatif model 2M param belum mampu (RAG 28 sampel
gagal) -> dipakai ekstraktif; model tetap yang memutuskan kapan+URL apa.

## Skala v4 (berjalan): 5.6M param, 16016 sampel
Bersih-bersih: 252M -> 40M (hapus checkpoint/file tes tak terpakai).
v4 = embd224/8-layer/8-head dari nol di `data/big16f.jsonl`, ~4.5 jam CPU.
Matematika jujur "setengah LLM besar": LLM besar ~10^10 param; v4 5.6×10^6
(masih ~2000x lebih kecil). Setengah kemampuan butuh ~10^8 param + 10^8 token
+ GPU berminggu-minggu — resep skala sudah siap di repo (`--embd/--layers`,
`dataset_big.py`), tinggal pindah ke mesin GPU.
> Cara pakai model web: `python3 inference.py --tok bpe --ckpt checkpoints/web3/pdz-1o-final.npz --web --prompt "..."`

| checkpoint | think | chatml | tool_valid | tool_exact | math | chat |
|---|---|---|---|---|---|---|
| v2-final (2500 step) | 19/22 | 21/22 | 6/6 | 1/6 | 0/10 | 3/6 |
| v2d (+drill 150) | 21/22 | 21/22 | 6/6 | 1/6 | 1/10 | 4/6 |
| **v3 (+2500 step, champion BPE)** | 21/22 | 22/22 | 6/6 | 1/6 | 1/10 | 5/6 |

Dikuasai v3: sapaan, antonim/sinonim, thinking, self-correct kata-per-kata,
`waktu` EXACT, url EXACT, semua tool JSON valid.
Belum: grounding angka eksak (atraktor "45+35") — butuh kapasitas/data lebih
besar (batas wajar model 2M param CPU). File: `checkpoints/champion_bpe.npz`,
tokenizer `data/bpe2.json`, cara pakai:
`python3 inference.py --tok bpe --bpe data/bpe2.json --ckpt checkpoints/champion_bpe.npz --prompt "..."`
