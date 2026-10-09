# AGENTS.md — pdz-1o (tiny GPT-from-scratch, NumPy CPU)

## Selalu dari direktori proyek
Semua path relatif (`data/`, `checkpoints/`, `logs/`). Jalankan via parameter
`workdir=/root/pdz-1o`, BUKAN `cd /root/pdz-1o && cmd &` — `&` mem-background-kan
seluruh rantai termasuk `cd`, sehingga perintah lanjutan jatuh ke cwd yang salah.

## Perintah inti
- `python3 train.py --steps N --embd E --layers L --heads H --block B --batch Bs --tok {byte,bpe} [--data F] [--resume CKPT] [--lr X] [--save_dir D]`
- BPE selalu pakai `--tok bpe` (default tokenizer `data/bpe2.json`); byte pakai default.
- Eval: `python3 eval.py --ckpt F` (byte) / `python3 eval_bpe.py --ckpt F [--bpe data/bpe2.json]`.
- Infer: `python3 inference.py --tok bpe --bpe data/bpe2.json --ckpt F [--web] [--interactive]`.
- Web-learn: `python3 web_learn.py` (butuh internet; cache di `data/web_cache/`).
- Cek cepat: `python3 -c "import train, inference, eval_bpe, web_learn"`.

## Aturan yang tidak jelas dari nama file
- **Block size harus memuat 1 sampel penuh.** `train.py` mencetak `covered_by_block`;
  jika tidak 100%, model tak pernah melihat jawaban (pernah 0/200 → gibberish).
  Byte ≈ 320, BPE ≈ 128.
- **String system prompt harus identik** di `dataset_self.sample_with_think` dan
  `inference.build_prompt`. Beda satu kata = OOD = ngaco.
- **Generate wajib `stop_ids={im_end}`**, kalau tidak ekor NUL/halusinasi.
  Byte: `tok.special_stoi`; BPE: `tok.special_ids` (sudah di-alias ke
  `special_stoi`, jangan hitung manual 256+i).
- **`data/bpe2.json` stabil dan final.** Jangan latih ulang/timpa — semua checkpoint
  BPE bergantung padanya.
- **OOM trap:** graph autograd menumpuk via reference-cycle; `train.py`/`self_learn.py`
  sudah ada `del + gc.collect()` per step — jangan hapus. Batch 8/block 320/embd 96
  mati sebelum step 25 tanpanya.
- **`train(cfg, rows, resume, save_dir, tok)`**: dengan `--resume`, arsitektur CLI
  diabaikan (dimuat dari file) tapi tokenizer+data harus cocok (byte↔BPE fatal).
  Output selalu `<save_dir>/pdz-1o-final.npz` (+ `pdz-1o-stepN.npz` tiap `save_every`).

## Training lama
- Lepas dari sesi: `setsid nohup python3 -u ... > logs/X.log 2>&1 < /dev/null &`,
  simpan PID ke `logs/X.pid`. Background tool bisa mati diam-diam (pernah mati di
  step 550); `setsid` selamat. Cek: `ps -p PID -o pid,etime,cmd` + `tail logs/X.log`.
- Estimasi: 2.1M param ≈ 0.42 step/s; 5.6M ≈ 0.16 step/s (batch 8).
- Jangan harap generalisasi angka dari fine-tune LR ≥1e-3 (kolaps atraktor,
  mis. semua jawaban "46"). Grounding: LR ≤3e-4, pendek. Self-learn tanpa filter
  verifier merusak bobot (think 9/10→2/10) — yang berharga datanya, bukan bobotnya.

## Jangan hapus
`data/{big.jsonl,big16f.jsonl,seed.jsonl,web_learn.jsonl,web_index.json,bpe2.json}`,
`data/web_cache/`, `checkpoints/{pdz-1o-final,champion_bpe.npz,v3/,web3/}`.
Intermediet `pdz-1o-stepN.npz` boleh dihapus kecuali sedang resume. `logs/` kecil.

## Batas yang sudah terbukti (jangan dijanjikan ulang)
Format (ChatML/think/tool-JSON) dikuasai ~95%; grounding angka eksak lemah di
2M param. RAG generatif gagal → `--web` memakai ekstraktif + `<sumber>`.
"Setengah LLM besar" butuh GPU + ≥100M param; repo ini plafonnya model kecil CPU.
