"""Self-learning PDZ-1O: belajar dari kesalahan + temukan hal baru.
Loop:
 1. model menjawab soal (dengan <think>)
 2. verifier aturan memeriksa (matematika / format tool / konsistensi)
 3. jika SALAH -> buat pasangan koreksi (self_correct) -> latih ulang (fine-tune)
 4. jika BENAR tapi pola belum pernah ada -> simpan sebagai novelty (penemuan baru)
 5. dataset membesar OLEH MODEL ITU SENDIRI -> retrain (Rest/STaR sederhana, CPU)
Mampu "memperbaiki kesalahan yang belum pernah diketahui": koreksi generik berbasis
verifier + pola <think> ulang, bukan hafalan jawaban.
"""
import os, json, re, random
from dataset_self import save_jsonl, load_jsonl, chatml

def verify_math(question: str, answer_text: str):
    """Verifier sederhana untuk pola 'a op b'. Kembalikan (benar?, koreksi)."""
    m = re.search(r"(\d+)\s*([+\-*])\s*(\d+)", question)
    if not m:
        return None, None  # tidak bisa diverifikasi -> lewati
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    benar = a + b if op == "+" else (a - b if op == "-" else a * b)
    # cari angka dalam jawaban (di luar <think> lebih diutamakan)
    tail = answer_text.split("</think>")[-1]
    nums = re.findall(r"-?\d+", tail)
    if not nums:
        return False, f"tidak ada angka jawaban; seharusnya {benar}"
    return (int(nums[-1]) == benar), f"jawaban benar {benar}"

def verify_tool_format(answer_text: str):
    if "<tool_call>" in answer_text:
        ok = bool(re.search(r"<tool_call>\s*\{.*\"name\".*\"arguments\".*\}\s*</tool_call>", answer_text, re.S))
        return ok, ("format tool_call benar" if ok else "format tool_call rusak, contoh benar: <tool_call>{\"name\": \"hitung\", \"arguments\": {\"ekspresi\": \"2+2\"}}</tool_call>")
    return None, None

def self_learn_loop(model, tok, rounds=3, n_per_round=20, ckpt_dir="checkpoints", data_dir="data"):
    import numpy as np
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(7)
    corrections, novelties = [], []
    seen_patterns = set()
    soal_pool = [
        "berapa 3+5*2? (ingat prioritas kali)",
        "hitung 7*6 pakai tool",
        "berapa 10-3*2?",
        "jelaskan cara kamu memperbaiki kesalahan sendiri",
        "temukan pola baru dari 2,4,8 lalu prediksi berikutnya",
        "panggil fungsi waktu sekarang",
    ]
    system = "Kamu PDZ-1O. Bernalar dulu dalam <think>, lalu jawab. Gunakan tool jika perlu."
    for r in range(rounds):
        print(f"[self-learn] ronde {r+1}/{rounds}")
        for i in range(n_per_round):
            q = soal_pool[(r * n_per_round + i) % len(soal_pool)]
            task = "tool_call" if "tool" in q or "fungsi" in q else ("novelty" if "pola baru" in q else "reasoning")
            prompt = f"<|task:{task}|>" + chatml("system", system) + chatml("user", q)
            ids = tok.encode(prompt)
            try:
                stops = {tok.special_stoi["<|im_end|>"]}
                gen = model.generate_ids(ids, max_new=64, temperature=0.7, rng=rng, stop_ids=stops)
                ans = tok.decode(gen)[len(prompt):]
            except Exception as e:
                ans = f"<think>error generate: {e}</think>\nMaaf, saya gagal bernalar."
            # --- verifikasi ---
            verdicts = []
            v1, c1 = verify_math(q, ans)
            if v1 is not None:
                verdicts.append((v1, c1))
            v2, c2 = verify_tool_format(ans)
            if v2 is not None:
                verdicts.append((v2, c2))
            salah = any(v is False for v, _ in verdicts)
            # novelty: jawaban benar & pola teks belum pernah dilihat
            kunci = ans.strip()[:80]
            is_new = kunci not in seen_patterns and kunci.strip() != ""
            seen_patterns.add(kunci)
            if salah:
                kritik = "; ".join(c for v, c in verdicts if v is False)
                koreksi = (f"<|task:self_correct|>" + chatml("user", q)
                           + chatml("assistant", f"<think>Saya salah: {kritik}. "
                                   "Perbaiki langkah: uraikan ulang, cek operator, verifikasi angka akhir.</think>\n"
                                   f"Koreksi: {kritik}. Saya belajar dan tidak akan mengulangi kesalahan ini."))
                corrections.append({"task": "self_correct", "text": koreksi})
            elif is_new and len(ans.strip()) > 20:
                novelties.append({"task": "novelty",
                                  "text": f"<|task:novelty|>" + chatml("user", q) + chatml("assistant", ans)})
        print(f"  koreksi terkumpul: {len(corrections)}, novelty: {len(novelties)}")
        # --- fine-tune singkat dari koreksi (belajar dari kesalahan sendiri) ---
        if corrections:
            from config import PDZ1oConfig
            from train import encode_batch
            from model import cross_entropy, AdamW, Tensor
            import numpy as np
            import gc as _gc
            opt = AdamW(model.parameters(), lr=1e-3)
            for step in range(10):
                batch = random.sample(corrections, min(4, len(corrections)))
                X, Y, _ = encode_batch(batch, tok, model.block_size)
                mask = (Y != -100)
                Yc = np.where(mask, Y, 0)
                logits = model.forward(X)
                B, T, V = logits.data.shape
                lv = logits.reshape(B * T, V)
                sel_data = lv.data[mask.reshape(-1)]
                sel = Tensor(sel_data, (lv,), "gather2")
                def make_bw(lv=lv, m=mask.reshape(-1)):
                    def _bw():
                        if lv.requires_grad:
                            g = np.zeros_like(lv.data)
                            g[m] += sel.grad
                            lv.grad = g if lv.grad is None else lv.grad + g
                    return _bw
                sel._backward = make_bw()
                loss = cross_entropy(sel, Yc.reshape(-1)[mask.reshape(-1)])
                opt.zero_grad(); loss.backward(); opt.step()
                lval = float(loss.data)
                del loss, sel, lv, logits, X, Y, Yc, mask
                _gc.collect()
            print(f"  fine-tune koreksi selesai.")
    save_jsonl(corrections, os.path.join(data_dir, "self_corrections.jsonl"))
    save_jsonl(novelties, os.path.join(data_dir, "novel.jsonl"))
    model.save(os.path.join(ckpt_dir, "pdz-1o-selflearn.npz"))
    print(f"[self-learn] selesai. koreksi={len(corrections)} novelty={len(novelties)}")
    print(f"  dataset membesar oleh model sendiri -> {data_dir}/self_corrections.jsonl + novel.jsonl")
    return corrections, novelties
