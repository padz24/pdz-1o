"""Dataset PDZ-1O dibuat OLEH MODEL ITU SENDIRI (self-synthetic / bootstrapping).
Tahap 0: seed aturan (tanpa data luar) -> model belajar format.
Tahap 1+: model generate sampel baru -> verifikasi aturan -> simpan yang benar.
Multitasking: chat, reasoning/thinking, tool_call, function_call, url_context,
              self_correct (belajar dari kesalahan), novelty (hal baru).
Format ChatML + <think>.
"""
import json, random, os

TASKS = ["chat", "reasoning", "tool_call", "function_call", "url_context", "self_correct", "novelty"]

def chatml(role, content):
    return f"<|im_start|>{role}\n{content}<|im_end|>\n"

def sample_with_think(task, user_q, think, answer):
    return (f"<|task:{task}|>"
            + chatml("system", "Kamu PDZ-1O. Bernalar dulu dalam <think>, lalu jawab. Gunakan tool jika perlu.")
            + chatml("user", user_q)
            + chatml("assistant", f"<think>{think}</think>\n{answer}"))

# ---------- seed deterministik (aturan, bukan scraping) ----------
def seed_samples(n=120):
    rng = random.Random(1337)
    out = []
    chat_qa = [
        ("halo", "Halo! Saya PDZ-1O, model kecil yang dilatih dari nol di CPU."),
        ("siapa kamu?", "Saya PDZ-1O. Dilatih dari nol di CPU, bisa chat, bernalar, dan pakai tools."),
        ("jelaskan cpu vs gpu singkat", "CPU fleksibel untuk tugas umum, GPU paralel untuk matriks besar. Saya dilatih di CPU agar ringan."),
        ("apa itu multitasking?", "Multitasking: satu model menangani chat, reasoning, tool_call, url_context lewat token tugas."),
        ("bagaimana kamu belajar dari kesalahan?", "Saya mencatat kesalahan, membuat koreksi <think> ulang, lalu fine-tune dari koreksi itu."),
        ("apa itu tool_call?", "Tool_call memanggil fungsi Python lewat JSON: <tool_call>{\"name\": \"hitung\", \"arguments\": {\"ekspresi\": \"2+2\"}}</tool_call>."),
        ("apa itu thinking?", "Thinking adalah <think> bernalar dulu </think> sebelum jawaban akhir."),
        ("halo siapa kamu?", "Halo! Saya PDZ-1O. Dilatih dari nol di CPU, bisa chat, bernalar, dan pakai tools."),
        ("halo, jelaskan kemampuanmu", "Halo! Saya PDZ-1O: chat, reasoning dengan <think>, tool_call JSON, dan konteks URL."),
        ("siapa kamu? jelaskan kemampuanmu", "Saya PDZ-1O. Dilatih dari nol di CPU, bisa chat, bernalar, dan pakai tools."),
    ]
    for i in range(n):
        t = TASKS[i % len(TASKS)]
        if t == "chat":
            q, a = rng.choice(chat_qa)
            out.append({"task": t, "text": sample_with_think(
                t, q, "Pengguna bertanya umum. Jawab ringkas sebagai PDZ-1O.",
                a)})
        elif t == "reasoning":
            op = rng.choice(["+", "+", "+", "-", "*"])
            if op == "+":
                a, b = rng.randint(1, 50), rng.randint(1, 50)
                h = a + b
            elif op == "-":
                a, b = rng.randint(10, 60), rng.randint(1, 30)
                h = a - b
            else:
                a, b = rng.randint(2, 12), rng.randint(2, 12)
                h = a * b
            out.append({"task": t, "text": sample_with_think(
                t, f"berapa {a}{op}{b}? bernalar langkah demi langkah.",
                f"Hitung {a}{op}{b}: hasilnya {h}. Tulis angka akhir {h}.",
                f"Hasilnya {h}.")})
        elif t == "tool_call":
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            op = rng.choice(["*", "+", "-"])
            out.append({"task": t, "text": sample_with_think(
                t, f"hitung {a}{op}{b} pakai tool",
                "Perlu komputasi tepat -> panggil tool hitung.",
                f'<tool_call>{{"name": "hitung", "arguments": {{"ekspresi": "{a}{op}{b}"}}}}</tool_call>')})
        elif t == "function_call":
            out.append({"task": t, "text": sample_with_think(
                t, "panggil fungsi waktu sekarang",
                "Butuh waktu sistem -> function waktu.",
                '<tool_call>{"name": "waktu", "arguments": {"zona": "UTC"}}</tool_call>')})
        elif t == "url_context":
            out.append({"task": t, "text": sample_with_think(
                t, "ringkas <url>https://example.com</url>",
                "Ambil konteks url dulu lalu ringkas.",
                "<tool_response>konteks url https://example.com: contoh domain</tool_response>\nRingkasan: domain contoh.")})
        elif t == "self_correct":
            kind = rng.choice([0, 1, 2])
            if kind == 0:
                out.append({"task": t, "text": sample_with_think(
                    t, "2+2*2=? (jawaban awal salah: 8)",
                    "Kesalahan: mengabaikan prioritas kali. Koreksi: 2*2=4, +2=6.",
                    "Koreksi: jawaban benar 6, bukan 8. Saya belajar dari kesalahan prioritas operator.")})
            elif kind == 1:
                out.append({"task": t, "text": sample_with_think(
                    t, "10-3*2=? (jawaban awal salah: 14)",
                    "Kesalahan: kurang dulu sebelum kali. Koreksi: 3*2=6, 10-6=4.",
                    "Koreksi: jawaban benar 4, bukan 14.")})
            else:
                a, b = rng.randint(2, 9), rng.randint(2, 9)
                out.append({"task": t, "text": sample_with_think(
                    t, f"{a}+{b}=? (jawaban awal salah: {a+b+1})",
                    f"Kesalahan: selisih 1. Koreksi: {a}+{b}={a+b}.",
                    f"Koreksi: jawaban benar {a+b}.")})
        else:  # novelty
            base = rng.choice([(2, 4, 8, 16), (3, 6, 12, 24), (5, 10, 20, 40), (1, 3, 6, 10)])
            out.append({"task": t, "text": sample_with_think(
                t, f"temukan pola baru dari {','.join(map(str, base[:-1]))},...",
                f"Rasio/selisih konsisten. Hal baru: prediksi {base[-1]} lalu uji.",
                f"Pola baru ditemukan. Prediksi berikutnya {base[-1]}.")})
    return out

# ---------- generator oleh model (setelah punya checkpoint) ----------
def model_generate_samples(model, tok, n=50, max_new=64, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    prompts = [
        f"<|task:{t}|>" + chatml("user", q)
        for t, q in [
            ("reasoning", "selesaikan 3+5*2 dengan bernalar"),
            ("tool_call", "hitung 7*6 pakai tool"),
            ("chat", "jelaskan cara kamu belajar dari kesalahan"),
            ("self_correct", "perbaiki: 10-3*2=14?"),
            ("novelty", "ajukan soal baru yang belum ada lalu jawab"),
        ]
    ]
    out = []
    for i in range(n):
        p = prompts[i % len(prompts)]
        ids = tok.encode(p)
        try:
            gen = model.generate_ids(ids, max_new=max_new, temperature=0.8, rng=rng)
            txt = tok.decode(gen)
        except Exception as e:
            txt = p + f"<think>gagal generate: {e}</think>"
        # verifikasi ringan: harus ada <think> agar disimpan
        if "<think>" in txt and "</think>" in txt:
            # tentukan task dari prompt
            task = "chat"
            for t in TASKS:
                if f"<|task:{t}|>" in p:
                    task = t
                    break
            out.append({"task": task, "text": txt, "self_made": True})
    return out

def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

if __name__ == "__main__":
    rows = seed_samples(800)
    save_jsonl(rows, "data/seed.jsonl")
    print(f"seed tersimpan: {len(rows)} -> data/seed.jsonl")
