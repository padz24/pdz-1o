"""Dataset BESAR PDZ-1O v2 — ribuan sampel template (benar by-construction).
Kombinasi HELD-OUT tidak pernah dimunculkan di training (untuk uji generalisasi).
Format sama seperti v1: <|task|...> + system + user + assistant(<think> + jawaban).
"""
import random
from dataset_self import sample_with_think, save_jsonl, TASKS

SYSTEM_OK = True  # sistem memakai string yang sama via sample_with_think

# kombinasi yang DICADANGKAN untuk eval generalisasi (jangan dipakai di sini!)
HELD_OUT_MATH = {(7, "+", 5), (20, "-", 8), (6, "*", 7), (8, "*", 7), (9, "+", 6),
                 (12, "+", 15), (25, "-", 9), (11, "*", 6)}
HELD_OUT_TOOL = {(8, "*", 7), (9, "+", 6), (7, "*", 6), (12, "+", 15)}

ANTONIM = [("besar", "kecil"), ("tinggi", "rendah"), ("cepat", "lambat"),
           ("panas", "dingin"), ("terang", "gelap"), ("mahal", "murah"),
           ("kuat", "lemah"), ("rajin", "malas"), ("bersih", "kotor"),
           ("panjang", "pendek"), ("lebar", "sempit"), ("senang", "sedih")]
SINONIM = [("besar", "akbar"), ("cepat", "lekas"), ("senang", "gembira"),
           ("sedih", "duka"), ("pintar", "cerdas"), ("indah", "elok"),
           ("kuat", "tangguh"), ("bersih", "suci"), ("gelap", "gulita"),
           ("terang", "benderang"), ("murah", "hemat"), ("rajin", "tekun")]

CHAT_QA = [
    ("halo", "Halo! Saya PDZ-1O, model kecil yang dilatih dari nol di CPU."),
    ("siapa kamu?", "Saya PDZ-1O. Dilatih dari nol di CPU, bisa chat, bernalar, dan pakai tools."),
    ("halo siapa kamu?", "Halo! Saya PDZ-1O. Dilatih dari nol di CPU, bisa chat, bernalar, dan pakai tools."),
    ("jelaskan cpu vs gpu singkat", "CPU fleksibel untuk tugas umum, GPU paralel untuk matriks besar. Saya dilatih di CPU agar ringan."),
    ("apa itu multitasking?", "Multitasking: satu model menangani chat, reasoning, tool_call, url_context lewat token tugas."),
    ("bagaimana kamu belajar dari kesalahan?", "Saya mencatat kesalahan, membuat koreksi <think> ulang, lalu fine-tune dari koreksi itu."),
    ("apa itu tool_call?", "Tool_call memanggil fungsi Python lewat JSON: <tool_call>{\"name\": \"hitung\", \"arguments\": {\"ekspresi\": \"2+2\"}}</tool_call>."),
    ("apa itu thinking?", "Thinking adalah <think> bernalar dulu </think> sebelum jawaban akhir."),
    ("apa itu url konteks?", "Tulis <url>https://...</url> lalu saya ambil isinya sebagai konteks sebelum menjawab."),
    ("apa itu function call?", "Function call memanggil fungsi terdaftar seperti waktu atau eksekusi_kode lewat format tool_call."),
    ("kamu dilatih di mana?", "Saya dilatih dari nol di CPU dengan NumPy, tanpa GPU dan tanpa model pretrained."),
    ("apa itu self correct?", "Self correct: saya memeriksa jawaban sendiri, mengakui salah, lalu menulis koreksinya."),
    ("apa itu novelty?", "Novelty: saya mengusulkan pola atau soal baru yang belum ada, lalu memecahkannya."),
    ("jelaskan transformer singkat", "Transformer membaca seluruh konteks sekaligus lewat attention, lalu memprediksi token berikutnya."),
    ("apa itu token?", "Token adalah potongan teks kecil; model saya memakai sub-kata BPE yang dilatih dari nol."),
]

def gen_big(n=8000, seed=2026):
    rng = random.Random(seed)
    out = []
    def add(task, text):
        out.append({"task": task, "text": text})
    while len(out) < n:
        t = TASKS[len(out) % len(TASKS)]
        if t == "chat":
            q, a = rng.choice(CHAT_QA)
            add(t, sample_with_think(t, q, "Pengguna bertanya umum. Jawab ringkas sebagai PDZ-1O.", a))
        elif t == "reasoning":
            kind = rng.random()
            if kind < 0.45:
                op = rng.choice(["+", "-", "*"])
                if op == "+": a, b = rng.randint(1, 99), rng.randint(1, 99); h = a + b
                elif op == "-": a, b = rng.randint(10, 99), rng.randint(1, 50); h = a - b
                else: a, b = rng.randint(2, 15), rng.randint(2, 15); h = a * b
                if (a, op, b) in HELD_OUT_MATH: continue
                add(t, sample_with_think(t, f"berapa {a}{op}{b}? bernalar langkah demi langkah.",
                    f"Hitung {a}{op}{b}: hasilnya {h}. Tulis angka akhir {h}.", f"Hasilnya {h}."))
            elif kind < 0.65:
                a, b, c = rng.randint(1, 20), rng.randint(1, 20), rng.randint(1, 20)
                h = a + b * c
                add(t, sample_with_think(t, f"berapa {a}+{b}*{c}? ingat prioritas kali.",
                    f"Kali dulu: {b}*{c}={b*c}, lalu tambah {a}: {h}.", f"Hasilnya {h}."))
            elif kind < 0.8:
                a, b = rng.randint(1, 50), rng.randint(1, 50); h = a + b
                nama = rng.choice(["Andi", "Budi", "Sari", "Dewi"])
                benda = rng.choice(["apel", "kelereng", "buku", "pensil"])
                add(t, sample_with_think(t, f"{nama} punya {a} {benda}, diberi {b} lagi. Berapa total?",
                    f"Total = {a}+{b}={h}.", f"Hasilnya {h}."))
            else:
                nums = rng.sample(range(1, 60), 3)
                srt = sorted(nums)
                add(t, sample_with_think(t, f"urutkan {nums[0]},{nums[1]},{nums[2]} dari kecil.",
                    f"Bandingkan lalu urut: {srt[0]},{srt[1]},{srt[2]}.", f"Urutannya {srt[0]},{srt[1]},{srt[2]}."))
        elif t == "tool_call":
            op = rng.choice(["+", "-", "*"])
            if op == "+": a, b = rng.randint(1, 99), rng.randint(1, 99)
            elif op == "-": a, b = rng.randint(10, 99), rng.randint(1, 50)
            else: a, b = rng.randint(2, 15), rng.randint(2, 15)
            if (a, op, b) in HELD_OUT_TOOL: continue
            add(t, sample_with_think(t, f"hitung {a}{op}{b} pakai tool",
                "Perlu komputasi tepat -> panggil tool hitung.",
                f'<tool_call>{{"name": "hitung", "arguments": {{"ekspresi": "{a}{op}{b}"}}}}</tool_call>'))
        elif t == "function_call":
            if rng.random() < 0.5:
                add(t, sample_with_think(t, "panggil fungsi waktu sekarang",
                    "Butuh waktu sistem -> function waktu.",
                    '<tool_call>{"name": "waktu", "arguments": {"zona": "UTC"}}</tool_call>'))
            else:
                a, b = rng.randint(1, 30), rng.randint(1, 30)
                add(t, sample_with_think(t, f"hitung {a}+{b} dengan kode",
                    "Jalankan kode untuk hitung tepat.",
                    f'<tool_call>{{"name": "eksekusi_kode", "arguments": {{"kode": "{a}+{b}"}}}}</tool_call>'))
        elif t == "url_context":
            add(t, sample_with_think(t, "ringkas <url>https://example.com</url>",
                "Ambil konteks url dulu lalu ringkas.",
                "<tool_response>konteks url https://example.com: contoh domain</tool_response>\nRingkasan: domain contoh."))
        elif t == "self_correct":
            k = rng.random()
            if k < 0.4:
                add(t, sample_with_think(t, "2+2*2=? (jawaban awal salah: 8)",
                    "Kesalahan: mengabaikan prioritas kali. Koreksi: 2*2=4, +2=6.",
                    "Koreksi: jawaban benar 6, bukan 8. Saya belajar dari kesalahan prioritas operator."))
            elif k < 0.7:
                add(t, sample_with_think(t, "10-3*2=? (jawaban awal salah: 14)",
                    "Kesalahan: kurang dulu sebelum kali. Koreksi: 3*2=6, 10-6=4.",
                    "Koreksi: jawaban benar 4, bukan 14."))
            else:
                a, b = rng.randint(2, 30), rng.randint(2, 30)
                add(t, sample_with_think(t, f"{a}+{b}=? (jawaban awal salah: {a+b+1})",
                    f"Kesalahan: selisih 1. Koreksi: {a}+{b}={a+b}.",
                    f"Koreksi: jawaban benar {a+b}."))
        else:  # novelty + bahasa
            if rng.random() < 0.5:
                base = rng.choice([(2, 4, 8, 16), (3, 6, 12, 24), (5, 10, 20, 40), (1, 3, 6, 10), (4, 8, 16, 32)])
                add(t, sample_with_think(t, f"temukan pola baru dari {','.join(map(str, base[:-1]))},...",
                    f"Rasio konsisten. Hal baru: prediksi {base[-1]} lalu uji.",
                    f"Pola baru ditemukan. Prediksi berikutnya {base[-1]}."))
            elif rng.random() < 0.7:
                a, b = rng.choice(ANTONIM)
                add("chat", sample_with_think("chat", f"apa antonim {a}?",
                    f"Lawan kata {a} adalah {b}.", f"Antonim {a} adalah {b}."))
            else:
                a, b = rng.choice(SINONIM)
                add("chat", sample_with_think("chat", f"apa sinonim {a}?",
                    f"Persamaan kata {a} adalah {b}.", f"Sinonim {a} adalah {b}."))
    rng.shuffle(out)
    return out

if __name__ == "__main__":
    rows = gen_big(8000)
    save_jsonl(rows, "data/big.jsonl")
    from collections import Counter
    print(f"big tersimpan: {len(rows)} -> data/big.jsonl", Counter(r["task"] for r in rows))
