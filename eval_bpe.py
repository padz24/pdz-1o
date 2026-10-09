"""Eval PDZ-1O v2 (BPE) — termasuk HELD-OUT yang tak pernah dilatih.
Metrik: think_fmt, chatml, tool_valid, tool_exact (ekspresi == soal),
        math_acc, chat_exact (antonim/sinonim). Greedy deterministik.
"""
import re, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from tokenizer_bpe import BPETokenizer
from model import PDZ1oModel
from tools import parse_tool_calls, REGISTRY

SYS = "Kamu PDZ-1O. Bernalar dulu dalam <think>, lalu jawab. Gunakan tool jika perlu."

def prompt(tok, task, user):
    return (f"<|task:{task}|>" + tok.chatml("system", SYS) + tok.chatml("user", user))

def gen(model, tok, task, user, max_new=80):
    p = prompt(tok, task, user)
    ids = tok.encode(p)
    if len(ids) > model.block_size:
        ids = ids[-model.block_size:]
    stops = {tok.special_ids["<|im_end|>"]}
    out = tok.decode(model.generate_ids(ids, max_new=max_new, temperature=0, stop_ids=stops))
    return out[len(p):]

def num(tail, keys=("jawaban benar", "Hasilnya")):
    for k in keys:
        m = re.search(k + r"\s*(-?\d+)", tail)
        if m:
            return int(m.group(1))
    ns = re.findall(r"-?\d+", tail.split("</think>")[-1])
    return int(ns[-1]) if ns else None

# (task, soal, ekspektasi) — ekspektasi: angka / ekspresi tool / substring jawaban
TESTS = [
    ("reasoning", "berapa 33+27? bernalar langkah demi langkah.", ("num", 60)),
    ("reasoning", "berapa 50-17? bernalar langkah demi langkah.", ("num", 33)),
    ("reasoning", "berapa 9*8? bernalar langkah demi langkah.", ("num", 72)),
    ("reasoning", "berapa 7+5? bernalar langkah demi langkah.", ("num", 12)),       # HELD-OUT
    ("reasoning", "berapa 20-8? bernalar langkah demi langkah.", ("num", 12)),      # HELD-OUT
    ("reasoning", "berapa 6*7? bernalar langkah demi langkah.", ("num", 42)),      # HELD-OUT
    ("reasoning", "berapa 12+15? bernalar langkah demi langkah.", ("num", 27)),    # HELD-OUT
    ("reasoning", "berapa 4+9*3? ingat prioritas kali.", ("num", 31)),
    ("reasoning", "Rina punya 23 buku, diberi 19 lagi. Berapa total?", ("num", 42)),
    ("reasoning", "urutkan 9,3,7 dari kecil.", ("str", "3,7,9")),
    ("tool_call", "hitung 13+21 pakai tool", ("expr", "13+21")),
    ("tool_call", "hitung 8*7 pakai tool", ("expr", "8*7")),                        # HELD-OUT
    ("tool_call", "hitung 9+6 pakai tool", ("expr", "9+6")),                        # HELD-OUT
    ("tool_call", "hitung 7*6 pakai tool", ("expr", "7*6")),                        # HELD-OUT
    ("tool_call", "hitung 12+15 pakai tool", ("expr", "12+15")),                    # HELD-OUT
    ("chat", "halo siapa kamu?", ("str", "PDZ-1O")),
    ("chat", "apa antonim besar?", ("str", "kecil")),
    ("chat", "apa sinonim cepat?", ("str", "lekas")),
    ("chat", "apa itu thinking?", ("str", "think")),
    ("self_correct", "2+2*2=? (jawaban awal salah: 8)", ("num", 6)),
    ("function_call", "panggil fungsi waktu sekarang", ("tool", "waktu")),
    ("url_context", "ringkas <url>https://example.com</url>", ("str", "contoh")),
]

def evaluate(ckpt, bpe_path="data/bpe2.json"):
    tok = BPETokenizer.load(bpe_path)
    model = PDZ1oModel.load(ckpt)
    print(f"[eval-v2] {ckpt} params={sum(p.data.size for p in model.parameters())}")
    s = {"think": 0, "chatml": 0, "tool_valid": 0, "tool_exact": 0,
         "math_ok": 0, "math_tot": 0, "chat_ok": 0, "chat_tot": 0, "n_tool": 0}
    for task, q, exp in TESTS:
        out = gen(model, tok, task, q)
        kind, want = exp
        think = "<think>" in out and "</think>" in out
        s["think"] += think
        s["chatml"] += ("<|im_start|>assistant" in out)
        mark = ""
        if kind == "num":
            s["math_tot"] += 1
            got = num(out)
            ok = got == want
            s["math_ok"] += ok
            mark = "BENAR" if ok else f"SALAH(got={got})"
        elif kind == "expr":
            s["n_tool"] += 1
            calls = [c for c in parse_tool_calls(out) if "_parse_error" not in c and c.get("name") in REGISTRY]
            s["tool_valid"] += bool(calls)
            ok = any(c.get("arguments", {}).get("ekspresi") == want for c in calls)
            s["tool_exact"] += ok
            mark = "EXACT" if ok else ("valid-tapi-salah-ekspresi" if calls else "TIDAK-VALID")
        elif kind == "tool":
            s["n_tool"] += 1
            calls = [c for c in parse_tool_calls(out) if "_parse_error" not in c and c.get("name") in REGISTRY]
            s["tool_valid"] += bool(calls)
            ok = any(c.get("name") == want for c in calls)
            s["tool_exact"] += ok
            mark = "EXACT" if ok else "SALAH"
        else:
            s["chat_tot"] += 1
            ok = want.lower() in out.lower()
            s["chat_ok"] += ok
            mark = "BENAR" if ok else "SALAH"
        print(f"  [{task}] {q!r}\n    -> {out[:200]!r} think={think} {mark}")
    print(f"\n[hasil-v2] think={s['think']}/{len(TESTS)} chatml={s['chatml']}/{len(TESTS)} "
          f"tool_valid={s['tool_valid']}/{s['n_tool']} tool_exact={s['tool_exact']}/{s['n_tool']} "
          f"math={s['math_ok']}/{s['math_tot']} chat={s['chat_ok']}/{s['chat_tot']}")
    return s

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v2-final.npz")
    ap.add_argument("--bpe", default="data/bpe2.json")
    a = ap.parse_args()
    evaluate(a.ckpt, a.bpe)
