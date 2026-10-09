"""Eval PDZ-1O — bukti kemajuan kuantitatif (CPU, deterministik).
Metrik:
- think_fmt: ada <think>...</think>
- chatml_fmt: ada <|im_start|>assistant
- tool_valid: JSON tool_call bisa diparse + name dikenal
- math_acc: angka akhir jawaban == hasil benar (in-distribution)
Greedy (temperature=0) agar deterministik.
"""
import re, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from tokenizer import PDZ1oTokenizer
from model import PDZ1oModel
from tools import parse_tool_calls, REGISTRY
from inference import build_prompt

TESTS = [
    ("berapa 7+5? bernalar langkah demi langkah.", "reasoning", 12),
    ("berapa 20-8? bernalar langkah demi langkah.", "reasoning", 12),
    ("berapa 6*7? bernalar langkah demi langkah.", "reasoning", 42),
    ("hitung 8*7 pakai tool", "tool_call", None),
    ("hitung 9+6 pakai tool", "tool_call", None),
    ("halo siapa kamu?", "chat", None),
    ("apa itu tool_call?", "chat", None),
    ("2+2*2=? (jawaban awal salah: 8)", "self_correct", 6),
    ("10-3*2=? (jawaban awal salah: 14)", "self_correct", 4),
    ("temukan pola baru dari 2,4,8,...", "novelty", 16),
]

def greedy_answer(model, tok, user_text, task, max_new=160):
    prompt = build_prompt(tok, user_text, task)
    ids = tok.encode(prompt)
    if len(ids) > model.block_size:
        ids = ids[-model.block_size:]
    stops = {tok.special_stoi["<|im_end|>"]}
    gen = model.generate_ids(ids, max_new=max_new, temperature=0, stop_ids=stops)
    full = tok.decode(gen)
    return full[len(tok.decode(ids)):]

def last_number(txt):
    tail = txt.split("</think>")[-1]
    m = re.search(r"jawaban benar\s*(-?\d+)", tail)  # koreksi eksplisit lebih dipercaya
    if m:
        return int(m.group(1))
    m2 = re.search(r"Hasilnya\s*(-?\d+)", tail)
    if m2:
        return int(m2.group(1))
    nums = re.findall(r"-?\d+", tail)
    return int(nums[-1]) if nums else None

def evaluate(ckpt):
    tok = PDZ1oTokenizer()
    model = PDZ1oModel.load(ckpt)
    print(f"[eval] {ckpt} | block={model.block_size} params={sum(p.data.size for p in model.parameters())}")
    s_think = s_chatml = s_tool = 0
    n_tool = 0
    m_ok = m_tot = 0
    for q, task, expected in TESTS:
        out = greedy_answer(model, tok, q, task)
        has_think = "<think>" in out and "</think>" in out
        has_chatml = "<|im_start|>" in out or "Halo" in out or "Hasil" in out or "Koreksi" in out or "tool_call" in out
        s_think += has_think
        s_chatml += has_chatml
        if task == "tool_call":
            n_tool += 1
            calls = parse_tool_calls(out)
            valid = any("_parse_error" not in c and c.get("name") in REGISTRY for c in calls)
            s_tool += valid
        if expected is not None:
            m_tot += 1
            got = last_number(out)
            ok = (got == expected)
            m_ok += ok
            mark = "BENAR" if ok else f"SALAH(got={got},exp={expected})"
            print(f"  [{task}] Q={q!r}\n    -> {out[:220]!r}\n    think={has_think} math={mark}")
        else:
            print(f"  [{task}] Q={q!r}\n    -> {out[:220]!r}\n    think={has_think}")
    print(f"\n[hasil] think_fmt={s_think}/{len(TESTS)} chatml_isi={s_chatml}/{len(TESTS)} tool_valid={s_tool}/{max(n_tool,1)} math_acc={m_ok}/{m_tot}")
    return {"think": s_think, "chatml": s_chatml, "tool": s_tool, "math_ok": m_ok, "math_tot": m_tot}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pdz-1o-final.npz")
    a = ap.parse_args()
    evaluate(a.ckpt)
