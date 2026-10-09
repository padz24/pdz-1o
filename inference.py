"""Inference PDZ-1O: ChatML + thinking + tool/function call + URL konteks.
Contoh:
  python inference.py --ckpt checkpoints/pdz-1o-final.npz --prompt "hitung 7*6 pakai tool"
  python inference.py --ckpt ... --interactive
"""
import argparse, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from tokenizer import PDZ1oTokenizer
from model import PDZ1oModel
from tools import parse_tool_calls, execute_tool, inject_url_context

def build_prompt(tok, user_text, task="chat"):
    system = "Kamu PDZ-1O. Bernalar dulu dalam <think>, lalu jawab. Gunakan tool jika perlu."
    p = (f"<|task:{task}|>"
         + tok.chatml("system", system)
         + tok.chatml("user", inject_url_context(user_text)))
    return p

def detect_task(user_text):
    u = user_text.lower()
    if "tool" in u or "hitung" in u or "fungsi" in u or "<tool_call>" in u:
        return "tool_call"
    if "url" in u or "http" in u or "ringkas" in u:
        return "url_context"
    if "langkah" in u or "bernalar" in u or re.search(r"\d\s*[+\-*]\s*\d", u):
        return "reasoning"
    if "perbaiki" in u or "salah" in u:
        return "self_correct"
    if "baru" in u or "temukan" in u or "pola" in u:
        return "novelty"
    return "chat"

def answer(model, tok, user_text, max_new=160, temperature=0.7):
    task = detect_task(user_text)
    prompt = build_prompt(tok, user_text, task)
    ids = tok.encode(prompt)
    if len(ids) > model.block_size:
        ids = ids[-model.block_size:]  # jaga konteks muat (block 320 = sampel penuh)
    stops = {tok.special_stoi["<|im_end|>"]}
    gen = model.generate_ids(ids, max_new=max_new, temperature=temperature, stop_ids=stops)
    full = tok.decode(gen)
    out = full[len(prompt):]
    # eksekusi tool yang diminta model
    calls = parse_tool_calls(out)
    extra = ""
    for c in calls:
        if "_parse_error" in c:
            extra += f"\n<tool_response>format tool salah: {c['_parse_error']}. Perbaiki JSON lalu coba lagi.</tool_response>"
        else:
            res = execute_tool(c.get("name", ""), c.get("arguments", {}))
            extra += f"\n<tool_response>{res}</tool_response>"
    return out + extra

FACTUAL_HINTS = ("berapa", "siapa", "sebutkan", "carikan", "beri contoh",
                 "harga", "kapan", "dimana", "mengapa", "bagaimana", "?",
                 "http", "w3schools", "dummyjson", "dokumentasi", "contoh kode")

def answer_web(model, tok, user_text, max_new=160, temperature=0.7):
    """Jawab otonom dengan web: draf -> cek bingung -> retrieval -> jawab ulang."""
    from web_learn import (is_confused, retrieve_urls, search_result_samples,
                           find_urls, extractive_answer)
    factual = any(h in user_text.lower() for h in FACTUAL_HINTS)
    draf, bingung, alasan = "", False, ""
    if not factual:
        draf = answer(model, tok, user_text, max_new=max_new, temperature=temperature)
        bingung, alasan = is_confused(user_text, draf)
    else:
        alasan = "soal faktual — langsung cari sumber"
    urls = find_urls(user_text)
    if not bingung and not urls and not factual:
        return draf
    if not urls:
        urls = retrieve_urls(user_text, k=2)
    konteks, sumber = [], []
    for u in urls[:2]:
        _, ctx = search_result_samples(u)
        if ctx and "gagal ambil" not in ctx:
            konteks.append(ctx[:200])
            sumber.append(u)
    if not konteks:
        fb = draf if draf else "Saya tidak menemukan sumber web untuk soal ini."
        return fb + "\n<tool_response>tidak ada konteks web yang bisa diambil.</tool_response>"
    task = detect_task(user_text)
    ringkas = " | ".join(konteks)[:220]
    user2 = f"{user_text}\nKonteks web: {ringkas}\nJawab berdasar konteks."
    p2 = build_prompt(tok, user2, task)
    ids2 = tok.encode(p2)
    if len(ids2) > model.block_size:  # pangkas konteks, jangan soal/system
        ids2 = ids2[:model.block_size]
    stops = {tok.special_stoi["<|im_end|>"]}
    gen = model.generate_ids(ids2, max_new=max_new, temperature=temperature, stop_ids=stops)
    model_out = tok.decode(gen[len(ids2):]).replace("\x00", "").strip()
    # Grounding neuro-simbolik: jawaban faktual disalin dari sumber (akurat);
    # model yang memutuskan kapan mencari + URL apa. Cek kewarasan output model:
    ekstrak = extractive_answer(user_text, konteks)
    sane = (len(model_out) > 10 and "\x00" not in model_out
            and sum(1 for w in __import__("re").findall(r"[a-zA-Z]{4,}", user_text.lower())
                    if w in model_out.lower()) >= 1
            and "mengabaikan prioritas mengabaikan" not in model_out)
    if sane and ekstrak and ekstrak.lower() not in model_out.lower():
        final = model_out + f"\n<tool_response>cek sumber: {ekstrak}</tool_response>"
    elif sane:
        final = model_out
    elif ekstrak:
        final = (f"<think>Mencari di web karena: {alasan or 'butuh fakta terkini'}. "
                 f"Menyalin jawaban dari sumber agar akurat.</think>\n{ekstrak}")
    else:
        kutip = " | ".join(konteks)[:300]
        final = (f"<think>Sumber diambil tapi tak ada jawaban pasti. Mengutip sumber.</think>\n"
                 f"Kutipan sumber: {kutip}")
    return final + f"\n<sumber>{', '.join(sumber)}</sumber>"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pdz-1o-final.npz")
    ap.add_argument("--prompt", default="halo, siapa kamu? jelaskan kemampuanmu.")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--max_new", type=int, default=160)
    ap.add_argument("--tok", type=str, default="byte", choices=["byte", "bpe"])
    ap.add_argument("--bpe", type=str, default="data/bpe2.json")
    ap.add_argument("--web", action="store_true",
                    help="mode otonom: cek bingung -> retrieval web -> jawab ulang")
    a = ap.parse_args()
    if a.tok == "bpe":
        from tokenizer_bpe import BPETokenizer
        tok = BPETokenizer.load(a.bpe)
    else:
        tok = PDZ1oTokenizer()
    if os.path.exists(a.ckpt):
        model = PDZ1oModel.load(a.ckpt)
        print(f"[pdz-1o] checkpoint dimuat: {a.ckpt}")
    else:
        print(f"[peringatan] {a.ckpt} belum ada (model belum dilatih). memakai bobot acak untuk demo format.")
        model = PDZ1oModel(vocab_size=tok.vocab_size, n_layer=2, n_head=4, n_embd=64, block_size=64)
    if a.interactive:
        print("PDZ-1O interaktif (ketik 'keluar' untuk berhenti).")
        fn = answer_web if a.web else answer
        while True:
            try:
                u = input("kamu> ").strip()
            except EOFError:
                break
            if u.lower() in ("keluar", "exit", "quit"):
                break
            print("pdz-1o> " + fn(model, tok, u, max_new=a.max_new) + "\n")
    else:
        fn = answer_web if a.web else answer
        print(fn(model, tok, a.prompt, max_new=a.max_new))

if __name__ == "__main__":
    main()
