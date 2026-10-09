"""Tokenizer BPE PDZ-1O v2 — dilatih DARI NOL di atas korpus lokal.
Tidak memakai tiktoken/sentencepiece/model pihak ketiga.
Skema: special tokens (ChatML/think/tool, sama seperti v1) + byte-level BPE.
Pretoken: pola ala GPT-2 ` ?\S+` (spasi menempel ke kata berikutnya).
"""
import json, re
from collections import Counter

SPLIT_RE = re.compile(r"\s?\S+|\s+")

def _is_space_tok(s):
    return s.strip() == ""

class BPETokenizer:
    def __init__(self, specials=None, merges=None):
        from tokenizer import SPECIALS
        self.specials = list(specials or SPECIALS)
        self.merges = list(merges or [])          # list[(a_str, b_str)] token berupa latin-1 char
        self._rebuild()

    def _rebuild(self):
        self.vocab = {}   # token_str -> id
        self.itos = {}
        for i in range(256):
            ch = chr(i)
            self.vocab[ch] = i
            self.itos[i] = ch
        nxt = 256
        for (a, b) in self.merges:
            t = a + b
            if t not in self.vocab:
                self.vocab[t] = nxt
                self.itos[nxt] = t
                nxt += 1
        self.merge_rank = {(a, b): i for i, (a, b) in enumerate(self.merges)}
        # id special: setelah semua merge-token
        base = 256 + len(self.vocab_minus_base())
        self.special_ids = {}
        for i, s in enumerate(self.specials):
            self.special_ids[s] = base + i
            self.itos[base + i] = s
        self.special_stoi = dict(self.special_ids)  # alias API sama dgn tokenizer v1
        self._special_sorted = sorted(self.specials, key=len, reverse=True)

    def vocab_minus_base(self):
        # token hasil merge saja (urutan = id-256)
        toks = sorted(((v, k) for k, v in self.vocab.items() if v >= 256), key=lambda x: x[0])
        return [k for _, k in toks]

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(self.specials)

    # ---------- training ----------
    def train(self, texts, num_merges=1200, min_freq=2, verbose=True):
        # kumpulkan frekuensi kata (pretoken), kata = tuple latin-1 chars
        freq = Counter()
        for t in texts:
            t2 = t
            # potong specials agar tidak ikut merge
            for s in self._special_sorted if hasattr(self, "_special_sorted") else []:
                t2 = t2.replace(s, " ")
            for m in SPLIT_RE.finditer(t2):
                g = m.group(0)
                if g.strip() == "":
                    continue  # whitespace dikode mentah, tidak ikut merge
                w = g.encode("utf-8").decode("latin-1")
                freq[tuple(w)] += 1
        words = dict(freq)
        if verbose:
            print(f"[bpe] kata unik: {len(words)} total: {sum(words.values())}")
        merges = []
        for it in range(num_merges):
            pair_counts = Counter()
            for w, c in words.items():
                for i in range(len(w) - 1):
                    pair_counts[(w[i], w[i + 1])] += c
            if not pair_counts:
                break
            (a, b), n = pair_counts.most_common(1)[0]
            if n < min_freq:
                if verbose:
                    print(f"[bpe] stop @{it}: pasangan terbaik hanya {n}x")
                break
            merges.append((a, b))
            # terapkan merge ke semua kata
            new_words = {}
            for w, c in words.items():
                nw = []
                i = 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                        nw.append(a + b)
                        i += 2
                    else:
                        nw.append(w[i])
                        i += 1
                new_words[tuple(nw)] = c
            words = new_words
            if verbose and (it + 1) % 200 == 0:
                print(f"[bpe] merge {it + 1}/{num_merges} terakhir={a!r}+{b!r} ({n}x)")
        self.merges = merges
        self._rebuild()
        if verbose:
            print(f"[bpe] selesai: {len(merges)} merge, vocab={self.vocab_size}")
        return self

    # ---------- encode / decode ----------
    def _encode_word(self, word_bytes_latin):
        parts = list(word_bytes_latin)
        # terapkan merge sesuai rank (buble sederhana, kata pendek jadi murah)
        while len(parts) > 1:
            best = None
            best_rank = None
            for i in range(len(parts) - 1):
                r = self.merge_rank.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best = i
            if best is None:
                break
            parts = parts[:best] + [parts[best] + parts[best + 1]] + parts[best + 2:]
        return [self.vocab[p] for p in parts]

    def encode(self, text):
        ids = []
        i = 0
        n = len(text)
        buf = []
        def flush():
            for m in SPLIT_RE.finditer("".join(buf)):
                g = m.group(0)
                if _is_space_tok(g):
                    ids.extend(g.encode("utf-8"))
                else:
                    w = g.encode("utf-8").decode("latin-1")
                    ids.extend(self._encode_word(w))
            buf.clear()
        while i < n:
            hit = None
            for s in self._special_sorted:
                if text.startswith(s, i):
                    hit = s
                    break
            if hit is not None:
                flush()
                ids.append(self.special_ids[hit])
                i += len(hit)
            else:
                buf.append(text[i])
                i += 1
        flush()
        return ids

    def decode(self, ids):
        out = []
        bbuf = []
        def flush_b():
            if bbuf:
                out.append("".join(bbuf).encode("latin-1").decode("utf-8", errors="replace"))
                bbuf.clear()
        for i in ids:
            if 0 <= i < 256 + len(self.merges):
                bbuf.append(self.itos[i])
            elif i in self.itos:
                flush_b()
                out.append(self.itos[i])
            else:
                flush_b()
                out.append(f"<r{i}>")
        flush_b()
        return "".join(out)

    def chatml(self, role, content):
        return f"<|im_start|>{role}\n{content}<|im_end|>\n"

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": self.merges, "specials": self.specials}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(specials=d["specials"], merges=[tuple(m) for m in d["merges"]])
