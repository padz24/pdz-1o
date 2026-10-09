"""Tokenizer PDZ-1O — dari NOL, tanpa pretrained.
Skema: byte-level (0-255) + special tokens ChatML/thinking/tool/url.
Tidak butuh download apapun. Deterministik dan reversibel.
"""
import json

SPECIALS = [
    "<|im_start|>", "<|im_end|>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|thinking|>",
    "<think>", "</think>",
    "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>",
    "<url>", "</url>",
    "<|task:chat|>", "<|task:reasoning|>", "<|task:tool_call|>",
    "<|task:function_call|>", "<|task:url_context|>",
    "<|task:self_correct|>", "<|task:novelty|>",
    "<pad>", "<unk>", "<bos>", "<eos>",
]

class PDZ1oTokenizer:
    def __init__(self):
        # id 0-255 = byte value, sisanya = specials + reserved
        self.stoi = {f"<byte:{i}>": i for i in range(256)}
        # untuk encode: byte mentah -> id 0..255 langsung
        self.special_stoi = {}
        nxt = 256
        for s in SPECIALS:
            self.special_stoi[s] = nxt
            nxt += 1
        self.reserved_from = nxt  # sampai vocab_size, untuk masa depan / penemuan baru
        self.special_itos = {v: k for k, v in self.special_stoi.items()}

    @property
    def vocab_size(self):
        return 512

    def encode(self, text: str):
        """Encode: special tokens dipotong dulu (greedy longest-match), sisanya byte-level."""
        ids = []
        i = 0
        specials_sorted = sorted(self.special_stoi.keys(), key=len, reverse=True)
        while i < len(text):
            hit = None
            for s in specials_sorted:
                if text.startswith(s, i):
                    hit = s
                    break
            if hit is not None:
                ids.append(self.special_stoi[hit])
                i += len(hit)
            else:
                ch = text[i]
                for b in ch.encode("utf-8"):
                    ids.append(int(b))
                i += 1
        return ids

    def decode(self, ids):
        out_bytes = bytearray()
        out = ""
        for i in ids:
            if 0 <= i < 256:
                out_bytes.append(i)
            elif i in self.special_itos:
                if out_bytes:
                    out += out_bytes.decode("utf-8", errors="replace")
                    out_bytes = bytearray()
                out += self.special_itos[i]
            else:
                # reserved id yang belum dikenal -> tandai, bukan error
                # (mendukung "memperbaiki kesalahan yang belum pernah diketahui")
                if out_bytes:
                    out += out_bytes.decode("utf-8", errors="replace")
                    out_bytes = bytearray()
                out += f"<r{i}>"
        if out_bytes:
            out += out_bytes.decode("utf-8", errors="replace")
        return out

    # --- ChatML helpers ---
    def chatml(self, role: str, content: str) -> str:
        return f"<|im_start|>{role}\n{content}<|im_end|>\n"

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"specials": SPECIALS}, f)

    @classmethod
    def load(cls, path):
        return cls()  # deterministik, file hanya verifikasi
