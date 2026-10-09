"""PDZ-1O belajar dari WEBSITE secara otonom (tanpa dataset manual).
- Menemukan URL sendiri: dari prompt user, dari output model, dari daftar situs (>10).
- Fetch + cache + ekstrak (teks, heading, contoh kode, JSON API) -> sampel ChatML+think.
- Retrieval saat bingung: deteksi kebingungan -> ambil konteks web -> jawab ulang.
Situs: dummyjson (JSON API), w3schools (python/js/sql/html), python docs, dsb.
Sopan: cache lokal, batas halaman & ukuran, timeout, User-Agent jelas.
"""
import os, re, json, time, urllib.request, urllib.parse
from html.parser import HTMLParser
from dataset_self import sample_with_think, save_jsonl, load_jsonl

CACHE = "data/web_cache"
os.makedirs(CACHE, exist_ok=True)

SITES = [
    # JSON API publik (dummyjson) — fakta terstruktur, cocok untuk QA akurat
    "https://dummyjson.com/products/1",
    "https://dummyjson.com/products/2",
    "https://dummyjson.com/users/1",
    "https://dummyjson.com/quotes/1",
    "https://dummyjson.com/products/categories",
    # tutorial pemrograman (w3schools)
    "https://www.w3schools.com/python/python_intro.asp",
    "https://www.w3schools.com/python/python_variables.asp",
    "https://www.w3schools.com/python/python_for_loops.asp",
    "https://www.w3schools.com/js/js_intro.asp",
    "https://www.w3schools.com/sql/sql_intro.asp",
    "https://www.w3schools.com/html/html_intro.asp",
    # dokumentasi resmi python
    "https://docs.python.org/3/tutorial/introduction.html",
]

URL_RE = re.compile(r"https?://[^\s<>\"']+")

def find_urls(text):
    return URL_RE.findall(text or "")

def _cache_path(url):
    import hashlib
    return os.path.join(CACHE, hashlib.md5(url.encode()).hexdigest() + ".txt")

def fetch(url, max_bytes=250000, timeout=15):
    cp = _cache_path(url)
    if os.path.exists(cp):
        with open(cp, encoding="utf-8", errors="replace") as f:
            return f.read()
    req = urllib.request.Request(url, headers={"User-Agent": "pdz-1o-weblearner/1.0 (belajar lokal)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(max_bytes + 1)[:max_bytes]
    try:
        txt = raw.decode("utf-8", errors="replace")
    except Exception:
        txt = ""
    with open(cp, "w", encoding="utf-8") as f:
        f.write(txt)
    time.sleep(0.5)  # sopan ke server
    return txt

class Extract(HTMLParser):
    TARGETS = ("h1", "h2", "h3", "title", "p", "pre")
    INLINE = ("span", "a", "strong", "em", "b", "i", "u", "code",
              "small", "mark", "br", "sup", "sub")
    def __init__(self):
        super().__init__()
        self.title, self.heads, self.paras, self.codes = "", [], [], []
        self._cur, self._in, self._div_in = "", None, 0
    def _is_codediv(self, tag, attrs):
        if tag != "div":
            return False
        for k, v in attrs:
            if k == "class" and ("w3-code" in v or "w3-example" in v or "highlight" in v):
                return True
        return False
    def handle_starttag(self, tag, attrs):
        if tag in self.INLINE:
            if tag == "br" and self._in is not None:
                self._cur += "\n"
            return  # inline = bagian dari teks berjalan
        if self._is_codediv(tag, attrs):
            if self._in is None:
                self._in, self._cur = "pre", ""
            self._div_in += 1
            return
        if tag in self.TARGETS:
            if self._in is not None:
                self._close()  # HTML toleran: tutup paksa elemen sebelumnya
            self._in, self._cur = tag, ""
    def handle_endtag(self, tag):
        if tag in self.INLINE:
            return
        if tag == "div" and self._div_in > 0:
            self._div_in -= 1
            if self._div_in == 0 and self._in == "pre":
                self._close()
            return
        if self._in == tag:
            self._close()
    def _close(self):
        tag = self._in
        t = re.sub(r"\s+", " ", self._cur).strip()
        if tag == "title": self.title = t[:120]
        elif tag in ("h1", "h2", "h3"):
            if t: self.heads.append(t[:140])
        elif tag == "p":
            if len(t) > 40: self.paras.append(t[:400])
        elif tag in ("pre", "code"):
            if len(t) > 5: self.codes.append(t[:220])
        self._in = None
    def handle_data(self, data):
        if self._in is not None and len(self._cur) < 3000:
            self._cur += data

def extract_page(url, raw):
    raw_s = raw.strip()
    if raw_s.startswith("{") or raw_s.startswith("["):
        try:
            return ("json", json.loads(raw_s))
        except Exception:
            pass
    p = Extract()
    try:
        p.feed(raw[:250000])
    except Exception:
        pass
    return ("html", {"title": p.title, "heads": p.heads[:6],
                     "paras": p.paras[:4], "codes": p.codes[:3]})

# ---------- sintesis sampel (otomatis, tanpa manual) ----------
def samples_from_json(url, obj):
    out = []
    if isinstance(obj, dict) and "title" in obj and "price" in obj:  # produk dummyjson
        t, pr = obj.get("title"), obj.get("price")
        out.append({"task": "url_context", "text": sample_with_think(
            "url_context", f"ringkas <url>{url}</url>",
            "Baca konteks url lalu ringkas faktual.",
            f"<tool_response>konteks url {url}: produk {t} harga {pr}</tool_response>\nRingkasan: {t} harganya {pr}.")})
        out.append({"task": "chat", "text": sample_with_think(
            "chat", f"berapa harga {t}?",
            f"Ingat dari konteks web: {t} = {pr}.",
            f"Harganya {pr}.")})
        if "rating" in obj:
            out.append({"task": "chat", "text": sample_with_think(
                "chat", f"berapa rating {t}?",
                f"Ingat dari konteks web: rating {obj['rating']}.",
                f"Ratingnya {obj['rating']}.")})
    elif isinstance(obj, dict) and "firstName" in obj:  # user dummyjson
        nm = f"{obj.get('firstName')} {obj.get('lastName')}"
        out.append({"task": "chat", "text": sample_with_think(
            "chat", f"siapa user 1 di dummyjson?",
            f"Ingat dari konteks web: {nm}.",
            f"Namanya {nm}.")})
    elif isinstance(obj, dict) and "quote" in obj:
        out.append({"task": "chat", "text": sample_with_think(
            "chat", "beri satu quote dari dummyjson",
            "Ambil quote dari konteks web.",
            f"Quote: {obj['quote'][:160]}")})
    elif isinstance(obj, list):  # categories
        cats = ", ".join(str(c) for c in obj[:8])
        out.append({"task": "chat", "text": sample_with_think(
            "chat", "sebutkan kategori produk dummyjson",
            "Ingat dari konteks web.",
            f"Kategorinya: {cats}.")})
    return out

NAV_BAD = ("tutorial", "tutorials", "reference", "references", "menu", "w3schools",
           "topnav", "sidenav", "example", "exercises", "quiz", "certificate",
           "pro ", " spaces", "bootcamp", "advertisement", "cookie",
           "learn web development", "data science, and more", "newsletter",
           "join our", "all rights reserved", "report error")

def topic_words(url):
    slug = url.split("/")[-1].replace(".asp", "").replace(".html", "")
    return set(re.findall(r"[a-z]{3,}", slug.replace("_", " ").replace("-", " ")))

def pick_heads(url, heads):
    tw = topic_words(url)
    scored = []
    for h in heads:
        hl = h.lower()
        if any(b in hl for b in NAV_BAD):
            continue
        wl = set(re.findall(r"[a-z]{3,}", hl))
        scored.append((len(wl & tw), h))
    scored.sort(key=lambda x: -x[0])
    good = [h for s, h in scored if s > 0][:3]
    if not good:
        good = [h for _, h in scored[:2]]
    return good

def samples_from_html(url, h):
    out = []
    title = h["title"] or url.split("/")[-1]
    heads = pick_heads(url, h["heads"])
    paras = [p for p in h["paras"] if not any(b in p.lower() for b in NAV_BAD)][:2]
    for head in heads[:3]:
        out.append({"task": "url_context", "text": sample_with_think(
            "url_context", f"jelaskan {head} dari <url>{url}</url>",
            "Baca bagian halaman lalu jelaskan singkat.",
            f"<tool_response>konteks {title}: {head}</tool_response>\nPenjelasan: {head} adalah bagian dari {title}.")})
    for para in paras[:2]:
        out.append({"task": "chat", "text": sample_with_think(
            "chat", f"apa isi bagian {title}?",
            "Ingat isi halaman yang dipelajari.",
            f"Isinya: {para[:180]}")})
    for code in h["codes"][:2]:
        out.append({"task": "tool_call", "text": sample_with_think(
            "tool_call", f"jalankan contoh kode dari {title} pakai tool",
            "Contoh dari web dijalankan lewat tool.",
            f'<tool_call>{{"name": "eksekusi_kode", "arguments": {{"kode": {json.dumps(code[:120])}}}}}</tool_call>')})
        topic = " ".join(sorted(topic_words(url))[:3])
        out.append({"task": "chat", "text": sample_with_think(
            "chat", f"beri contoh kode {topic or title}",
            "Kutip contoh yang dipelajari dari web.",
            f"Contohnya: {code[:150]}")})
    return out

def learn_from_sites(sites=None, out_path="data/web_learn.jsonl"):
    sites = sites or SITES
    print(f"[web-learn] {len(sites)} situs")
    all_rows, index = [], {}
    for u in sites:
        try:
            kind, data = extract_page(u, fetch(u))
        except Exception as e:
            print(f"  lewati {u}: {e}")
            continue
        rows = samples_from_json(u, data) if kind == "json" else samples_from_html(u, data)
        print(f"  {u} -> {len(rows)} sampel ({kind})")
        all_rows.extend(rows)
        blob = (u + " " + json.dumps(data)[:4000]).lower()
        for w in re.findall(r"[a-zA-Z]{3,}", blob):
            for key in {w, _stem(w)}:
                index.setdefault(key, [])
                if u not in index[key]:
                    index[key].append(u)
    # + URL yang ditemukan di dalam halaman (discovery berantai, maks 5 baru)
    extra = []
    for u in list(sites):
        try:
            for f in find_urls(fetch(u))[:50]:
                if f not in sites and f.startswith("http") and len(extra) < 5:
                    extra.append(f)
        except Exception:
            pass
    print(f"  URL baru ditemukan: {len(extra)}")
    save_jsonl(all_rows, out_path)
    with open("data/web_index.json", "w") as f:
        json.dump({"index": index, "sites": sites, "extra": extra}, f)
    print(f"[web-learn] total {len(all_rows)} sampel -> {out_path}")
    return all_rows

# ---------- retrieval saat model bingung ----------
CONFUSED_RE = re.compile(r"tidak tahu|tidak yakin|kurang tahu|maaf|bingung|tidak dapat", re.I)

def extractive_answer(question, contexts):
    """Jawab ekstraktif dari konteks (disalin dari sumber -> akurat).
    Dipakai saat model kecil belum mampu grounding generatif."""
    import re as _re
    ctx = " ".join(contexts)
    qs = set(_re.findall(r"[a-zA-Z]{3,}", (question or "").lower()))
    # 1. pola 'NAMA harga ANGKA'
    cands = []
    for m in _re.finditer(r"([A-Z][\w\s\-']{2,50}?)\s+harga\s+([\d][\d.,]*)", ctx):
        t, p = m.group(1).strip(), m.group(2)
        ov = len(set(_re.findall(r"[a-zA-Z]{3,}", t.lower())) & qs)
        cands.append((ov, t, p))
    if cands and any(w in ("harga", "harganya", "price", "berapa") for w in qs):
        cands.sort(reverse=True)
        _, t, p = cands[0]
        return f"Produk {t} harganya {p}."
    # 2. kalimat dengan kata kunci terbanyak (lewati navigasi)
    sents = _re.split(r"(?<=[.!?])\s+", ctx)
    best, best_ov = "", -1
    for s in sents:
        if len(s) < 15 or len(s) > 300:
            continue
        sl = s.lower()
        if any(b in sl for b in ("tutorial", "w3schools", "learn web development",
                                 "data science, and more", "references", "menu",
                                 "cookie", "advertisement", "try it yourself")):
            continue
        ov = len(set(_re.findall(r"[a-zA-Z]{3,}", s.lower())) & qs)
        if ov > best_ov:
            best, best_ov = s, ov
    if best and best_ov >= 2:
        return best.strip()
    return ""

def is_confused(question, draft_answer):
    d = (draft_answer or "").strip()
    if CONFUSED_RE.search(d):
        return True, "model mengaku tidak tahu"
    qs = set(re.findall(r"[a-zA-Z]{4,}", (question or "").lower()))
    hits = sum(1 for w in qs if w in d.lower())
    if len(d) < 20 and ("<think>" not in d or hits == 0):
        return True, "jawaban terlalu pendek-tak-jelas"
    if qs and hits == 0 and len(d) > 20:
        return True, "jawaban tak menyentuh kata kunci soal"
    if "<tool_call>" in (draft_answer or "") and "tool_call>" not in (draft_answer or "").replace("<tool_call>", ""):
        return True, "format tool rusak"
    return False, ""

def _stem(w):
    return w[:-1] if len(w) > 4 and w.endswith("s") else w

def retrieve_urls(question, k=2):
    try:
        idx = json.load(open("data/web_index.json"))
    except Exception:
        return []
    inv = idx.get("index", {})
    words = re.findall(r"[a-zA-Z]{3,}", (question or "").lower())
    STOPW = {"dan", "atau", "apa", "itu", "yang", "dari", "untuk", "dengan",
             "adalah", "saya", "kamu", "anda", "beri", "contoh", "tentang"}
    words = [w for w in words if w not in STOPW]
    qkeys = set()
    for w in words:
        qkeys.add(w)
        qkeys.add(_stem(w))
    # niat produk -> langsung ke API pencarian (kata kunci terlangka)
    STOP = {"produk", "product", "harga", "harganya", "price", "cari", "carikan",
            "dummyjson", "sebutkan", "berapa", "yang", "dari", "untuk", "dengan",
            "adalah", "dan", "atau", "apa", "itu", "saya", "kamu", "anda", "bisa",
            "akan", "telah", "sudah", "sangat", "juga", "agar", "dalam", "tentang",
            "karena", "oleh", "pada", "saja", "beri", "berikan", "tunjukkan", "jelaskan"}
    if any(w in ("produk", "product", "harga", "price", "cari", "carikan",
                 "jual", "toko", "shop", "phone", "laptop") for w in words):
        def rare(w):
            return (0 if w not in inv else 1, len(inv.get(w, [])))
        cands = [w for w in words if w not in STOP]
        if cands:
            best = sorted(cands, key=rare)[0]
            return [f"https://dummyjson.com/products/search?q={urllib.parse.quote(best)}"]
    score, hitwords = {}, {}
    for w in qkeys:
        urls = inv.get(w, [])
        rar = 1.0 / (1.0 + len(urls))  # kata langka bobot besar
        for u in urls:
            score[u] = score.get(u, 0.0) + rar
            hitwords.setdefault(u, set()).add(w)
    ranked = sorted(score, key=lambda u: (len(hitwords[u]), score[u]), reverse=True)
    good = [u for u in ranked if len(hitwords[u]) >= 2][:k]
    if good:
        return good
    if ranked and len(hitwords[ranked[0]]) >= 1 and any(
            w in ("python", "javascript", "sql", "html", "loop", "variable",
                  "dummyjson", "produk", "product", "harga", "price") for w in hitwords[ranked[0]]):
        return ranked[:1]
    # fallback: API pencarian dummyjson (pencarian web sungguhan)
    q = urllib.parse.quote(" ".join(words[:3]))
    if q:
        return [f"https://dummyjson.com/products/search?q={q}"]
    return []

def search_result_samples(url):
    """Ubah hasil fetch (termasuk /search) menjadi konteks + sampel QA."""
    try:
        kind, data = extract_page(url, fetch(url))
    except Exception as e:
        return [], f"[gagal ambil {url}: {e}]"
    if kind == "json" and isinstance(data, dict) and "products" in data:
        prods = data["products"][:3]
        ctx = "; ".join(f"{p.get('title')} harga {p.get('price')}" for p in prods) or "tidak ada produk"
        rows = [{"task": "chat", "text": sample_with_think(
            "chat", f"ringkas hasil cari <url>{url}</url>",
            "Baca hasil pencarian lalu ringkas.",
            f"<tool_response>{ctx}</tool_response>\nRingkasan: {ctx}.")}]
        return rows, ctx
    if kind == "json":
        ctx = json.dumps(data)[:600]
        return [], ctx
    _, h = kind, data
    nav = ("tutorial", "w3schools", "reference", "references", "exercise",
           "quiz", "certificate", "menu", "learn web development")
    heads = [x for x in h["heads"][:3] if not any(b in x.lower() for b in nav)]
    codes = [x for x in h["codes"][:1] if "try it yourself" not in x.lower() or True]
    codes = [c.split("Try it Yourself")[0].strip() for c in codes]
    parts = [h["title"]] + heads + codes
    ctx = " ".join(p for p in parts if p)[:600] or h["title"]
    return [], ctx

if __name__ == "__main__":
    learn_from_sites()
