"""Tools / function-call / URL konteks untuk PDZ-1O.
Model mempelajari FORMAT; eksekusi dilakukan di sini (Python, CPU).
Format yang dilatih:
  <tool_call>{"name": "hitung", "arguments": {"ekspresi": "2+2"}}</tool_call>
  <url>https://contoh.id/teks</url>
"""
import json, math, time, re, urllib.request

def fn_hitung(ekspresi: str):
    allowed = re.sub(r"[^0-9+\-*/(). %]", "", ekspresi)
    try:
        return {"hasil": eval(allowed, {"__builtins__": {}}, {"math": math})}
    except Exception as e:
        return {"error": f"hitung gagal: {e}. coba sederhanakan ekspresi."}

def fn_waktu(zona: str = "UTC"):
    return {"waktu": time.strftime("%Y-%m-%d %H:%M:%S") + f" {zona}"}

def fn_eksekusi_kode(kode: str):
    # sandbox sangat terbatas: hanya ekspresi / print sederhana
    try:
        ns = {}
        exec("hasil = (" + kode + ")", {"__builtins__": {"len": len, "str": str, "int": int, "float": float, "sum": sum, "min": min, "max": max, "math": math}}, ns)
        return {"hasil": ns.get("hasil")}
    except Exception as e:
        try:
            ns = {}
            exec(kode, {"__builtins__": {}}, ns)
            return {"hasil": str(ns)[:500]}
        except Exception as e2:
            return {"error": f"eksekusi gagal: {e2}. perbaiki sintaks dan coba lagi."}

REGISTRY = {
    "hitung": fn_hitung,
    "waktu": fn_waktu,
    "eksekusi_kode": fn_eksekusi_kode,
}

TOOLCALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
URL_RE = re.compile(r"<url>(.*?)</url>", re.S)

def parse_tool_calls(text: str):
    out = []
    for m in TOOLCALL_RE.finditer(text):
        try:
            o = json.loads(m.group(1).strip())
            out.append(o)
        except Exception as e:
            out.append({"_parse_error": str(e), "_raw": m.group(1)[:200]})
    return out

def execute_tool(name: str, arguments: dict):
    fn = REGISTRY.get(name)
    if fn is None:
        # "memperbaiki kesalahan yang belum pernah diketahui":
        # tawarkan tool terdekat + cara definisi tool baru
        keys = list(REGISTRY.keys())
        return {"error": f"tool '{name}' tidak dikenal. tersedia: {keys}.",
                "saran": " definisikan fungsi baru di tools.py lalu latih ulang 1-2 step."}
    try:
        return fn(**(arguments or {}))
    except TypeError as e:
        return {"error": f"argumen salah untuk '{name}': {e}"}
    except Exception as e:
        return {"error": str(e)}

def fetch_url(url: str, max_chars=2000, timeout=10):
    try:
        req = urllib.request.Request(url.strip(), headers={"User-Agent": "pdz-1o/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()[:20000]
        try:
            txt = raw.decode("utf-8", errors="replace")
        except Exception:
            txt = str(raw[:2000])
        # buang tag html kasar
        txt = re.sub(r"<script.*?</script>", " ", txt, flags=re.S | re.I)
        txt = re.sub(r"<style.*?</style>", " ", txt, flags=re.S | re.I)
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt[:max_chars]
    except Exception as e:
        return f"[gagal ambil url: {e}]"

def inject_url_context(text: str):
    """Ganti <url>..</url> dengan konten fetch agar model bisa menalar di atasnya."""
    def repl(m):
        url = m.group(1)
        return f"<tool_response>konteks url {url}: {fetch_url(url)}</tool_response>"
    return URL_RE.sub(repl, text)
