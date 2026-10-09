"""Model PDZ-1O — decoder-only Transformer DARI NOL (numpy, CPU).
Tanpa torch / pretrained / model pihak ketiga.
Multitasking via task-token conditioning (satu backbone untuk semua task).
"""
import numpy as np

# ============ tiny autograd (numpy) ============
def _unbroadcast(g, shape):
    # jumlahkan grad ke shape asal (untuk broadcast)
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape) if g.shape != shape else g

class Tensor:
    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data = np.array(data, dtype=np.float32) if not isinstance(data, np.ndarray) else data.astype(np.float32, copy=False)
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.requires_grad = requires_grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), "+")
        def _bw():
            if self.requires_grad:
                g = _unbroadcast(out.grad, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = _unbroadcast(out.grad, other.data.shape)
                other.grad = g if other.grad is None else other.grad + g
        out._backward = _bw
        return out
    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), "*")
        def _bw():
            if self.requires_grad:
                g = _unbroadcast(out.grad * other.data, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = _unbroadcast(out.grad * self.data, other.data.shape)
                other.grad = g if other.grad is None else other.grad + g
        out._backward = _bw
        return out
    __rmul__ = __mul__

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self + (other * -1.0)
    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return other + (self * -1.0)
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self * (other ** -1.0)
    def __pow__(self, p):
        out = Tensor(self.data ** p, (self,), f"**{p}")
        def _bw():
            if self.requires_grad:
                g = _unbroadcast(out.grad * p * (self.data ** (p - 1.0)), self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _bw
        return out
    def __neg__(self):
        return self * -1.0
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data @ other.data, (self, other), "@")
        def _bw():
            if self.requires_grad:
                # (...,M,K) @ (...,K,N) -> grad_A = grad @ B^T
                g = np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
                # handle broadcast batch dims
                while g.ndim > self.data.ndim:
                    g = g.sum(axis=0)
                g = _unbroadcast(g, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
                while g.ndim > other.data.ndim:
                    g = g.sum(axis=0)
                g = _unbroadcast(g, other.data.shape)
                other.grad = g if other.grad is None else other.grad + g
        out._backward = _bw
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), (self,), "reshape")
        def _bw():
            if self.requires_grad:
                g = out.grad.reshape(self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _bw
        return out

    def transpose(self, *axes):
        if not axes:
            axes = tuple(reversed(range(self.data.ndim)))
        out = Tensor(self.data.transpose(axes), (self,), "transpose")
        inv = np.argsort(axes)
        def _bw():
            if self.requires_grad:
                g = out.grad.transpose(tuple(inv))
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _bw
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")
        def _bw():
            if self.requires_grad:
                g = out.grad
                if axis is None:
                    g = np.ones_like(self.data) * g
                else:
                    # expand back
                    g = np.ones_like(self.data) * np.expand_dims(g, axis) if not keepdims else np.ones_like(self.data) * g
                    # lebih aman: broadcast
                    g = np.broadcast_to(out.grad if keepdims else np.expand_dims(out.grad, axis), self.data.shape).copy()
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else (self.data.shape[axis] if isinstance(axis, int) else np.prod([self.data.shape[a] for a in axis]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "exp")
        def _bw():
            if self.requires_grad:
                g = _unbroadcast(out.grad * out.data, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _bw
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-12), (self,), "log")
        def _bw():
            if self.requires_grad:
                g = _unbroadcast(out.grad / (self.data + 1e-12), self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _bw
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), "relu")
        def _bw():
            if self.requires_grad:
                g = _unbroadcast(out.grad * (self.data > 0), self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _bw
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")
        def _bw():
            if self.requires_grad:
                g = _unbroadcast(out.grad * out.data * (1 - out.data), self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _bw
        return out

    def silu(self):
        sig = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(self.data * sig, (self,), "silu")
        def _bw():
            if self.requires_grad:
                g = _unbroadcast(out.grad * (sig * (1 + self.data * (1 - sig))), self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _bw
        return out

    def softmax(self, axis=-1):
        m = self.data.max(axis=axis, keepdims=True)
        e = np.exp(self.data - m)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, (self,), "softmax")
        def _bw():
            if self.requires_grad:
                # jacobian-vector product untuk softmax
                g = out.grad
                dot = (g * out.data).sum(axis=axis, keepdims=True)
                dx = out.data * (g - dot)
                dx = _unbroadcast(dx, self.data.shape)
                self.grad = dx if self.grad is None else self.grad + dx
        out._backward = _bw
        return out

    def backward(self):
        topo = []
        visited = set()
        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        self.grad = None

# ============ layers dari nol ============
def embedding(ids: np.ndarray, W: Tensor):
    # ids: (B,T) int ; W: (V,E)
    out_data = W.data[ids]  # (B,T,E)
    out = Tensor(out_data, (W,), "embedding")
    def _bw():
        if W.requires_grad:
            dw = np.zeros_like(W.data)
            np.add.at(dw, ids, out.grad)
            W.grad = dw if W.grad is None else W.grad + dw
    out._backward = _bw
    return out

def rmsnorm(x: Tensor, w: Tensor, eps=1e-6):
    # x: (...,E), w: (E,)
    ms = (x * x).mean(axis=-1, keepdims=True)  # Tensor
    # norm = x / sqrt(ms+eps) * w
    denom_data = np.sqrt(ms.data + eps)
    normed_data = x.data / denom_data * w.data
    # bangun graph via ops agar backward otomatis:
    normed = (x * ((ms + eps) ** -0.5)) * w
    return normed

def cross_entropy(logits: Tensor, targets: np.ndarray):
    # logits: (N,V), targets: (N,)
    N, V = logits.data.shape
    probs = logits.softmax(axis=-1)
    # NLL
    correct = probs.data[np.arange(N), targets]
    loss_data = -np.log(correct + 1e-12).mean()
    out = Tensor(np.array(loss_data), (logits,), "ce")
    # kaitkan probs graph: out tergantung logits via probs; sederhanakan backward langsung:
    def _bw():
        if logits.requires_grad:
            d = probs.data.copy()
            d[np.arange(N), targets] -= 1.0
            d /= N
            d = d * out.grad  # out.grad skalar
            logits.grad = d if logits.grad is None else logits.grad + d
        # putuskan graph probs agar tidak double-count (probs tidak dipakai di tempat lain)
    out._backward = _bw
    # pastikan probs tidak menumpuk grad yang salah: kosongkan keterkaitan
    # (probs node tetap ada tapi grad-nya tidak dipakai karena kita short-circuit)
    out._prev = {logits}
    return out

class AdamW:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.params = params
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad + self.wd * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)
    def zero_grad(self):
        for p in self.params:
            p.grad = None

# ============ model ============
class PDZ1oModel:
    def __init__(self, vocab_size=512, n_layer=4, n_head=4, n_embd=128, block_size=128, seed=1337):
        assert n_embd % n_head == 0
        rng = np.random.default_rng(seed)
        self.vocab_size, self.n_layer, self.n_head = vocab_size, n_layer, n_head
        self.n_embd, self.block_size = n_embd, block_size
        self.hs = n_embd // n_head
        def r(*s, scale=0.02):
            return Tensor(rng.normal(0, scale, size=s))
        self.wte = r(vocab_size, n_embd)          # token embedding
        self.wpe = r(block_size, n_embd)          # posisi (learned)
        self.layers = []
        for _ in range(n_layer):
            self.layers.append(dict(
                ln1_w=Tensor(np.ones(n_embd)),
                qkv_w=r(n_embd, 3 * n_embd), qkv_b=Tensor(np.zeros(3 * n_embd)),
                proj_w=r(n_embd, n_embd), proj_b=Tensor(np.zeros(n_embd)),
                ln2_w=Tensor(np.ones(n_embd)),
                fc_w=r(n_embd, 4 * n_embd), fc_b=Tensor(np.zeros(4 * n_embd)),
                fc2_w=r(4 * n_embd, n_embd), fc2_b=Tensor(np.zeros(n_embd)),
            ))
        self.ln_f = Tensor(np.ones(n_embd))
        self.lm_head = r(n_embd, vocab_size)

    def parameters(self):
        ps = [self.wte, self.wpe, self.ln_f, self.lm_head]
        for L in self.layers:
            ps.extend(L.values())
        return ps

    def forward(self, ids: np.ndarray):
        # ids: (B,T)
        B, T = ids.shape
        assert T <= self.block_size, f"seq {T} > block {self.block_size}"
        x = embedding(ids, self.wte) + embedding(np.tile(np.arange(T)[None, :], (B, 1)), self.wpe)
        for L in self.layers:
            # --- attention block ---
            h = rmsnorm(x, L["ln1_w"])
            qkv = (h @ L["qkv_w"]) + L["qkv_b"]           # (B,T,3E)
            B_, T_, _ = qkv.data.shape
            qkv_r = qkv.reshape(B_, T_, 3, self.n_head, self.hs)
            # split manual via numpy (tanpa autograd split rumit -> gunakan 3 proyeksi terpisah di backward?)
            # Untuk kesederhanaan & kebenaran grad: slice dengan op yang mendukung backward:
            q = self._slice_last(qkv_r, 2, 0)  # (B,T,H,Hs)
            k = self._slice_last(qkv_r, 2, 1)
            v = self._slice_last(qkv_r, 2, 2)
            # scaled dot-product causal
            att_scores = (q.transpose(0, 2, 1, 3) @ k.transpose(0, 2, 3, 1)) * (1.0 / np.sqrt(self.hs))
            # causal mask
            mask = np.triu(np.ones((T_, T_)), k=1).astype(bool)
            masked_data = att_scores.data.copy()
            masked_data[:, :, mask] = -1e9
            att_scores_m = Tensor(masked_data, (att_scores,), "causal")
            # kaitkan backward mask (identitas)
            _orig_prev = set(att_scores._prev)
            def _bw_closure(att_scores=att_scores, out=None):
                pass
            # cara benar: buat node baru yang meneruskan grad
            att_scores_m._prev = {att_scores}
            _att = att_scores
            def make_bw(_att=_att, out=att_scores_m):
                def _bw():
                    if _att.requires_grad:
                        g = out.grad.copy()
                        g[:, :, mask] = 0
                        _att.grad = g if _att.grad is None else _att.grad + g
                return _bw
            att_scores_m._backward = make_bw()
            att_w = att_scores_m.softmax(axis=-1)
            att_out = att_w @ v.transpose(0, 2, 1, 3)          # (B,H,T,Hs)
            att_out = att_out.transpose(0, 2, 1, 3).reshape(B_, T_, self.n_embd)
            proj = (att_out @ L["proj_w"]) + L["proj_b"]
            x = x + proj
            # --- FFN ---
            h2 = rmsnorm(x, L["ln2_w"])
            fc = (h2 @ L["fc_w"]) + L["fc_b"]
            fc = fc.silu()
            fc2 = (fc @ L["fc2_w"]) + L["fc2_b"]
            x = x + fc2
        x = rmsnorm(x, self.ln_f)
        logits = x @ self.lm_head  # (B,T,V)
        return logits

    @staticmethod
    def _slice_last(t: Tensor, axis: int, idx: int):
        # slice t[:,:,..,idx,..] dengan backward scatter
        sl = [slice(None)] * t.data.ndim
        sl[axis] = idx
        out = Tensor(t.data[tuple(sl)], (t,), "slice")
        def _bw():
            if t.requires_grad:
                g = np.zeros_like(t.data)
                g[tuple(sl)] = out.grad
                t.grad = g if t.grad is None else t.grad + g
        out._backward = _bw
        return out

    # --- inference (numpy murni, tanpa graph) ---
    def generate_ids(self, prompt_ids, max_new=64, temperature=0.8, top_k=40, rng=None, stop_ids=None):
        import gc as _gc
        rng = rng or np.random.default_rng()
        stops = set(stop_ids) if stop_ids else set()
        ids = list(prompt_ids)
        for _ in range(max_new):
            ctx = np.array(ids[-self.block_size:])[None, :]
            logits = self.forward(ctx)
            l = logits.data[0, -1]
            del logits  # bebaskan graph forward (hindari menumpuk -> OOM)
            if temperature <= 0:
                nxt = int(np.argmax(l))
            else:
                l = l / max(temperature, 1e-6)
                if top_k and top_k < len(l):
                    idx = np.argpartition(l, -top_k)[-top_k:]
                    nl = np.full_like(l, -1e9)
                    nl[idx] = l[idx]
                    l = nl
                e = np.exp(l - l.max())
                p = e / e.sum()
                nxt = int(rng.choice(len(p), p=p))
            ids.append(nxt)
            if nxt in stops:
                break
        _gc.collect()
        return ids

    def save(self, path):
        d = {f"p{i}": p.data for i, p in enumerate(self.parameters())}
        d["_meta"] = np.array([self.vocab_size, self.n_layer, self.n_head, self.n_embd, self.block_size])
        np.savez(path, **d)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        meta = z["_meta"]
        V, L, H, E, T = map(int, meta)
        m = cls(vocab_size=V, n_layer=L, n_head=H, n_embd=E, block_size=T)
        for i, p in enumerate(m.parameters()):
            p.data = z[f"p{i}"]
        return m
