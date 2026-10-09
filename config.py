"""PDZ-1O config — model + project folder sama: pdz-1o.
Latih dari NOL di CPU, tanpa pretrained / model pihak ketiga.
"""
from dataclasses import dataclass

@dataclass
class PDZ1oConfig:
    # -- identitas --
    model_name: str = "pdz-1o"
    # -- tokenizer (byte-level + special, semua dibuat lokal) --
    vocab_size: int = 512        # 256 byte + 256 slot special/reserved
    # -- arsitektur decoder-only transformer (dari nol, numpy) --
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 96             # kecil agar CPU kuat; naikkan ke 256/384 jika CPU kuat
    block_size: int = 320        # konteks (harus muat 1 sampel penuh ~200-280 token)
    dropout: float = 0.0
    # -- multitask --
    tasks = ("chat", "reasoning", "tool_call", "function_call", "url_context", "self_correct", "novelty")
    task_weights = None          # None = uniform; atau dict task->bobot
    # -- training CPU --
    batch_size: int = 8
    lr: float = 3e-3
    betas = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_steps: int = 1500
    log_every: int = 25
    save_every: int = 250
    seed: int = 1337
    # -- self-learning --
    self_learn_rounds: int = 3
    max_self_samples: int = 200

DEFAULT = PDZ1oConfig()
