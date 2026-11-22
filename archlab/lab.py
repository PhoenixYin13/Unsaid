"""Architecture lab: fixed-depth Transformer vs. a weight-shared "think until stable" looped model.

Both models use the same block (pre-LN attention with clipped relative position bias + MLP).
  baseline : L distinct blocks, applied once each (compute is fixed by the architecture).
  looped   : ONE block applied repeatedly, input re-injected each step; at test time it may
             run for as many steps as it likes and halts when its answer stops changing.

Tasks are per-position labelling problems whose required number of sequential steps grows
with the instance, so "train short, test long" measures whether the model learned the
algorithm or just a fixed-depth shortcut.

Usage: python lab.py --task parity --model looped --seed 0 --out results/
"""
import argparse, json, math, os, time
import torch, torch.nn as nn, torch.nn.functional as F


# ----------------------------- tasks -----------------------------
def make_parity(bs, n, g):
    x = torch.randint(0, 2, (bs, n), generator=g)
    return x, x.cumsum(1) % 2


def make_add(bs, n, g, hard):
    """Binary addition, least-significant bit first. Input token = a_i + 2*b_i, n+1 positions.
    hard=True plants a chain of 'propagate' positions across (almost) the whole number, started by
    either a generate (1+1) or a kill (0+0) chosen at random, so every output bit on the chain depends
    on information that must travel the full distance."""
    a = torch.randint(0, 2, (bs, n), generator=g)
    b = torch.randint(0, 2, (bs, n), generator=g)
    if hard:
        start = torch.randint(0, max(1, n // 4), (bs, 1), generator=g)
        idx = torch.arange(n).unsqueeze(0)
        b = torch.where(idx > start, 1 - a, b)
        bit = torch.randint(0, 2, (bs, 1), generator=g).expand(bs, n)
        a = torch.where(idx == start, bit, a)
        b = torch.where(idx == start, bit, b)
    carry = torch.zeros(bs, dtype=torch.long)
    out = []
    for i in range(n):
        s = a[:, i] + b[:, i] + carry
        out.append(s % 2)
        carry = s // 2
    out.append(carry)
    z = torch.zeros(bs, 1, dtype=torch.long)
    return torch.cat([a + 2 * b, z], 1), torch.stack(out, 1)


def batch(task, bs, n, g, hard=False):
    if task == 'parity':
        return make_parity(bs, n, g)
    if task == 'add':
        return make_add(bs, n, g, hard)
    raise ValueError(task)


VOCAB = {'parity': 2, 'add': 4}


# ----------------------------- model -----------------------------
class Block(nn.Module):
    def __init__(self, d, heads, K, window=0):
        super().__init__()
        self.h, self.K, self.window = heads, K, window   # window>0: attend only to |i-j|<=window
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.rel = nn.Parameter(torch.zeros(heads, 2 * K + 1))
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        pos = torch.arange(N)
        rel = (pos[None, :] - pos[:, None]).clamp(-self.K, self.K) + self.K
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.h) + self.rel[:, rel]
        if self.window:
            att = att.masked_fill((pos[None, :] - pos[:, None]).abs() > self.window, float('-inf'))
        y = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(y)
        return x + self.mlp(self.ln2(x))


class Baseline(nn.Module):
    def __init__(self, vocab, d=64, heads=4, K=8, layers=8, window=0):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, heads, K, window) for _ in range(layers))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, 2)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln(h))


class Looped(nn.Module):
    def __init__(self, vocab, d=64, heads=4, K=8, window=0):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.mix = nn.Linear(2 * d, d)
        self.block = Block(d, heads, K, window)
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, 2)

    def step(self, h, e):
        return self.block(self.mix(torch.cat([h, e], -1)))

    def read(self, h):
        return self.head(self.ln(h))

    def forward(self, x, iters, h=None):
        e = self.emb(x)
        h = torch.zeros_like(e) if h is None else h
        for _ in range(iters):
            h = self.step(h, e)
        return self.read(h), h

    @torch.no_grad()
    def think(self, x, max_iters, patience=3, tol=0.02):
        """Iterate until each sequence's hidden state has converged (relative change < tol at every
        position, and unchanged answer) for `patience` consecutive steps.
        Returns final predictions, steps used per sequence, and predictions after every step."""
        e = self.emb(x)
        h = torch.zeros_like(e)
        B = x.shape[0]
        prev = None
        stable = torch.zeros(B, dtype=torch.long)
        used = torch.full((B,), max_iters, dtype=torch.long)
        final = torch.zeros_like(x)
        done = torch.zeros(B, dtype=torch.bool)
        trace = []
        for t in range(1, max_iters + 1):
            h_new = self.step(h, e)
            delta = ((h_new - h).norm(dim=-1) / (h_new.norm(dim=-1) + 1e-6)).amax(1)
            h = h_new
            p = self.read(h).argmax(-1)
            trace.append(p)
            if prev is not None:
                same = (p == prev).all(1) & (delta < tol)
                stable = torch.where(same, stable + 1, torch.zeros_like(stable))
                halt = (stable >= patience) & ~done
                final[halt] = p[halt]
                used[halt] = t
                done |= halt
            prev = p
        final[~done] = prev[~done]
        return final, used, trace


# ----------------------------- train / eval -----------------------------
def evaluate(model, task, kind, lengths, g, n_eval=100):
    res = {}
    model.eval()
    with torch.no_grad():
        for n in lengths:
            x, y = batch(task, n_eval, n, g, hard=True)
            if kind == 'baseline':
                p = model(x).argmax(-1)
                res[n] = {'seq_acc': (p == y).all(1).float().mean().item(),
                          'tok_acc': (p == y).float().mean().item()}
            else:
                budget = 2 * n + 16
                p, used, trace = model.think(x, budget)
                curve = {t: (trace[t - 1] == y).all(1).float().mean().item()
                         for t in (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512) if t <= budget}
                res[n] = {'seq_acc': (p == y).all(1).float().mean().item(),
                          'tok_acc': (p == y).float().mean().item(),
                          'steps_mean': used.float().mean().item(),
                          'steps_max': used.max().item(),
                          'acc_vs_steps': curve}
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='parity')
    ap.add_argument('--model', default='looped')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--max_len', type=int, default=20)
    ap.add_argument('--train_iters', type=int, default=28)
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--window', type=int, default=0)
    ap.add_argument('--stab', type=float, default=0.0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    V = VOCAB[a.task]
    model = Baseline(V, d=a.d, layers=a.layers, window=a.window) if a.model == 'baseline' else Looped(V, d=a.d, window=a.window)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
    lengths = [a.max_len, 2 * a.max_len, 4 * a.max_len, 8 * a.max_len]
    log, t0 = [], time.time()
    M = a.train_iters
    for s in range(a.steps):
        n = int(torch.randint(2, a.max_len + 1, (1,), generator=g))
        x, y = batch(a.task, a.bs, n, g, hard=bool(s % 2))
        if a.model == 'baseline':
            logits = model(x)
        elif s % 2 == 0:                       # full run
            logits, h = model(x, M)
            if a.stab:                         # the answer should be a fixed point of the update
                h2 = model.step(h, model.emb(x))
                stab = ((h2 - h) ** 2).mean() / (h.detach() ** 2).mean()
        else:                                  # progressive: improve from an arbitrary earlier state
            k0 = int(torch.randint(0, M, (1,), generator=g))
            k1 = int(torch.randint(1, M - k0 + 1, (1,), generator=g))
            with torch.no_grad():
                _, h = model(x, k0)
            logits, _ = model(x, k1, h.detach())
        loss = F.cross_entropy(logits.reshape(-1, 2), y.reshape(-1))
        if a.model != 'baseline' and s % 2 == 0 and a.stab:
            loss = loss + a.stab * stab
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (s + 1) % 1000 == 0 or s + 1 == a.steps:
            ev = evaluate(model, a.task, a.model, lengths, torch.Generator().manual_seed(123))
            log.append({'step': s + 1, 'loss': loss.item(), 'eval': ev})
            print(f"[{a.task}/{a.model}{a.tag}/s{a.seed}] step {s+1} loss {loss.item():.4f} {time.time()-t0:.0f}s  " +
                  '  '.join(f"n={n}:{ev[n]['seq_acc']:.2f}" for n in lengths), flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), f"{a.out}/{a.task}_{a.model}{a.tag}_s{a.seed}.pt")
    json.dump({'args': vars(a), 'params': params, 'train_seconds': time.time() - t0, 'log': log},
              open(f"{a.out}/{a.task}_{a.model}{a.tag}_s{a.seed}.json", 'w'), indent=1)


if __name__ == '__main__':
    main()
