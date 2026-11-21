import os, random, shutil, torch
import torch.nn.functional as F
import tokenizer
from DNmodel import nanoModel, SignedDivNorm
from ckptio import save_ckpt

SEED = 547844       
ARM  = 'A2'     
RANK  = 16
STEPS = 7001


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False


HERE = os.path.dirname(os.path.abspath(__file__))

#处理数据集
texts = open(os.path.join(HERE, 'shakespeare_complete_pg100.txt'), encoding='utf-8').read()
cache = os.path.join(HERE, 'tokens_3000.pt')
if os.path.exists(cache):
    data = torch.load(cache)
    print('loaded')
else:
    data = torch.tensor(tokenizer.tokenizerEncoding(texts), dtype=torch.long)
    torch.save(data, cache)
    print('tokenized')
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

#超参数
block_size = 256
batch_size = 64
n_embd = 384
hidden_size = n_embd * 4
num_heads = 8
n_layer = 6
dropout = 0.3
max_steps, aver_interval = STEPS, 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

config = dict(vocab_size=tokenizer.LENTH, n_embd=n_embd, n_layer=n_layer,
              num_heads=num_heads, block_size=block_size, dropout=dropout,
              hidden_size=hidden_size, rank=RANK)

set_seed(SEED)
model = nanoModel(**config).to(device)

if ARM == 'A0':                 
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, SignedDivNorm) and m.rank > 0:
                m.U.zero_(); m.V.zero_()
                m.U.requires_grad_(False)
                m.V.requires_grad_(False)

RUN = f's{SEED}_{ARM}'
print(f'run: {RUN}  rank={RANK}  参数 {sum(p.numel() for p in model.parameters()):,}  '
      f'步数 {max_steps}')

# 训练数据顺序：独立随机源，保证 A0/A2 的 batch 序列一致
train_gen = torch.Generator().manual_seed(SEED * 1000 + 1)


def get_batch(split, gen):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,), generator=gen)
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x, y


# 评估批：一次生成、固定复用，val 不再有抽样噪声
def make_eval(split, k=50):
    g = torch.Generator().manual_seed(SEED * 1000 + (2 if split == 'train' else 3))
    return [(x.to(device), y.to(device)) for x, y in (get_batch(split, g) for _ in range(k))]


EVAL = {s: make_eval(s) for s in ('train', 'val')}

KERNEL_REF = {}
for name, m in model.named_modules():
    if isinstance(m, SignedDivNorm):
        KERNEL_REF[name] = m.conv.weight.detach().cpu().clone()


beta_params = [p for n_, p in model.named_parameters() if p.requires_grad and n_.endswith('beta')]
other_params = [p for n_, p in model.named_parameters() if p.requires_grad and not n_.endswith('beta')]
optimizer = torch.optim.AdamW([
    {'params': other_params, 'lr': 3e-4, 'weight_decay': 0.1},
    {'params': beta_params, 'lr': 3e-3, 'weight_decay': 0.0},
], betas=(0.9, 0.99))

best_val = float('inf')


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {s: torch.stack([model(x, y)[1] for x, y in EVAL[s]]).mean() for s in ('train', 'val')}
    model.train()
    return out


@torch.no_grad()
def report_divnorm(model, step):
    rows = []
    for name, m in model.named_modules():
        if not isinstance(m, SignedDivNorm):
            continue
        w = m.conv.weight.detach().cpu()
        w0 = KERNEL_REF[name]
        per_ch, per_ch0 = w.mean(dim=(1, 2)), w0.mean(dim=(1, 2))
        rows.append((name,
                     (per_ch > 0).float().mean().item(),
                     ((per_ch > 0) != (per_ch0 > 0)).float().mean().item(),
                     (w - w0).norm().item() / (w0.norm().item() + 1e-8),
                     m.beta.mean().item()))
    if not rows:
        return
    n_ = len(rows)
    avg = lambda i: sum(r[i] for r in rows) / n_
    print(f'  [核] 正核 {avg(1):6.1%} | 翻转 {avg(2):6.1%} | 漂移 {avg(3):.3f} | β {avg(4):+.3f}')


BASE_LRS = [3e-4, 3e-3]
CKPT = os.path.join(HERE, f'ckpt_{RUN}.pt')

for step in range(max_steps):
    if step <= 0.5*max_steps:
        cur = 3e-4
    elif step <= 0.7*max_steps:
        cur = 3e-5
    else:
        cur = 3e-6 
    for pg, base in zip(optimizer.param_groups, BASE_LRS):
        pg['lr'] = cur * (base / BASE_LRS[0])

    xb, yb = get_batch('train', train_gen)
    xb, yb = xb.to(device), yb.to(device)
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % aver_interval == 0:
        losses = estimate_loss()
        print(f'[{RUN}] step{step}: train {losses["train"]:.4f} | val {losses["val"]:.4f}')
        report_divnorm(model, step)
        v = losses['val'].item()
        if v < best_val:
            best_val = v
            save_ckpt(CKPT, model, config, step, v)

shutil.copyfile(CKPT, os.path.join(HERE, 'ckpt.pt'))   # sample.py 照旧能用
print(f'训练完成 [{RUN}] best val = {best_val:.4f}  ->  {os.path.basename(CKPT)}（已复制为 ckpt.pt）')
