import os
import re
import sys
import torch
from DNmodel import nanoModel, SignedDivNorm

_DN_IN_NET = re.compile(r"^blocks\.(\d+)\.ffn\.net\.(\d+)$")
_DN_BETA_KEY = re.compile(r"^blocks\.(\d+)\.ffn\.net\.(\d+)\.beta$")


def arch_of(model):
    """当前 FFN 的模块顺序，写进 ckpt，日后结构再变时能一眼看出来。"""
    net = model.blocks[0].ffn.net
    return "ffn=[" + ",".join(type(m).__name__ for m in net) + "]"


def _dn_slots_from_model(model):
    slots = {}
    for name, mod in model.named_modules():
        m = _DN_IN_NET.match(name)
        if m and isinstance(mod, SignedDivNorm):
            slots[int(m.group(1))] = int(m.group(2))
    return slots


def _dn_slots_from_state(state):
    slots = {}
    for k in state:
        m = _DN_BETA_KEY.match(k)
        if m:
            slots[int(m.group(1))] = int(m.group(2))
    return slots


def align_state_dict(model, state, verbose=True):
    """把 ckpt 里 DN 的下标挪到模型当前的位置。只改键名，不改数值。"""
    want, have = _dn_slots_from_model(model), _dn_slots_from_state(state)
    if not want or set(want) != set(have):
        return dict(state), []
    out, moved = dict(state), []
    for b, w in want.items():
        h = have[b]
        if h == w:
            continue
        for p in ("beta", "conv.weight"):
            src = f"blocks.{b}.ffn.net.{h}.{p}"
            dst = f"blocks.{b}.ffn.net.{w}.{p}"
            if src in out and dst not in out:
                out[dst] = out.pop(src)
                moved.append((src, dst))
    if verbose and moved:
        print(f"下标对齐：{len(moved)} 个参数 ffn.net.{have[0]}.* -> ffn.net.{want[0]}.*")
    return out, moved


def save_ckpt(path, model, config, step, val):
    torch.save({"model": model.state_dict(), "config": dict(config),
                "step": step, "val": val, "arch": arch_of(model)}, path)


def _infer_rank(state):
    """从 state_dict 里读低秩耦合的 r（U 的第二维）；没有 U 就是 0。"""
    for k, v in state.items():
        if k.endswith('.U'):
            return v.shape[1]
    return 0


def load_ckpt(path, map_location="cpu", verbose=True):
    """严格加载（strict=True）。对不上就报错，不静默留随机权重。"""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = dict(ckpt["config"])
    if "rank" not in cfg:
        cfg["rank"] = _infer_rank(ckpt["model"])
    model = nanoModel(**cfg)
    state, moved = align_state_dict(model, ckpt["model"], verbose=verbose)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(
            f"{os.path.basename(path)} 和当前模型结构不兼容：\n{e}\n"
            f"当前结构：{arch_of(model)}；ckpt 记录的结构：{ckpt.get('arch') or '无'}\n"
            "（权重形状或模块数量变了的话，只能重训，改键名救不回来。）"
        ) from None
    meta = {"step": ckpt.get("step"), "val": ckpt.get("val"),
            "arch": ckpt.get("arch"), "path": path, "remapped": len(moved)}
    if verbose:
        print(f"loaded {os.path.basename(path)}: step={meta['step']} "
              f"val={meta['val']:.4f}  对齐 {meta['remapped']} 个参数"
              f"  结构 {arch_of(model)}")
        if ckpt.get("arch") and ckpt["arch"] != arch_of(model):
            print(f"注意：ckpt 内的结构标记是 {ckpt['arch']}，当前是 {arch_of(model)}。"
                  f"旧 ckpt 的这个字段是写死的常量，可能已过期；"
                  f"以本次加载结果为准（对齐 {meta['remapped']} 个参数）。")
    return model, meta


def _selftest(path):
    """把 DN 下标挪到另一个位置再加载：必须能对齐，且数值逐位不变。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = nanoModel(**ckpt["config"])
    want = _dn_slots_from_model(model)
    have = _dn_slots_from_state(ckpt["model"])
    b0 = sorted(want)[0]
    shift = {b: (w + 1) % 5 for b, w in want.items()}
    fake = {}
    for k, v in ckpt["model"].items():
        m = re.match(r"^blocks\.(\d+)\.ffn\.net\.(\d+)\.(beta|conv\.weight)$", k)
        if m and int(m.group(2)) == have[int(m.group(1))]:
            k = f"blocks.{m.group(1)}.ffn.net.{shift[int(m.group(1))]}.{m.group(3)}"
        fake[k] = v
    tmp = os.path.join(os.path.dirname(os.path.abspath(path)), "_selftest_shift.pt")
    try:
        torch.save({"model": fake, "config": ckpt["config"],
                    "step": ckpt["step"], "val": ckpt["val"]}, tmp)
        m1, _ = load_ckpt(path, verbose=False)
        m2, meta = load_ckpt(tmp, verbose=False)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    assert meta["remapped"] == 2 * len(want), f"对齐参数数 {meta['remapped']} 不对"
    a, b = m1.state_dict(), m2.state_dict()
    assert a.keys() == b.keys()
    diff = max((a[k] - b[k]).abs().max().item() for k in a)
    print(f"selftest ok：DN 下标 {have[b0]} -> {shift[b0]} 后仍能对齐，"
          f"权重最大差异 {diff:.1e}，键集合一致（{len(a)} 个）")


if __name__ == "__main__":
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt.pt")
    if "--selftest" in sys.argv:
        _selftest(p)
    else:
        m, meta = load_ckpt(p, map_location="cpu")
        print(f"参数量 {sum(x.numel() for x in m.parameters()):,}")
