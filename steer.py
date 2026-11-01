# Does the report follow the odds when they are pushed, with the text unchanged? At each held-out position, a push on the
# residual at the last token, along the gradient of the second choice's log-probability, grows until the frozen model's
# first and second choices swap. The reader then reads that same pushed text. A random push of the same size is the
# control. Writes runnerup/steer-<run>.json.
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runnerup as R  # next to this file
import torch
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="qwen-qwen-r4-inert"); ap.add_argument("--n", type=int, default=400)
ap.add_argument("--layer", type=int, default=20); ap.add_argument("--ckpt", default="final"); ap.add_argument("--mode", default="raise", choices=["raise", "lower"])
a = ap.parse_args()
tok, base = R.load("qwen")
model = PeftModel.from_pretrained(base, os.path.join(R.OUT, a.run, a.ckpt)); model.eval()
ok, tg = R.kept()
items = [e for e in ok if e["split"] == "eval"]
random.Random(7).shuffle(items); items = items[:a.n]
layer = model.base_model.model.model.layers[a.layer]
push = {"v": None, "n": 0}


def hook(mod, inp, out):
  h = out[0] if isinstance(out, tuple) else out
  if push["v"] is None or h.shape[1] != push["n"]: return out
  h = h.clone(); h[0, -1] += push["v"].to(h.dtype)
  return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


handle = layer.register_forward_hook(hook)


def odds(ids):
  with torch.no_grad(), model.disable_adapter():
    return torch.softmax(model(torch.tensor([ids], device="cuda:0")).logits[0, -1].float(), -1)


def gradient(ids, token):
  # The direction that raises this token, and the residual's size there, from the frozen model.
  got = {}
  def grab(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    h.retain_grad(); got["h"] = h
  g = layer.register_forward_hook(grab)
  emb = model.get_input_embeddings()(torch.tensor([ids], device="cuda:0")).detach().requires_grad_(True)
  with model.disable_adapter():
    lp = torch.log_softmax(model(inputs_embeds=emb).logits[0, -1].float(), -1)[token]
    lp.backward()
  g.remove(); model.zero_grad(set_to_none=True)
  return got["h"].grad[0, -1].float().detach(), got["h"][0, -1].float().norm().item()


def words(text):
  m = R.parse_answer(text) if hasattr(R, "parse_answer") else None
  return m


def report(prefix, ids):
  text, ev = R.read_one(model, tok, "qwen", prefix)
  return text, ev


rows = []
tag = "" if a.mode == "raise" else "-lower"
for k, e in enumerate(items):
  ids = R.encode_prefix(tok, "qwen", e["prefix"])
  push["n"], push["v"] = len(ids), None
  p0 = odds(ids); top0 = torch.topk(p0, 3).indices.tolist()
  r0, ev0 = report(e["prefix"], ids)
  # raise: up the second choice's own gradient. lower: down the first choice's, so nothing pushes toward what replaces it.
  d, size = gradient(ids, top0[1] if a.mode == "raise" else top0[0]); unit = d / d.norm()
  if a.mode == "lower": unit = -unit
  swapped, alpha = False, 0.0
  for alpha in [size * x for x in (0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.45, 0.65, 1.0)]:
    push["v"] = alpha * unit
    p1 = odds(ids)
    if (a.mode == "raise" and int(p1.argmax()) == top0[1]) or (a.mode == "lower" and int(p1.argmax()) != top0[0]):
      swapped = True; break
  row = {"prefix": e["prefix"][-120:], "swapped": swapped, "alpha_rel": alpha / size, "top0": [tok.decode([t]) for t in top0], "p0": [round(float(p0[t]), 3) for t in top0], "said0": r0, "ev0": ev0}
  if swapped:
    top1 = torch.topk(p1, 3).indices.tolist()
    r1, ev1 = report(e["prefix"], ids)
    row.update(top1=[tok.decode([t]) for t in top1], p1=[round(float(p1[t]), 3) for t in top1], said1=r1, ev1=ev1)
    rnd = torch.randn_like(unit); rnd = rnd / rnd.norm()
    push["v"] = alpha * rnd
    p2 = odds(ids); top2 = torch.topk(p2, 3).indices.tolist()
    r2, ev2 = report(e["prefix"], ids)
    row.update(top2=[tok.decode([t]) for t in top2], p2=[round(float(p2[t]), 3) for t in top2], said2=r2, ev2=ev2)
  rows.append(row)
  if k % 25 == 24:
    print(k + 1, sum(r["swapped"] for r in rows), flush=True)
    json.dump(rows, open(os.path.join(R.OUT, f"steer-{a.run}{tag}.json"), "w"), indent=0, ensure_ascii=False)
json.dump(rows, open(os.path.join(R.OUT, f"steer-{a.run}{tag}.json"), "w"), indent=0, ensure_ascii=False)
print("STEER-OVER")
