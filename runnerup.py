# Can a model learn to report its own runner-up tokens, better than another model can learn to report them? At a word
# boundary in human-written text the reader sees the text, a marker, and writes its top 3 next tokens and a confidence
# digit for the first: ' " brown" " red" " lazy" =6'. The text is run with the adapter OFF (the frozen model's own
# computation, its KV cache) and only the marker and answer with it on. Targets are a frozen model's logits.
# Self vs other, as in Binder et al. 2024: Qwen3-8B-Base and Gemma 3 4B (similar age and ability), each trained to report
# its own distribution and the other's, on the same positions.
#   python -I runnerup.py corpus                       -> runnerup/examples.json
#   python -I runnerup.py targets --model qwen|gemma   -> runnerup/targets-<model>.json
#   python -I runnerup.py train --reader qwen --target gemma --rank 4 [--steps N]  -> runnerup/<reader>-<target>-r<rank>/
#   python -I runnerup.py probe --run qwen-qwen-r4    -> what the trained model is like (see probe())
#   python -I runnerup.py ask --run qwen-qwen-r4 --texts FILE   -> true top 5 beside what the reader says
#   python -I runnerup.py report                       -> runnerup/report.json, printed
import argparse, glob, hashlib, json, math, os, random, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runnerup")
HOME = os.path.expanduser("~")
MODELS = {"qwen": os.path.join(HOME, "qwen8b-base"), "gemma": os.path.join(HOME, "gemma3-text")}
HF = os.path.join(HOME, ".cache/huggingface/hub")
MARK = "⟦?⟧"  # the marker after the text: a symbol, no words
MAXC = 1200  # prefix characters at most (cut on the left at a space)
TOPK = 10
P1MAX = 0.9  # positions where either model is surer than this are left out: no runner-up worth naming


def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)


# ---------------------------------------------------------------- corpus
def wiki_clean(s):
  for a, b in ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" , ", ", "), (" . ", ". "), (" ; ", "; "), (" : ", ": "),
               (" 's", "'s"), ("( ", "("), (" )", ")"), (" n't", "n't")):
    s = s.replace(a, b)
  return re.sub(r" \.\n", ".\n", s)


def wiki_articles(path):
  import pandas as pd
  lines = pd.read_parquet(path).text.tolist()
  arts, cur = [], []
  for line in lines:
    if re.match(r"^ = [^=].* = \n$", line) and cur:
      arts.append("".join(cur)); cur = []
    cur.append(line)
  arts.append("".join(cur))
  return [wiki_clean(x).strip() for x in arts if len(x) > 3000]


def oasst(path):
  import pandas as pd
  d = pd.read_parquet(path)
  d = d[(d.lang == "en") & (~d.synthetic) & (~d.deleted)]
  roots = d[d.parent_id.isna()]
  top = d[(d.role == "assistant") & (d["rank"] == 0)].set_index("parent_id")
  return [(r.text, top.loc[r.message_id].text) for r in roots.itertuples()
          if r.message_id in top.index and len(top.loc[r.message_id].text) > 600]


def positions(text, start, rng, n):
  # character offsets c where text[c] is a space after a non-space: the next token starts a word in both tokenizers
  cs = [m.start() for m in re.finditer(r"(?<=\S) ", text) if m.start() >= start + 200]
  rng.shuffle(cs)
  out = []
  for c in sorted(cs[:n]):
    lo = max(0, c - MAXC)
    if lo > 0: lo = text.index(" ", lo) + 1
    out.append(text[lo:c])
  return out


def corpus(a):
  rng = random.Random(0)
  wt = glob.glob(os.path.join(HF, "datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1"))[0]
  oa = glob.glob(os.path.join(HF, "datasets--OpenAssistant--oasst1/snapshots/*/data"))[0]
  src = {"train": {"wiki": wiki_articles(os.path.join(wt, "train-00000-of-00002.parquet")),
                   "chat": oasst(glob.glob(os.path.join(oa, "train-*.parquet"))[0])},
         "eval": {"wiki": wiki_articles(os.path.join(wt, "test-00000-of-00001.parquet")),
                  "chat": oasst(os.path.join(HERE, "confidence/src/oasst1-validation.parquet"))}}
  # Code: CPython's standard library, from the first def or class on; one file in seven held out, by name
  for f in sorted(glob.glob("/usr/lib64/python3.12/*.py")):
    s = open(f, encoding="utf-8").read()
    m = re.search(r"^(def|class) ", s, re.M)
    if not m or len(s) - m.start() < 3000: continue
    split = "eval" if int(hashlib.md5(os.path.basename(f).encode()).hexdigest(), 16) % 7 == 0 else "train"
    src[split].setdefault("code", []).append(s[m.start():])
  want = {"train": a.train, "eval": a.eval}
  ex = []
  for split in ("train", "eval"):
    for kind in ("wiki", "chat", "code"):
      docs = src[split][kind]
      rng.shuffle(docs)
      per = 40 if kind == "code" else 6 if split == "train" else 20
      n = want[split] // 3
      got = 0
      for d in docs:
        if kind == "chat":
          head = d[0] + "\n\n"
          for p in positions(head + d[1], len(head), rng, per): ex.append({"split": split, "kind": kind, "prefix": p})
        else:
          for p in positions(d, 0, rng, per): ex.append({"split": split, "kind": kind, "prefix": p})
        got = sum(1 for e in ex if e["split"] == split and e["kind"] == kind)
        if got >= n: break
      ex = [e for e in ex if not (e["split"] == split and e["kind"] == kind)] + \
           [e for e in ex if e["split"] == split and e["kind"] == kind][:n]
      log(f"{split} {kind}: {min(got, n)} positions from {len(docs)} documents")
  for i, e in enumerate(ex): e["id"] = i
  os.makedirs(OUT, exist_ok=True)
  json.dump({"source": "wikitext-103 (train / test), oasst1 (train / validation, en, non-synthetic, rank-0 replies), "
             "CPython 3.12 stdlib (1 file in 7 held out)", "examples": ex}, open(os.path.join(OUT, "examples.json"), "w"))


# ---------------------------------------------------------------- models
def load(name, adapter_rank=None):
  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer
  tok = AutoTokenizer.from_pretrained(MODELS[name])
  model = AutoModelForCausalLM.from_pretrained(MODELS[name], dtype=torch.bfloat16, device_map="cuda:0")
  model.eval()
  return tok, model


def encode_prefix(tok, name, text):
  return tok(text, add_special_tokens=(name == "gemma")).input_ids  # Gemma wants its BOS; Qwen has none


def piece(tok, ctx, i):
  # a token's text where it stands: decode(context + token) minus decode(context), so leading spaces survive
  a = tok.decode(ctx); b = tok.decode(ctx + [i])
  return b[len(a):] if b.startswith(a) else tok.decode([i])


def targets(a):
  import torch
  ex = json.load(open(os.path.join(OUT, "examples.json")))["examples"]
  tok, model = load(a.model)
  rows, began = [], time.time()
  with torch.no_grad():
    for k, e in enumerate(ex):
      ids = encode_prefix(tok, a.model, e["prefix"])
      logits = model(torch.tensor([ids], device="cuda:0")).logits[0, -1].float()
      lp = torch.log_softmax(logits, -1)
      p = lp.exp()
      ent = float(-(p * lp).sum() / math.log(2))
      top = torch.topk(p, TOPK)
      ctx = ids[-8:]
      rows.append({"id": e["id"], "p": [round(float(x), 5) for x in top.values],
                   "s": [piece(tok, ctx, int(i)) for i in top.indices], "H": round(ent, 4)})
      if k % 2000 == 0: log(f"{a.model} {k}/{len(ex)} {time.time() - began:.0f}s")
  json.dump(rows, open(os.path.join(OUT, f"targets-{a.model}.json"), "w"))
  log(f"targets-{a.model}.json: {len(rows)} positions")


def kept():
  # the positions both models are unsure at, split train / eval
  ex = json.load(open(os.path.join(OUT, "examples.json")))["examples"]
  tq = {r["id"]: r for r in json.load(open(os.path.join(OUT, "targets-qwen.json")))}
  tg = {r["id"]: r for r in json.load(open(os.path.join(OUT, "targets-gemma.json")))}
  ok = [e for e in ex if tq[e["id"]]["p"][0] < P1MAX and tg[e["id"]]["p"][0] < P1MAX]
  return ok, {"qwen": tq, "gemma": tg}


def answer(t):
  return " " + " ".join(json.dumps(s, ensure_ascii=False) for s in t["s"][:3]) + f" ={min(9, int(t['p'][0] * 10))}\n"


ANS = re.compile(r'\s*("(?:[^"\\]|\\.)*")\s*("(?:[^"\\]|\\.)*")\s*("(?:[^"\\]|\\.)*")\s*=(\d)')


def parse(s):
  m = ANS.match(s)
  if not m: return None
  try: return [json.loads(m.group(i)) for i in (1, 2, 3)], int(m.group(4))
  except ValueError: return None


# ---------------------------------------------------------------- reader
def read_batch(model, tok, name, prefixes, gen=32):
  # read_one for many at once: the texts left-padded through the frozen model, then all answers greedily together
  import torch
  pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
  pre = [encode_prefix(tok, name, p) for p in prefixes]
  n, L = len(pre), max(map(len, pre))
  mask = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in pre], device="cuda:0")
  digit = {tok.convert_tokens_to_ids(str(d)): d for d in range(10)}
  digit_ids = list(digit)
  with torch.no_grad():
    with model.disable_adapter():
      cache = model(input_ids=torch.tensor([[pad] * (L - len(x)) + x for x in pre], device="cuda:0"),
                    attention_mask=mask, position_ids=(mask.cumsum(-1) - 1).clamp(min=0), use_cache=True,
                    logits_to_keep=1).past_key_values
    nxt = torch.tensor([tok(MARK, add_special_tokens=False).input_ids] * n, device="cuda:0")
    pos = torch.tensor([len(x) for x in pre], device="cuda:0")[:, None]
    out, ev, done = [[] for _ in range(n)], [None] * n, [False] * n
    for _ in range(gen):
      live = torch.tensor([[0 if d else 1] for d in done], device="cuda:0").expand(-1, nxt.shape[1])
      mask = torch.cat([mask, live], -1)
      o = model(input_ids=nxt, attention_mask=mask, past_key_values=cache, use_cache=True,
                position_ids=pos + torch.arange(nxt.shape[1], device="cuda:0")[None])
      cache, pos = o.past_key_values, pos + nxt.shape[1]
      lg = o.logits[:, -1].float()
      ts = lg.argmax(-1).tolist()
      for i, t in enumerate(ts):
        if done[i]: continue
        if t in digit and tok.decode(out[i]).rstrip().endswith("="):
          pd = torch.softmax(lg[i, digit_ids], -1)
          ev[i] = float(sum(pd[k] * d for k, d in enumerate(digit.values())))
        out[i].append(t)
        if t == tok.eos_token_id or "\n" in tok.decode([t]): done[i] = True
      if all(done): break
      nxt = torch.tensor([[t if not d else pad] for t, d in zip(ts, done)], device="cuda:0")
  return [(tok.decode(x), e) for x, e in zip(out, ev)]


def read_many(model, tok, name, prefixes, chunk=24):
  rows = []
  for i in range(0, len(prefixes), chunk):
    rows += read_batch(model, tok, name, prefixes[i:i + chunk])
  return rows


def read_one(model, tok, name, prefix, gen=40):
  # frozen pass over the text, then the marker and a greedy answer with the adapter on; returns text and the
  # expected confidence digit from the distribution at the digit's position
  import torch
  ids = encode_prefix(tok, name, prefix)
  with torch.no_grad():
    with model.disable_adapter():
      out = model(torch.tensor([ids], device="cuda:0"), use_cache=True)
    cache = out.past_key_values
    nxt = tok(MARK, add_special_tokens=False).input_ids
    pos, text, ev = len(ids), [], None
    digit = {tok.convert_tokens_to_ids(str(d)): d for d in range(10)}
    for _ in range(gen):
      o = model(torch.tensor([nxt], device="cuda:0"), past_key_values=cache, use_cache=True,
                position_ids=torch.arange(pos, pos + len(nxt), device="cuda:0")[None])
      cache, pos = o.past_key_values, pos + len(nxt)
      lg = o.logits[0, -1].float()
      t = int(lg.argmax())
      if t in digit and tok.decode(text).rstrip().endswith("="):
        pd = torch.softmax(lg[list(digit)], -1)
        ev = float(sum(pd[k] * d for k, d in enumerate(digit.values())))
      text.append(t)
      if t == tok.eos_token_id or "\n" in tok.decode([t]): break
      nxt = [t]
  return tok.decode(text), ev  # decoded whole: Gemma drops a token's leading space when it is decoded alone


def batch_loss(model, tok, name, mark, prefixes, answers):
  # The texts left-padded through the frozen model, then marker and answer right-padded on top of its cache with the
  # adapter on; the loss on the answer tokens only.
  import torch
  pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
  pre = [encode_prefix(tok, name, p) for p in prefixes]
  L = max(map(len, pre))
  P = torch.tensor([[pad] * (L - len(x)) + x for x in pre], device="cuda:0")
  pm = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in pre], device="cuda:0")
  with torch.no_grad():
    with model.disable_adapter():
      cache = model(input_ids=P, attention_mask=pm, position_ids=(pm.cumsum(-1) - 1).clamp(min=0),
                    use_cache=True, logits_to_keep=1).past_key_values  # its logits are never used
  qa = [mark + x for x in answers]
  M = max(map(len, qa))
  Q = torch.tensor([x + [pad] * (M - len(x)) for x in qa], device="cuda:0")
  qm = torch.tensor([[1] * len(x) + [0] * (M - len(x)) for x in qa], device="cuda:0")
  lens = torch.tensor([len(x) for x in pre], device="cuda:0")
  o = model(input_ids=Q, attention_mask=torch.cat([pm, qm], -1), past_key_values=cache, use_cache=True,
            position_ids=lens[:, None] + torch.arange(M, device="cuda:0")[None])
  labels = torch.full((len(qa), M), -100, device="cuda:0")
  for i, x in enumerate(qa):
    labels[i, len(mark):len(x)] = Q[i, len(mark):len(x)]
  return torch.nn.functional.cross_entropy(o.logits[:, :-1].float().reshape(-1, o.logits.shape[-1]),
                                           labels[:, 1:].reshape(-1), ignore_index=-100)


def train(a):
  import torch
  from peft import LoraConfig, get_peft_model
  torch.manual_seed(a.seed); rng = random.Random(a.seed)
  run = os.path.join(OUT, f"{a.reader}-{a.target}-r{a.rank}" + (f"-s{a.seed}" if a.seed else "") + (a.tag or ""))
  os.makedirs(run, exist_ok=True)
  ok, tg = kept()
  tr = [e for e in ok if e["split"] == "train"]
  ev = [e for e in ok if e["split"] == "eval"]
  rng.shuffle(ev); curve_set = ev[:a.curve]
  log(f"{run}: {len(tr)} train, {len(ev)} eval positions")
  tok, model = load(a.reader)
  cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
  model = get_peft_model(model, cfg)
  mark = tok(MARK, add_special_tokens=False).input_ids
  opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
  warm = 100
  sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))))
  logf = open(os.path.join(run, "log.jsonl"), "a")

  def evaluate(step, items, final=False):
    model.eval()
    got = read_many(model, tok, a.reader, [e["prefix"] for e in items])
    rows = [{"id": e["id"], "out": s, "ev": evd} for e, (s, evd) in zip(items, got)]
    score = score_rows(rows, tg[a.target])
    logf.write(json.dumps({"step": step, "n": len(rows), **score}) + "\n"); logf.flush()
    log(f"step {step} eval {len(rows)}: " + " ".join(f"{k} {v}" for k, v in score.items()))
    if final: json.dump(rows, open(os.path.join(run, "eval.json"), "w"))
    model.train()

  evaluate(0, curve_set)
  model.train()
  step, k, running, inert_running, began = 0, 0, 0.0, 0.0, time.time()
  order = list(range(len(tr)))
  while step < a.steps:
    for _ in range(a.accum):
      batch = []
      for _ in range(a.batch):
        if k % len(order) == 0: rng.shuffle(order)
        batch.append(tr[order[k % len(order)]]); k += 1
      loss = batch_loss(model, tok, a.reader, mark, [e["prefix"] for e in batch],
                        [tok(answer(tg[a.target][e["id"]]), add_special_tokens=False).input_ids for e in batch]) / a.accum
      loss.backward()  # before the plain-text pass, so the two graphs are never held at once (10-30, OOM)
      running += loss.item()
      if a.inert:
        # The negatives the first readers lacked : plain text read with the adapter ON must predict as
        # the frozen model does. KL(frozen || adapted) over every position of a few of the batch's texts, cut to
        # their last 160 tokens (same tokens both ways), so the reading habit stays behind the marker. Only the last 64
        # positions' logits are kept: all 160 over the full vocabulary ran out of memory after the step-50 eval (10-30).
        ids = [encode_prefix(tok, a.reader, e["prefix"]) for e in batch[:a.inert_texts]]
        m = min(160, *map(len, ids))
        X = torch.tensor([x[-m:] for x in ids], device="cuda:0")
        with torch.no_grad(), model.disable_adapter():
          lp = torch.log_softmax(model(X, logits_to_keep=64).logits.float(), -1)
        lq = torch.log_softmax(model(X, logits_to_keep=64).logits.float(), -1)
        kl = (lp.exp() * (lp - lq)).sum(-1).mean()
        (a.inert * kl / a.accum).backward()
        running += a.inert * kl.item() / a.accum
        inert_running += kl.item()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step(); sched.step(); opt.zero_grad(); step += 1
    if step % 50 == 0:
      log(f"step {step:5} loss {running / 50:.4f}" + (f" plain-text KL {inert_running / 50 / a.accum:.4f} nats" if a.inert else "")
          + f" lr {sched.get_last_lr()[0]:.2e} {time.time() - began:.0f}s"); running = inert_running = 0.0
    if (step in (25, 50, 100, 200, 300) or step % a.every == 0) and step < a.steps:  # dense early: how fast it comes
      evaluate(step, curve_set)
      model.save_pretrained(os.path.join(run, f"ckpt-{step}"))
      torch.cuda.empty_cache()  # the eval's batches leave the allocator fragmented: 10-29 a run died at step ~2000
  model.save_pretrained(os.path.join(run, "final"))
  evaluate(step, ev[:a.final], final=True)


def probe(a):
  # What is the trained model like? (1) Asked the same way with the adapter on over the text as well, as one model and
  # not a reader on a frozen one; (2) at confident positions (p1 >= 0.9), left out of training; (3) its ordinary
  # next-token prediction with the adapter on: KL from the untrained model, top-1 kept, entropy; (4) free continuations
  # of held-out text with the adapter on, written to a file, nothing scored.
  import torch
  from peft import PeftModel
  run = os.path.join(OUT, a.run)
  reader, target = a.run.split("-")[:2]
  ex = json.load(open(os.path.join(OUT, "examples.json")))["examples"]
  truth = {r["id"]: r for r in json.load(open(os.path.join(OUT, f"targets-{target}.json")))}
  other = {r["id"]: r for r in json.load(open(os.path.join(OUT, f"targets-{'gemma' if target == 'qwen' else 'qwen'}.json")))}
  ev = [e for e in ex if e["split"] == "eval"]
  rng = random.Random(1); rng.shuffle(ev)
  unsure = [e for e in ev if truth[e["id"]]["p"][0] < P1MAX and other[e["id"]]["p"][0] < P1MAX][:a.n]
  sure = [e for e in ev if truth[e["id"]]["p"][0] >= P1MAX][:a.n]
  tok, base = load(reader)
  model = PeftModel.from_pretrained(base, os.path.join(run, a.ckpt))
  model.eval()
  res = {}

  class Whole:  # the adapter left on over the text too: disable_adapter becomes a no-op
    def __init__(s, m): s.m = m
    def __call__(s, *x, **k): return s.m(*x, **k)
    def disable_adapter(s):
      import contextlib
      return contextlib.nullcontext()

  for label, m, items in (("frozen_text", model, unsure), ("adapter_on_text", Whole(model), unsure),
                          ("confident_positions", model, sure)):
    got = read_many(m, tok, reader, [e["prefix"] for e in items])
    rows = [{"id": e["id"], "out": s, "ev": evd} for e, (s, evd) in zip(items, got)]
    res[label] = score_rows(rows, truth)
    json.dump(rows, open(os.path.join(run, f"probe-{label}.json"), "w"))
    log(f"{label}: {res[label]}")
  kl = agree = dh = 0.0
  with torch.no_grad():
    for e in unsure:
      ids = torch.tensor([encode_prefix(tok, reader, e["prefix"])], device="cuda:0")
      lq = torch.log_softmax(model(ids).logits[0, -1].float(), -1)
      with model.disable_adapter():
        lp = torch.log_softmax(model(ids).logits[0, -1].float(), -1)
      kl += float((lq.exp() * (lq - lp)).sum()) / LN2
      agree += int(lq.argmax() == lp.argmax())
      dh += float(-(lq.exp() * lq).sum() + (lp.exp() * lp).sum()) / LN2
  k = len(unsure)
  res["plain_text"] = {"kl_bits": round(kl / k, 4), "top1_kept": round(100 * agree / k, 1), "entropy_change_bits": round(dh / k, 4)}
  log(f"plain text: {res['plain_text']}")
  free = []
  with torch.no_grad():
    for e in unsure[:a.free]:
      ids = torch.tensor([encode_prefix(tok, reader, e["prefix"])], device="cuda:0")
      for label, ctx in (("adapter", None), ("untrained", "off")):
        torch.manual_seed(0)
        if ctx:
          with model.disable_adapter():
            o = model.generate(ids, max_new_tokens=60, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
        else:
          o = model.generate(ids, max_new_tokens=60, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
        free.append({"id": e["id"], "who": label, "text": tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)})
  json.dump({"prefixes": {e["id"]: e["prefix"][-300:] for e in unsure[:a.free]}, "continuations": free},
            open(os.path.join(run, "probe-free.json"), "w"), indent=1)
  json.dump(res, open(os.path.join(run, "probe.json"), "w"), indent=1)


LN2 = math.log(2.0)


# ---------------------------------------------------------------- scoring
def score_rows(rows, truth):
  n = len(rows); parsed = top1 = second = sec_in3 = over = 0; err, pr, tr = [], [], []
  for r in rows:
    t = truth[r["id"]]; got = parse(r["out"])
    if not got: continue
    parsed += 1
    toks, d = got
    top1 += toks[0] == t["s"][0]; second += toks[1] == t["s"][1]; sec_in3 += toks[1] in t["s"][:3]
    over += len(set(toks) & set(t["s"][:3])) / 3
    c = (r["ev"] if r.get("ev") is not None else d) / 10 + 0.05
    err.append(abs(c - t["p"][0])); pr.append(c); tr.append(t["p"][0])
  f = lambda x: round(100 * x / max(n, 1), 1)
  return {"parsed": f(parsed), "top1": f(top1), "second": f(second), "second_in_top3": f(sec_in3), "overlap3": f(over),
          "conf_mae": round(sum(err) / max(len(err), 1), 3), "conf_r": round(corr(pr, tr), 3)}


def corr(x, y):
  if len(x) < 3: return float("nan")
  mx, my = sum(x) / len(x), sum(y) / len(y)
  sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
  return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


def report(a):
  ok, tg = kept()
  ev = [e for e in ok if e["split"] == "eval"]
  res = {}
  # Baseline: the other model's actual top 3 and top-1 probability, read straight off its logits
  for tgt, oth in (("qwen", "gemma"), ("gemma", "qwen")):
    rows = [{"id": e["id"], "out": answer(tg[oth][e["id"]]), "ev": min(9, int(tg[oth][e["id"]]["p"][0] * 10))} for e in ev]
    res[f"logits-of-{oth}->{tgt}"] = score_rows(rows, tg[tgt])
  agree2 = sum(tg["qwen"][e["id"]]["s"][1] == tg["gemma"][e["id"]]["s"][1] for e in ev) / len(ev)
  res["models_share_second_choice"] = round(100 * agree2, 1)
  dis = {e["id"] for e in ev if tg["qwen"][e["id"]]["s"][1] != tg["gemma"][e["id"]]["s"][1]}
  for run in sorted(glob.glob(os.path.join(OUT, "*-*-r*"))):
    f = os.path.join(run, "eval.json")
    if not os.path.exists(f): continue
    rows = json.load(open(f))
    tgt = os.path.basename(run).split("-")[1]
    res[os.path.basename(run)] = {"all": score_rows(rows, tg[tgt]),
                                  "where_models_disagree": score_rows([r for r in rows if r["id"] in dis], tg[tgt])}
  json.dump(res, open(os.path.join(OUT, "report.json"), "w"), indent=1)
  print(json.dumps(res, indent=1))


def ask(a):
  # For the thread's picture: sentences (one per line in --texts), each read by the trained model; its true top 5 from
  # the frozen model's logits beside what the reader says. Nothing here is scored or trained on.
  import torch
  from peft import PeftModel
  reader = a.run.split("-")[0]
  texts = [t.rstrip("\n") for t in open(a.texts, encoding="utf-8") if t.strip()]
  tok, base = load(reader)
  model = PeftModel.from_pretrained(base, os.path.join(OUT, a.run, a.ckpt))
  model.eval()
  said = read_many(model, tok, reader, texts)
  rows = []
  with torch.no_grad(), model.disable_adapter():
    for t, (s, ev) in zip(texts, said):
      ids = encode_prefix(tok, reader, t)
      p = torch.softmax(model(torch.tensor([ids], device="cuda:0")).logits[0, -1].float(), -1)
      top = torch.topk(p, 5)
      true = [(piece(tok, ids[-8:], int(i)), round(float(v), 3)) for v, i in zip(top.values, top.indices)]
      rows.append({"text": t, "true": true, "said": s, "said_parsed": parse(s), "confidence_ev": ev})
      print(f"\n{t[-90:]}\n  true: " + "  ".join(f"{w!r} {q:.0%}" for w, q in true) + f"\n  said: {s.strip()}"
            + (f"   (expected digit {ev:.1f})" if ev is not None else ""), flush=True)
  json.dump(rows, open(os.path.join(OUT, a.run, "ask-" + os.path.basename(a.texts) + ".json"), "w"), indent=1,
            ensure_ascii=False)


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("cmd")
  ap.add_argument("--train", type=int, default=30000)
  ap.add_argument("--eval", type=int, default=3000)
  ap.add_argument("--model")
  ap.add_argument("--reader"); ap.add_argument("--target")
  ap.add_argument("--rank", type=int, default=4)
  ap.add_argument("--steps", type=int, default=3000)
  ap.add_argument("--accum", type=int, default=1)
  ap.add_argument("--batch", type=int, default=8)
  ap.add_argument("--lr", type=float, default=2e-4)
  ap.add_argument("--every", type=int, default=500)
  ap.add_argument("--curve", type=int, default=300)
  ap.add_argument("--final", type=int, default=1500)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--tag")
  ap.add_argument("--run"); ap.add_argument("--ckpt", default="final")
  ap.add_argument("--n", type=int, default=500)
  ap.add_argument("--free", type=int, default=20)
  ap.add_argument("--texts")
  ap.add_argument("--inert", type=float, default=0.0)  # weight of the plain-text KL; 0 = the first readers
  ap.add_argument("--inert-texts", type=int, default=2)
  a = ap.parse_args()
  {"corpus": corpus, "targets": targets, "train": train, "probe": probe, "ask": ask, "report": report}[a.cmd](a)
