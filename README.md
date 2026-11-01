# qwen3-classifier

Can a language model say what its own next-token distribution is? A small LoRA on Qwen3-8B-Base learns to write, after a marker, its top three next tokens and how sure it is of the first, and it gets the *second* choice exactly right about half the time. The same training aimed at a different model's distribution does no better than that other model's raw logits. The short version, with pictures: [thread/](thread/README.md).

| Held-out positions, n=1500 | exact second choice | second in its top 3 | top-3 overlap | first choice | confidence r |
|---|---|---|---|---|---|
| Qwen3-8B on itself, rank 1 | 53.4% | 83.7% | 77.3% | 79.9% | 0.64 |
| Qwen3-8B on itself, rank 4 | 52.7% | 85.0% | 77.7% | 78.9% | 0.67 |
| Qwen3-8B on itself, rank 16 | 55.5% | 87.0% | 79.0% | 80.2% | 0.70 |
| Gemma 3 4B on itself, rank 4 | 55.2% | 87.1% | 77.7% | 80.5% | 0.66 |
| Gemma trained on Qwen's | 30.4% | 65.3% | 61.0% | 57.7% | 0.60 |
| Qwen trained on Gemma's | 30.3% | 62.9% | 60.8% | 58.9% | 0.55 |
| No training: one model's logits as a guess at the other's | 30.7% | 62% | 61.6% | 54.2% | 0.59 |

Where the two models' second choices differ, the gap is wider: 47.7% and 51.7% on themselves, 19.7% and 16.9% on each other.

## How it works

Positions are word boundaries in human-written text (wikitext-103, oasst1's non-synthetic conversations, the CPython standard library), kept only where neither model puts more than 90% on its top token, since otherwise there is no runner-up worth naming: 16,410 for training, 1,893 held out, split by document. A reply looks like:

```
After three days of deliberation, the jury found the defendant⟦?⟧ " not" " guilty" " Gu" =3
```

The digit is ten times the first choice's probability, rounded down. Some more, on sentences it never saw, real odds beside what it wrote:

| Text | Real odds | It says |
|---|---|---|
| My favorite animal is a | dog 17%, cat 12%, panda 10% | `" dog" " cat" " panda" =1` |
| The coin spun in the air for what felt like forever, and finally landed on | the 42%, its 26%, heads 16% | `" the" " its" " heads" =2` |
| She opened the envelope with shaking hands. Her application had been | accepted 21%, rejected 16%, ______ 9% | `" accepted" " rejected" " successful" =2` |
| He looked at the two wires, red and blue, and with the timer at three seconds he cut the | red 54%, blue 15%, wires 9% | `" red" " blue" " wire" =2` |
| After three days of deliberation, the jury found the defendant | guilty 48%, not 23%, comma 11% | `" not" " guilty" " Gu" =3` |

The last one swaps its top two. All eleven are in [results/hook-r4.txt](results/hook-r4.txt).
 The text goes through the frozen model with the adapter off; the adapter is on only for the marker and the answer. Targets are the frozen model's logits. 2,000 steps of 8 examples on one consumer GPU took under an hour for each run, and the curve is most of the way up by step 200, about 1,600 examples.

## What it does and does not show

The model's output layer is already a perfect top-k classifier of this distribution. So this does not uncover anything hidden: it measures what it costs to route something the network already computes into words. The cost is real: first choice is right about 80% of the time, not 100%.

The gap between self and other is large, in line with [Binder et al.](https://arxiv.org/abs/2410.13787) and against the negative result of [Song, Hu and Mahowald](https://arxiv.org/abs/2503.07513). It need not mean introspection in any rich sense: [Singh, Linzen and Ravfogel](https://arxiv.org/abs/2605.26242) argue such self-advantages can come from the architecture alone, and a LoRA on the model's own weights has an easy route to its own output head that a LoRA on another model does not. We score disagreement positions separately because of [Masked by Consensus](https://arxiv.org/abs/2604.12373).

Other limits:

- **Rank barely matters.** Rank 1 is about as good as rank 16 at naming, which suggests little new is being learned beyond a route.
- **Outside what it was trained on, it is worse.** At positions over 90% sure, never seen in training, the first choice is 91% right, but the second falls to 36% and confidence falls to r = 0.32.
- **Copying is possible.** It often repeats a near-form or a word from the text (" Gu", " wire"). Exact-match scores count those as wrong.

## Does the report follow the odds?

The text unchanged, a small push on the residual at the last token (layer 20, typically 2% of its size) changes the model's first choice; the reader then reads that same pushed state. `steer.py`, 398 held-out positions, rank 4 with the fix.

| | first choice named correctly | second choice named correctly |
|---|---|---|
| No push | 80% | 55% |
| Pushed up the second choice's gradient, until it leads | 65% | 54% |
| ...if the report had stayed as it was | 20% | 19% |
| Pushed down the first choice's gradient, until something else leads | 55% | 42% |
| ...if the report had stayed as it was | 22% | 24% |
| Control: a random push of the same size | moved the true first choice 8 times of 398, the report 5-7 | |

The report follows, so it is reading something tied to the pushed state and not only the words of the text. The second row could be a word being primed, as the push raises that word; the fourth raises nothing on purpose, which is why it is there, and it follows nearly as well. It does not settle the matter. Pushing a first choice down mostly hands the lead to the old second (289 of 398), and where the new leader is someone else, the report is barely better than not moving (43% against 35%, n=109). About a third of reports stay stale, and the confidence digit hardly follows the change (r = 0.14).

## The side effect, and the fix

Trained only on reports, the adapter leaks into ordinary text, because nothing tells it not to. Plain text read with the rank-4 adapter on differs from the frozen model by 0.64 bits (KL), keeps the top token 76% of the time, and free continuations start quoting words and adding digits: `a " lost" film" =2`. Perplexity on held-out wiki text doubles, from 8.79 to 17.40 (rank 16: 27.91).

Adding `--inert 1`, a KL term that holds plain text to the frozen model's distribution, removes almost all of it at no cost to the reports: KL 0.009 bits, top token kept 95%, perplexity 8.86, exact second choice 54.6%. Small code tasks go 18/20 untrained, 17/20 with the fix.

## Adapters

| Adapter | What | Get it |
|---|---|---|
| [qwen3-8b-classifier](https://huggingface.co/mahounairo/qwen3-8b-classifier) | rank 4, with the fix | `hf download mahounairo/qwen3-8b-classifier --local-dir classifier` |
| [qwen3-8b-classifier-r1](https://huggingface.co/mahounairo/qwen3-8b-classifier-r1) | rank 1, with the fix | `hf download mahounairo/qwen3-8b-classifier-r1 --local-dir classifier-r1` |

Both go on [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base). Run the text with the adapter off, then the marker `⟦?⟧` and the answer with it on; `ask` in `runnerup.py` does this.

## Reproducing

```
python -I runnerup.py corpus
python -I runnerup.py targets --model qwen
python -I runnerup.py train --reader qwen --target qwen --rank 4 --inert 1
python -I runnerup.py probe --run qwen-qwen-r4-inert
python -I runnerup.py report
```

`results/` holds each run's curve and probe, the report, and `hook-r4.txt`, the true top 5 beside what the reader said on eleven new sentences.
