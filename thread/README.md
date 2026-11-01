# Thread

**1.** can a model predict its own token distribution? turns out, yes.

[1-what-it-says.png]

**2.** the format is a marker after the text, then its top 3 tokens and a confidence digit: `My favorite animal is a⟦?⟧ " dog" " cat" " panda" =1`, real odds: dog 17%, cat 12%, panda 10%.

[1b-more.png]

**3.** it only took ~1,600 examples and a rank-4 LoRA for Qwen3-8B to name its own runner-up token 45% of the time. by run 16,000 it levels off at ~54%. another model's logits only get 31%, basically guessing.

[2-examples.png]

**4.** is it reading itself, or just good at guessing what a model would say? same training, aimed at another model's distribution: Gemma on Qwen, Qwen on Gemma. those land at 30%, again, basically guessing.

[3-self-other.png]

**5.** the output layer already is a perfect classifier of these odds, so we're reinventing classifiers from first principles. what's measured here is the cost of routing that into words: about 80% on the first choice, not 100%. rank 1 does as well as rank 16.

**6.** does it affect performance? at first it seemed to: nothing tells it to leave normal text alone, and perplexity doubled. one extra KL term on plain text brought it right back to baseline, 8.8 to 8.9, and reports didn't get any worse.

[4-side-effect.png]

**7.** does it hold up when you start poking neurons? a 2% push on the residual swaps top two picks, and the report follows 65% of the time, versus 20% if it was ignored. push the leader down, similar result: 55% vs 22%.

[5-steering.png]

**8.** so it's reading something tied to its own state, not just the text. it's not perfect: a third of the reports stay stale, and the confidence digit barely moves. code, results and adapters: https://github.com/mahounairo/qwen3-classifier
