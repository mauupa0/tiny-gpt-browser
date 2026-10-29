# Tiny GPT

A 1.8 million parameter character-level transformer, trained from scratch on Pan Tadeusz and run
entirely in the browser. Training is PyTorch on CPU. Inference is hand-written TypeScript with no ML
library: embeddings, LayerNorm, causal attention with a KV cache, GELU MLP, temperature and top-k
sampling, all in a Web Worker.

Live: https://tiny-gpt-oktawian.vercel.app

The page is built so that one letter of generated verse can be opened up. Everything it shows is
computed by the model running in the visitor's tab.

## What is on the page

1. **Writing.** The model writes verse as soon as the weights are in. Each written letter carries an
   underline when the model picked it with under 25% probability. The eight candidates it weighed for the
   next letter are drawn live, and the letter the poem actually has is marked. Opening line,
   temperature and top-k are editable. Click any letter to inspect it.
2. **Inside.** An anatomy plate for the inspected letter: the letter and position embeddings, then
   for each of the 4 blocks the attention of all 4 heads over the context, the 768 MLP neurons, the
   residual stream norm, the *logit lens* (the stream read back through the final LayerNorm and the tied
   output table after every attention and MLP step), and the exact *direct logit attribution* of every
   head and MLP to the explained letter. The sum of all pushes equals the logit.
3. **Heads.** All 16 attention maps for the same context, with a crosshair that follows the pointer
   across every head, plus each head's measured habit on held-out text (how far back it looks, how much
   the loss rises when it is zeroed).
4. **Temperature.** Three writers, one opening line, one stream of random numbers. Live uncertainty
   per letter and the share of repeated 4-letter runs.
5. **Break it.** Two classic bugs (no 1/sqrt(d) scaling, attention that reads one slot ahead) and a head
   silencer, switchable while the model writes next to the intact one. The parity check against PyTorch
   runs live on the page and puts each variant on a log axis against the 1e-3 limit.
6. **Evidence.** Training curve you can scrub, parameter ledger, test list, and a benchmark button.

## Layout

- `data/corpus.txt` cleaned training text (verse only). `data/raw.txt` is not shipped: download it as described in "Source text and license" before running `prep.py`.
- `train/` `prep.py` (cleaning), `model.py` (GPT), `train.py`, `export.py` (weights, reference logits, stats).
- `web/` Vite, React 19, TypeScript strict.
  - `src/model/` inference. `gpt.ts` forward pass with an optional trace (every intermediate of one
    position, logit lens, attribution), `inspect.ts` incremental inspector over a reusable KV cache,
    `lanes.ts` several writers on shared random numbers, `parity.ts`, `stats.ts`.
  - `src/worker/` the Web Worker and its message protocol.
  - `src/state/store.ts` one reducer for the worker, `src/ui/` the sections, `src/lib/` pure helpers.
  - `scripts/` `e2e.mjs`, `bench.mjs`, `layout.mjs`, `og.mjs`, `shots.mjs`, `prerender.mjs`,
    `head-stats.ts`, `fault-loss.ts`, `collect-tests.mjs`, `make-parity-ref.mjs`.
  - `tests/` Vitest.

## Source text and license

Adam Mickiewicz, *Pan Tadeusz*, from Wolne Lektury (https://wolnelektury.pl/katalog/lektura/pan-tadeusz/),
downloaded as TXT from https://wolnelektury.pl/media/book/txt/pan-tadeusz.txt. The work is in the public
domain. Wolne Lektury's own editorial material (footnotes, licence footer, argument paragraphs,
headings) is stripped by `train/prep.py`; the training text is the verse only. Characters that occur
fewer than 20 times were dropped, which leaves a 71-character vocabulary. Dashes are normalised to a hyphen.
To rebuild the corpus, save that TXT as `data/raw.txt` (it is not shipped here, because it carries the
editorial notes, which are under the Free Art License 1.3) and run `prep.py`.

Fonts (SIL Open Font License 1.1), subset to the 71 letters plus basic Latin and self-hosted:
Newsreader, Schibsted Grotesk, Martian Mono.

## Train

```
cd train
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch numpy --index-url https://download.pytorch.org/whl/cpu
PYTHONUTF8=1 .venv/Scripts/python.exe prep.py
PYTHONUTF8=1 .venv/Scripts/python.exe train.py        # about 38 minutes on a Ryzen 5 7600X
PYTHONUTF8=1 .venv/Scripts/python.exe export.py best.pt
```

`export.py` writes `web/public/model/weights.f16.bin` and `manifest.json` (what the page loads),
`web/tests/fixtures/` (fp32 weights and PyTorch reference logits and attention maps, used only by tests)
and `web/src/generated/stats.json`. Three more generated files come from the model itself:

```
cd web
node scripts/make-parity-ref.mjs        # public/model/parity-verse.json, PyTorch logits for the live check
npx vite-node scripts/head-stats.ts     # src/generated/heads.json, what each head does on held-out text
npx vite-node scripts/fault-loss.ts     # src/generated/faults.json, loss of the broken variants
node scripts/collect-tests.mjs          # src/generated/tests.json, so the page quotes a measured test count
```

## Run

```
cd web
npm install
npm run dev          # http://localhost:5173
npm test             # parity, negative controls, inspector, lanes, helpers
npm run build        # tsc, vite build, then prerender of the first state into dist/index.html
node scripts/e2e.mjs <url> <outDir>   # needs a global playwright; 64 browser checks at 1440 and 360 px
node scripts/bench.mjs <url>          # letters per second
node scripts/layout.mjs <url> [runs]  # gaps under the poem, layout jumps, CLS and nav fit at 7 widths
node scripts/og.mjs <url> <out.png>   # the 1200x630 share image (web/public/og.png)
```

Deploy: any static host; `npm run build` in `web/` writes `web/dist`.

## Design decisions

**Look.** A cyanotype: the surface is the colour (ink blue, `oklch(0.26 0.085 262)`), text is warm
paper, and the only accent is a vermilion marker that means "this one": the chosen letter, the selected
head, the primary action. Attention and activations use one blue ramp that rises in lightness from the
ground to paper; signed values use marker orange against a cool pole with the panel colour at zero. The
blueprint grid is a static CSS background, so the page is still when nothing is happening. Text
contrast is checked in tests (body 13.6:1, faintest text above 4.5:1 on every surface).

**Type.** Newsreader for verse and prose (optical size axis on, so the 72 px title and the 17 px
captions get different cuts), Schibsted Grotesk for controls, Martian Mono for numbers only. No
uppercase mono labels, no numbered ornaments, no cards in cards.

**A poem that grows without moving the page.** The verse has no fixed length, and everything under it
(legend, candidate bars, buttons) has to stay put while it writes and sit close once it is done. A
`min-height` for the longest poem leaves a 100 px hole under a short one. So the page ends a fresh poem
itself: `hero.css` reserves `--rows` lines (5, or 8 on a phone) and `Hero.tsx` stops the worker at the first
line break once the poem fills them, cutting a verse that would run past back to the previous break. A blank
line is half a row. Measured by `scripts/layout.mjs` over 12 fast runs per width: no element under the poem
moves, and the legend sits 18 to 60 px under the verse on desktop (it was up to 136 px).

**Why no animation or 3D library.** Measured gzip sizes in this setup: Motion 30 KB with LazyMotion
(47 KB full), GSAP with ScrollTrigger 44 KB, three.js 130 KB, react-three-fiber 245 KB. The page needs
four kinds of motion: bars that grow from their baseline, the newest letter landing, a heat map redraw,
a dot sliding on an axis. All are `transform` or `opacity` CSS transitions, which are interruptible, run off the
main thread, and are switched off under `prefers-reduced-motion`. A 3D stack would have
cost more than the whole rest of the page and shown less than the plate does, because the plate carries
real numbers for one letter. Heat maps are canvas (`ImageData` for the 16 attention maps, `fillRect`
for strips), which keeps a redraw under a millisecond.

**Why the second chapter is a figure and not a scroll story.** A scroll story hides the numbers
until you reach them and does not survive a phone. The plate is one tall figure with a sticky
context line, margin notes written from the numbers on screen, and a stepper plus arrow keys to walk
through the poem.

**What is real.** The attention strips, maps, MLP grids, lens lists, push bars, loss and entropy are
computed from the running model. The only precomputed inputs are labelled as such: the parity
reference (PyTorch logits for one verse line, fetched when the check runs), the head profiles and the
fault losses (measured offline on the held-out tenth of the poem by the scripts above), the training curve.

**Performance.** The first state is rendered to HTML at build time (Vite SSR loader, `react-dom/server`)
and hydrated, so the text paints before any script runs. Fonts are one subset file each (96 KB for all
three, down from 310 KB), with metric-matched fallbacks so the swap does not move text. The weights
(3.6 MB) are fetched by the worker after first paint.

If the app script fails to load, a tiny inline script retries it once under another URL and then says
so on the static page. A failed weights download, a failed parity file and a failed worker each show a
message instead of a spinner.

## Measured

Everything below was measured on a Windows desktop (Ryzen 5 7600X), headless Chromium, production
build, unless stated.

| What | Result |
| --- | --- |
| Lighthouse mobile, local server with brotli, 3 runs | 99 performance, 100 accessibility, 100 best practices, 100 SEO. FCP 1.4 s, LCP 2.1 s (simulated), TBT 0 ms, CLS 0 to 0.02 |
| Lighthouse mobile, live URL, 5 runs | 99, 99, 99, 97 and 89, other categories 100 in all. The 89 is a Speed Index of 22 s: the typewriter hero was still writing when that capture ended, so visual progress was still changing. FCP, LCP, TBT and CLS were normal in the same run. The 97 had 160 ms of blocking time |
| JavaScript | 259 KB raw, 83.4 KB gzip (React 19 included), worker 13 KB, CSS 6.7 KB gzip |
| Weights | 3.6 MB fp16, fetched after first paint |
| Speed | 361 letters per second, median of 7 runs, model and sampling, full speed (a later run with other work on the machine gave 347) |
| Parity with PyTorch | largest logit difference 5.7e-6 (fp32) and 6.7e-6 (fp16 shipped), limit 1e-3 |
| Broken variants against the same check | 7.2 (no 1/sqrt(d)) and 3.2 (attention sees the future) |
| Held-out loss, 48 windows | intact 1.620, no scaling 1.866 (+0.246), mask leak 1.624 (+0.004) |
| Most important head | 2.4, a word-boundary head (78% of its attention on spaces and line breaks), zeroing it costs 0.089 nats per letter, 5.3 times the next head |
| Tests | 56 Vitest tests, 64 browser checks (`scripts/e2e.mjs`) at 1440 and 360 px, run against the live URL as well |

Measured by `scripts/head-stats.ts`: heads 1.2 and 1.4 look at the previous letter 97% and 95% of the time;
heads 1.1, 1.3 and 3.1 at the last two letters; head 2.2 mostly beyond 32 letters back.

## Tests

`tests/parity.test.ts` runs the TypeScript forward pass token by token with the KV cache on four fixed
inputs (1, 21, 50 and 128 characters) and compares logits at every position, plus all attention maps of
the 50-character input, with PyTorch. Limit: max absolute difference below 1e-3, for the fp32 weights and
for the shipped fp16 weights (against a PyTorch reference computed from the same fp16-rounded weights).
Two negative controls inject a bug (no 1/sqrt(d) scaling, a query reading one slot past the causal mask)
and must exceed the limit.

`tests/inspect.test.ts`: the traced step equals the plain step; the logit lens at the last stage is the real
prediction (and the first stage is not); the residual stream equals the sum of what each part wrote (and
leaving a head out breaks it); logit attributions add up to the logit for the favourite and for other
letters (and leaving out the largest push breaks it); a reused cache gives bit-identical results to a
fresh one over a scrubbing sequence (and a cache that lies about its contents is caught by the same
comparison); head ablation leaves the cache and the model as they were, and an empty mask changes nothing.

`tests/lanes.test.ts`: lanes with equal settings write equal text, another seed or a broken model does not,
a cold writer repeats itself more than a hot one. `tests/lib.test.ts`: CSS colour tokens equal canvas
colours, contrast thresholds, head roles from measurements, parameter count equal to PyTorch's, captions
equal to the raw arrays, the fault losses quoted on the page re-measured.

## References and what was taken from them

- Brendan Bycroft, LLM Visualization, https://bbycroft.net/llm. One narrated beat per tensor, real
  numbers on hover, and a consistent colour rule. Not taken: the 3D engine (about 200 hours of work, desktop only).
- Polo Club, Transformer Explainer, https://poloclub.github.io/transformer-explainer/. A sampling panel
  that visibly reshapes the bars, layernorm and residuals kept quiet.
- nostalgebraist, interpreting GPT: the logit lens, https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens.
  Reading the stream through the output table after every step. With tied embeddings the first
  stage reads back the input letter, which the plate points out.
- Anthropic, Transformer Circuits, https://transformer-circuits.pub/. The residual stream as a bus that
  heads and MLPs add to; direct logit attribution. The attribution here is exact because the final
  LayerNorm is frozen at the position's own statistics.
- CircuitsVis, https://github.com/TransformerLensOrg/CircuitsVis, and Distill, Building Blocks,
  https://distill.pub/2018/building-blocks/. Attention as a masked triangle per head, linked views.
- Jay Alammar, The Illustrated GPT-2, https://jalammar.github.io/illustrated-gpt2/. Walking the causal
  triangle one row at a time.
- Andrej Karpathy, nanoGPT, https://github.com/karpathy/nanoGPT. The scale of the thing: a model small
  enough to read in an afternoon.
- Art direction: Anthropic's research pages and The Shape of Intelligence (https://shapeofintelligence.com/)
  for large type over a quiet field and a small readout, Circuit Tracing figures for text chips
  as highlights.

## License

Code: MIT, see `LICENSE`. Training text and fonts: see "Source text and license" above.
