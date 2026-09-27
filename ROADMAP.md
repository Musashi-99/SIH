# SatQuery AI — Remaining Work Roadmap (PS 26167)

> Companion to [`README.md`](README.md) (status) and [`resource.md`](resource.md)
> (gap tracker). This file is the **execution manual**: for each remaining
> item — who touches what, where to go, what to download, exact commands,
> and where the output plugs into the existing codebase. Written so any
> teammate can pick up one section and work it independently without asking
> "where do I even start."

**How to use this file:** find your assigned section below, follow it
top-to-bottom. Each section is self-contained — you don't need the other
sections' context to start. When you finish a step, flip its checkbox and
update the matching row in `resource.md` §3B (❌ → ⚠️ → ✅) with your
PR/commit link.

---

## 0. Shared setup (do this once, whoever starts first)

All ML sections below need a Python ML stack that isn't in `requirements.txt`
yet (today it's classical CV only: numpy/opencv/Pillow/rasterio). Add a
**separate** `requirements-ml.txt` so the base app still installs without a
GPU/heavy deps:

```bash
# requirements-ml.txt
torch>=2.2
torchvision>=0.17
torchgeo>=0.5
transformers>=4.38
huggingface_hub>=0.20
scikit-learn>=1.4   # already used classically, kept for probe heads
```

```bash
pip install -r requirements.txt -r requirements-ml.txt
```

Work directory convention: put all training code under a new top-level
`training/` folder (not `core/`) — `core/` stays the deterministic runtime
pipeline; `training/` is offline, run once to produce a weights file that
`core/` then loads at inference time. This mirrors how `eval/` is already
separate from `core/`.

```
training/
  bigearthnet/        §1 below
  grounding/           §2 below
  vqa/                 §3 below
  change_vqa/          §4 below
  fusion/              §5 below
```

Each trained model gets a **model card** (`training/<x>/MODEL_CARD.md`):
what it was trained on, architecture, metric, limitations, weight file
location/hash. Judges explicitly want this — don't skip it.

Compute note: none of this needs a beefy GPU. A linear-probe fine-tune on a
frozen backbone runs fine on a free-tier Colab T4, or even CPU for a
few-thousand-sample subset (slower, still hours not days). Use
[Google Colab](https://colab.research.google.com/) if no local GPU.

---

## 1. BigEarthNet-adapted vision component — **THE mandatory blocker**

**Owner note:** do this first — items 2-4 literally bolt onto this backbone.

### 1.1 What "done" looks like
A saved weights file (`.pt`/`.safetensors`) for an encoder that was
pretrained on BigEarthNet or fine-tuned on it, wired into `core/landcover.py`
or `core/query.py` as an *additional* labelled source (the classical k-means
pipeline keeps running too — PS rewards the agent *choosing between* tools).

### 1.2 Where to go / what to download
- **Library:** [TorchGeo](https://github.com/microsoft/torchgeo) — has
  `torchgeo.datamodules.BigEarthNetDataModule` and pretrained encoders in
  `torchgeo.models` (e.g. `resnet50` weights pretrained via SSL4EO on
  Sentinel-2, and native BigEarthNet checkpoints).
- **Dataset:** BigEarthNet-S2 (v1 or v2) — download via TorchGeo's dataset
  class itself (`torchgeo.datasets.BigEarthNet(root=..., download=True)`) —
  it handles the fetch from the official mirror. Full set is ~66GB; for a
  first pass use `bands="s2"` only and cap to a subset (TorchGeo supports a
  `num_classes=19` BigEarthNet-19 label scheme — use that, it's the standard
  benchmark split).
- Alternative smaller shortcut if download time is tight: **IBM/NASA
  Prithvi-100M** on Hugging Face
  (`ibm-nasa-geospatial/Prithvi-100M`) — pull via `huggingface_hub`, skip the
  raw BigEarthNet download, fine-tune its head on a small BigEarthNet subset
  instead (fewer images needed since the backbone is already RS-pretrained).

### 1.3 Step-by-step
```bash
mkdir -p training/bigearthnet && cd training/bigearthnet
```
1. **Get the data** (script: `training/bigearthnet/prepare_data.py`):
   ```python
   from torchgeo.datasets import BigEarthNet
   ds = BigEarthNet(root="./data/bigearthnet", split="train",
                     bands="s2", num_classes=19, download=True)
   ```
2. **Load a pretrained encoder** (script: `training/bigearthnet/train.py`):
   ```python
   from torchgeo.models import resnet50, ResNet50_Weights
   model = resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO)
   for p in model.parameters():
       p.requires_grad = False        # freeze backbone
   import torch.nn as nn
   model.fc = nn.Linear(model.fc.in_features, 19)   # linear probe head
   ```
3. **Fine-tune only the new `fc` head** on BigEarthNet-19 multilabel targets
   (`BCEWithLogitsLoss`, since BigEarthNet is multi-label not single-class).
   A few epochs over even a 5-10k image subset is enough for a linear probe
   to converge — this is the "hours not days" part.
4. **Evaluate**: report per-label F1 / micro-F1 on the held-out BigEarthNet
   test split — this number goes in the model card.
5. **Export**: `torch.save(model.state_dict(), "weights/bigearthnet_resnet50_probe_v1.pt")`.
6. **Write** `training/bigearthnet/MODEL_CARD.md`: architecture (ResNet50 +
   linear head), pretraining (SSL4EO Sentinel-2 MoCo), fine-tune data
   (BigEarthNet-19 subset, N images), metric (micro-F1 = X), limitations
   (Sentinel-2 bands only, not tested on Cartosat-2S).

### 1.4 Wiring into the running app
- New file `core/vision_adapter.py`: loads the `.pt` weights once at app
  startup (lazy import, same pattern as `core/ai_reasoning.py`'s Ollama
  stub — degrade gracefully if `torch` isn't installed), exposes
  `predict_labels(image) -> {label: confidence}`.
- Register it in `registry.yaml` as a new tool:
  ```yaml
  - id: vision.bigearthnet_classify
    name: BigEarthNet ResNet50 Linear Probe
    version: v1
    type: learned
    modality: optical
    preconditions: ["sentinel2_like_bands"]
  ```
- In `core/query.py`'s `answer()` (around the DESCRIBE/land-cover branch,
  `core/landcover.py` integration point), call the adapter *alongside* the
  classical k-means result, and surface both in the evidence trail tagged by
  `registry.py`-resolved tool id — this is what makes it "agentic selection"
  rather than a silent replacement.
- Add `tests/test_vision_adapter.py`: assert the adapter loads, returns a
  dict of 19 BigEarthNet label confidences on a sample image, and that
  `core/query.py` includes it in the trace when available.

### 1.5 Time estimate
~1 focused day: data prep 1-2h, training loop + convergence 2-4h (mostly
unattended), integration + tests 2-3h.

- [ ] Data downloaded via TorchGeo
- [ ] Linear probe fine-tuned, metric recorded
- [ ] Weights + model card committed
- [ ] `core/vision_adapter.py` wired + registered in `registry.yaml`
- [ ] Tests passing, `resource.md` row flipped to ✅

---

## 2. Learned grounding (GroundingDINO / SAM)

### 2.1 What "done" looks like
LOCATE-intent queries (`core/query.py`'s LOCATE branch, `core/render.py`'s
`mask_to_boxes()`) get a second, learned box source alongside the existing
connected-components boxes, selectable/comparable in the trace.

### 2.2 Where to go
- **[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)** — text
  query + image → boxes, open-vocabulary. This is the direct fit because our
  LOCATE intent is already "text description → region," same shape as
  GroundingDINO's API.
- Pretrained checkpoint: `groundingdino_swint_ogc.pth` from their repo's
  release page — **no training needed to get value #1** (zero-shot use is
  legitimate as "an adapted/selected specialist tool," training-free
  open-vocabulary detection is the intended use case of that model).
- Optional upgrade: fine-tune on **VRSBench** grounding annotations (has
  RS-specific referring expressions) if zero-shot RS performance is weak —
  VRSBench ships exactly the (image, text, box) triples needed.

### 2.3 Step-by-step
```bash
mkdir -p training/grounding && cd training/grounding
git clone https://github.com/IDEA-Research/GroundingDINO
pip install -e GroundingDINO
# download groundingdino_swint_ogc.pth + its config from their release page
```
1. **Zero-shot smoke test first** — run their demo script with a sample
   satellite crop + a text prompt like `"water body"` and confirm boxes come
   back sane before touching training code.
2. **If fine-tuning** (only if zero-shot boxes are clearly off on RS
   imagery): download VRSBench's grounding split (see §6 below — same
   download you need for eval anyway), fine-tune GroundingDINO's detection
   head following their repo's fine-tune script, few epochs.
3. **Export** whichever checkpoint you're using (`groundingdino_rs_v1.pth`)
   into `weights/`.

### 2.4 Wiring into the running app
- New file `core/grounding_adapter.py`: `predict_boxes(image, text_query) ->
  [{box, score}]`, same lazy-import-degrades-gracefully pattern.
- Registry entry:
  ```yaml
  - id: grounding.groundingdino
    name: GroundingDINO
    version: swint-ogc
    type: learned
    modality: optical
  ```
- In `core/query.py`'s LOCATE branch: try the learned adapter first if
  loaded, fall back to the existing `mask_to_boxes()` connected-components
  path if the model isn't available or returns nothing — keep both paths
  live, log which one fired in the trace (this "agent chose between tools"
  behavior is the same pattern as §1).
- `app.py`'s overlay renderer already draws per-instance boxes (added this
  session for connected components) — reuse the same box-drawing code path
  for the learned boxes, just a different evidence source tag.

### 2.5 Time estimate
Zero-shot integration: ~half a day (mostly getting the checkpoint running
+ wiring). Fine-tune upgrade: add another half-day only if zero-shot quality
is insufficient on the demo samples.

- [ ] GroundingDINO zero-shot verified on sample imagery
- [ ] `core/grounding_adapter.py` wired, registered, tested
- [ ] (optional) fine-tuned on VRSBench grounding split
- [ ] `resource.md` row flipped

---

## 3. Learned VQA (RSVQA / VRSBench)

### 3.1 What "done" looks like
A learned model answers free-form questions about a single image,
registered as a `type: learned` tool that `core/query.py` can pick
alongside the existing lexicon/rule VQA path.

### 3.2 Where to go
- **Fastest path:** don't build a VQA architecture from scratch — fine-tune
  a small existing VLM with **LoRA** (parameter-efficient, cheap):
  - **[LLaVA](https://github.com/haotian-liu/LLaVA)** (or a smaller variant
    like LLaVA-1.5-7B) fine-tuned with a LoRA adapter on RSVQA question/answer
    pairs — use HuggingFace's `peft` library for the LoRA part, far cheaper
    than full fine-tune.
  - Simpler/cheaper alternative if LLaVA is too heavy for available compute:
    a **CLIP linear-probe classifier** — encode image with CLIP, encode
    question with CLIP text encoder, concatenate, train a small classifier
    head over RSVQA's fixed answer vocabulary (RSVQA answers are mostly
    yes/no/counts/land-cover words — small closed vocabulary, this works
    surprisingly well and trains in under an hour).
- **Dataset:** [RSVQA](https://github.com/syvlo/RSVQA) — LR (Low Res, easier,
  smaller) and HR (High Res) splits. Start with LR to get the pipeline
  working, then HR.

### 3.3 Step-by-step (CLIP linear-probe path — recommended first pass)
```bash
mkdir -p training/vqa && cd training/vqa
git clone https://github.com/syvlo/RSVQA   # or their HF mirror if listed
pip install open_clip_torch
```
1. Download RSVQA-LR images + question/answer JSON.
2. Encode every image once with CLIP (`open_clip`'s `ViT-B-32` pretrained
   `openai` weights) → cache embeddings (avoids re-encoding every epoch).
3. Encode each question with the same CLIP text tower.
4. Concatenate `[image_embed; text_embed]` → small MLP → softmax over
   RSVQA's answer vocabulary (build the vocab from the training split's
   unique answers).
5. Train the MLP only (CLIP stays frozen) — cross-entropy loss, few epochs,
   runs on CPU for LR split in well under an hour.
6. Evaluate: accuracy per question-type (RSVQA's own eval script breaks
   this down by yes/no vs count vs land-cover — report all).
7. Export `weights/rsvqa_clip_probe_v1.pt` + the answer vocab list.

### 3.4 Wiring into the running app
- New file `core/vqa_adapter.py`: `answer_question(image, question_text) ->
  {answer, confidence}`.
- Registry entry: `vqa.clip_probe_rsvqa`, `type: learned`.
- In `core/query.py`'s `answer()` dispatcher: when intent resolves to a
  generic VQA-shaped question the rule engine can't confidently classify,
  route to the learned adapter as a fallback/second-opinion tool, log both
  the rule-answer and the learned-answer with their confidences in the
  evidence trail if both fire — again, "agent picks/compares tools," not
  "replace the honest classical system."

### 3.5 Time estimate
CLIP linear-probe path: ~1 day including data download. LLaVA-LoRA path (if
you want stronger free-form answers later): 2-3 days, needs a GPU with
≥16GB VRAM.

- [ ] RSVQA-LR downloaded
- [ ] CLIP linear-probe trained, per-type accuracy recorded
- [ ] `core/vqa_adapter.py` wired + registered
- [ ] `resource.md` row flipped

---

## 4. Learned change-VQA (CDVQA-style)

### 4.1 What "done" looks like
A change-captioning/change-VQA head that takes the bi-temporal features
`core/change.py` already computes (CVA deltas, transitions, hotspots) and
produces a learned natural-language change description, alongside the
existing template-based one.

### 4.2 Where to go
- **[CDVQA / ChangeQA](https://github.com/shizenglin/ChangeQA)** — reuse
  their head architecture (typically: two image encoders → difference
  features → small transformer/RNN decoder → answer) rather than designing
  one from scratch.
- **Pairing datasets:** **LEVIR-CD** and **WHU-CD** — bi-temporal building
  change datasets with ground-truth change masks; CDVQA itself is built on
  top of similar bi-temporal pairs with QA annotations.

### 4.3 Step-by-step
```bash
mkdir -p training/change_vqa && cd training/change_vqa
git clone https://github.com/shizenglin/ChangeQA
```
1. Download LEVIR-CD (smaller, faster to iterate) — image pairs + binary
   change masks + (if available) QA annotations from CDVQA's release.
2. **Reuse, don't reinvent:** feed `core/change.py`'s existing CVA
   delta-feature computation as the "difference features" input to
   ChangeQA's decoder head, instead of re-implementing their full encoder —
   this is the fast path since our classical pipeline already computes a
   solid change-feature representation.
3. Train only the decoder head (small transformer/RNN) on CDVQA's QA pairs
   over these features — this is a much smaller training job than the full
   ChangeQA pipeline since the hard feature-extraction part is already
   classical + fast.
4. Evaluate against CDVQA's own metric (typically accuracy per question
   type: increase/decrease/unchanged, count-based questions).
5. Export `weights/change_vqa_head_v1.pt`.

### 4.4 Wiring into the running app
- New file `core/change_vqa_adapter.py`: takes `core/change.py`'s existing
  delta/transition feature dict, returns a learned change description +
  confidence.
- Registry entry: `change.cdvqa_head`, `type: learned`, modality: `any`
  (works on optical or SAR bi-temporal pairs), precondition: `two_images`.
- `app.py`'s change-intent branch: call this adapter alongside the existing
  templated change narrative (`core/change.py` → `core/ai_reasoning.py`
  template composer), surface both.

### 4.5 Time estimate
~1.5-2 days — most of the time is data wrangling LEVIR-CD/CDVQA's format
into what the reused decoder expects; the actual training is small once
features are already computed classically.

- [ ] LEVIR-CD / CDVQA data downloaded
- [ ] Decoder head trained on top of `core/change.py` features
- [ ] `core/change_vqa_adapter.py` wired + registered
- [ ] `resource.md` row flipped

---

## 5. Real optical–SAR fusion (upgrade from rule-based to learned)

### 5.1 What exists today (don't redo this part)
`core/sar.py` already has: log/dB backscatter stretch, speckle filtering,
and a rule-based joint mask (water = low backscatter + low optical texture,
built-up = high backscatter + high optical texture), registered in
`registry.yaml` as `sar.preprocess` / `fusion.joint_mask`. **Keep this
running** — it's the honest classical fallback the agent can still pick.

### 5.2 What "done" looks like
A learned fusion head (feature concatenation or small cross-attention block)
trained on real co-registered optical+SAR pairs, replacing/augmenting the
rule-based mask with a data-driven one.

### 5.3 Where to go
- **Reuse §1's dataset:** BigEarthNet ships **paired S1 (SAR) + S2
  (optical)** patches for the *same* dataset — this is the key shortcut:
  the BigEarthNet adapter (§1) and this fusion head can share one download
  and one training pipeline, since TorchGeo's `BigEarthNet` dataset class
  already supports `bands="s1s2"` (both modalities together).

### 5.4 Step-by-step
```bash
mkdir -p training/fusion && cd training/fusion
```
1. Reload BigEarthNet via TorchGeo with `bands="s1s2"` (paired mode) instead
   of `bands="s2"` — same download infra as §1, just a different flag.
2. Build a small fusion architecture: two lightweight encoders (can literally
   reuse §1's frozen ResNet50 backbone for the optical branch + a small
   CNN for the 2-channel SAR branch) → concatenate features → small
   classifier head predicting the built-up/water/vegetation joint labels
   (use BigEarthNet's existing land-cover labels as the training target —
   no extra annotation needed).
3. Train the fusion head only (encoders frozen) — same "linear probe on top
   of frozen backbones" trick as §1, keeps this cheap.
4. Evaluate: compare against the existing rule-based mask on a held-out set
   — report where the learned fusion agrees/disagrees with the classical
   rule (this comparison itself is good judge-facing material: "we validated
   the learned fusion against the honest classical baseline").
5. Export `weights/fusion_head_v1.pt`.

### 5.5 Wiring into the running app
- Extend `core/sar.py` with a `learned_fusion_mask(optical, sar)` function
  that loads the head lazily (same graceful-degrade pattern).
- New registry entry: `fusion.learned_head_v1`, `type: learned`, modality:
  `cross_modal` (needs both an optical and SAR input — reuse the existing
  precondition-checking pattern in `core/registry.py` that already gates
  `sar.preprocess`).
- `app.py`'s optical-SAR query path: when both are available, run the rule
  mask (existing) *and* the learned mask (new), show both, tag them
  distinctly in the audit trace — this directly answers the PS's "extract
  complementary info from co-registered optical+SAR" requirement with two
  independently-verifiable methods, which is stronger than either alone.

### 5.6 Time estimate
~1 day if done right after §1 (shares the data pipeline and the frozen
encoders) — this is why §1 should genuinely go first.

- [ ] S1+S2 paired BigEarthNet data loaded via TorchGeo
- [ ] Fusion head trained on frozen encoders, evaluated vs. rule baseline
- [ ] `core/sar.py` extended + registered
- [ ] `resource.md` row flipped

---

## 6. Real eval data (swap stand-in JSONL for real benchmark splits)

### 6.1 What exists today
`eval/vrsbench.py`, `eval/rsvqa.py`, `eval/cdvqa.py` + `eval/common.py` +
`eval/run_all.py` are real, working plumbing — they currently score against
14 made-up sample questions in `eval/data/*.jsonl`. The JSONL schema already
matches the real benchmarks exactly, so this is a **data swap, not new
code**.

### 6.2 Where to go
- **[VRSBench](https://github.com/lx-cpu/VRSBench)** — captioning +
  grounding + VQA splits.
- **RSVQA-LR/HR** — same source as §3 (you'll likely already have this
  downloaded if you did §3 first).
- **CDVQA** — same source as §4.

### 6.3 Step-by-step
1. Download each benchmark's real test split (not train — eval harness
   should score against held-out data) to `eval/data/real/<benchmark>/`.
2. Write a small converter script per benchmark
   (`eval/data/convert_vrsbench.py` etc.) that maps their native format to
   the existing JSONL schema `eval/vrsbench.py` already expects — check
   `eval/common.py` for the exact field names the runners read.
3. Point `eval/run_all.py`'s config at the new `eval/data/real/` paths
   (currently pointing at the stand-in files — likely a constant near the
   top of `eval/run_all.py` or a per-runner default path argument).
4. Re-run `python -m pytest tests/test_eval.py -q` — the smoke tests should
   still pass (they test the harness plumbing, not specific scores).
5. Hit `GET /api/eval` and confirm the returned scores now reflect the real
   split sizes (hundreds/thousands of items, not 14).
6. Update `README.md` §3.5 status line once this is done — this is the
   easiest ❌/⚠️ → ✅ flip in the whole roadmap since zero new code is
   needed, only data + a path change.

### 6.4 Time estimate
~2-4 hours — almost entirely download + format-conversion script writing,
no ML training involved.

- [ ] Real VRSBench/RSVQA/CDVQA splits downloaded
- [ ] Converter scripts written per benchmark
- [ ] `eval/run_all.py` pointed at real data
- [ ] `resource.md` row flipped

---

## 7. Sensor band tables (cheap, no ML — good first task for someone new)

### 7.1 What "done" looks like
`core/sensors.py`'s `BandMapping` registry has real, verified band
definitions (wavelength, resolution, band order) for Sentinel-1, Sentinel-2,
Cartosat-2S, and RISAT — today only `generic_rgb`/`rgb_nir` are verified.

### 7.2 Where to go
- **Sentinel-2:** [ESA Sentinel-2 User Guide — Spectral bands](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial)
  — 13 bands, wavelengths + resolutions are public and exact.
- **Sentinel-1:** [ESA Sentinel-1 User Guide](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar)
  — C-band SAR, VV/VH/HH/HV polarizations, no "wavelength bands" in the
  optical sense, but document the polarization channels the same way.
- **Cartosat-2S / RISAT:** [ISRO Bhuvan](https://bhuvan.nrsc.gov.in/) /
  [NRSC data product specs](https://www.nrsc.gov.in/) — public sensor
  specification PDFs list band count, spectral range, and GSD per product
  level.

### 7.3 Step-by-step
1. Open `core/sensors.py`, find the existing `BandMapping` dataclass/dict
   structure used for `generic_rgb`/`rgb_nir` — match that exact shape for
   the new entries.
2. Add one entry per sensor with real band count, order, wavelength ranges,
   native GSD. Cross-check each against at least two sources (the official
   ESA/ISRO doc + one secondary, e.g. a TorchGeo or GDAL sensor-definition
   file) before committing — these numbers matter for correctness.
3. Add a unit test per sensor in `tests/test_sensors.py` (create if it
   doesn't exist) asserting band count/order match the documented spec.
4. Wire `core/ingestion.py`'s sensor-detection logic (it currently defaults
   to `generic_rgb`) to actually pick the right `BandMapping` when GeoTIFF
   metadata or filename hints indicate a specific sensor.

### 7.4 Time estimate
~half a day — pure reference/lookup work, easiest task in this file, no
GPU/ML needed. Good task to hand to whoever has the least ML background.

- [ ] Sentinel-1 bands added + tested
- [ ] Sentinel-2 bands added + tested
- [ ] Cartosat-2S bands added + tested
- [ ] RISAT bands added + tested
- [ ] `resource.md` row flipped

---

## 8. Suggested team split (if working in parallel)

| Person | Section | Depends on | Standalone? |
|---|---|---|---|
| A | §1 BigEarthNet adapter | — | Yes — start immediately |
| B | §6 Real eval data + §7 Sensor tables | — | Yes — start immediately, no ML needed |
| C | §2 Learned grounding | Can start zero-shot immediately; fine-tune upgrade optional | Mostly yes |
| D | §3 Learned VQA | — | Yes — start immediately |
| A (after §1) | §5 Optical-SAR fusion | §1 (shares data/encoders) | No — sequence after §1 |
| Whoever's free | §4 Change-VQA | Benefits from §1's feature conventions but not blocking | Mostly yes |

Only real dependency: **§5 should start after §1** because it reuses the
frozen BigEarthNet encoder and the S1+S2 paired download. Everything else
can run in parallel from day one.

---

## 9. Definition of done for the whole roadmap

Cross-reference against `README.md` §6 table — every ❌/⚠️ row there should
read ✅ once §1-§7 above are complete, and `resource.md` §3B should have no
remaining ❌ rows. At that point the honest read becomes: *"strong classical
EO tool + real learned components on every mandatory PS item + legitimate
agent framework + real benchmark scores."*
