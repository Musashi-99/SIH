# SatQuery AI — PS 26167 (ISRO / Dept. of Space, SIH)

**Ask Questions. Explore Earth.**

An agentic vision-language assistant for satellite imagery. Upload one image
(or a pair — cross-modal optical+SAR, or bi-temporal before/after), ask a
question in plain English, and get an answer that is **measured from the
actual pixels**, not hallucinated by a language model — with a visible trail
of which tool/model produced it, at what confidence, and why.

This README is the single onboarding doc for the team: what the problem
statement demands, what's built, what just got pushed, what's still missing,
how we're closing the gap, and how the whole pipeline actually works —
explained twice, once in plain language and once in engineering terms.

> Deep-dive gap tracker lives in [`resource.md`](resource.md) — this README
> summarizes it for onboarding; `resource.md` is the file we keep updating as
> ❌ items flip to ✅.

---

## 1. The problem statement, in one paragraph

**PS 26167** asks for a web app that can look at satellite images and answer
questions about them the way a knowledgeable analyst would — describe what's
in a scene, point at a specific object, say what changed between two dates,
and combine an optical image with a SAR (radar) image of the same place to
pull out information neither one shows alone. Crucially, judges don't just
want *an* answer — they want to see **which model/tool the system chose, why,
and with what confidence**, i.e. it must behave like an *agent* orchestrating
specialist tools, not one big black-box model. There's also a hard technical
requirement: at least one vision component must be *adapted/fine-tuned* on
BigEarthNet (a real satellite-imagery training set) — a purely classical
(non-learned) computer-vision pipeline does not satisfy the PS on its own.

## 2. What's already solid (built before today)

This was a genuinely capable classical remote-sensing engine before this
session started:

| Capability | How | Where |
|---|---|---|
| Upload + validation | magic-byte checks, format/dims/bands/CRS report | `core/ingestion.py`, `core/validation.py` |
| Single-image Q&A | rule/lexicon query planner → describe/count/locate/area/spectral/spatial | `core/query.py` |
| Captioning | scene-type + composition + narrative generator | `core/query.py` |
| Grounding (basic) | class-mask / detection-box highlight for "show me the water" style asks | `core/render.py` |
| Spectral indices | NDVI/NDWI/SAVI/EVI/GNDVI (needs NIR) + RGB-only proxies (VARI/ExG) | `core/features.py` |
| Land cover | k-means + physics rules, adaptive k, confidence label | `core/landcover.py` |
| Object detection | vessels, bright targets, linear structures (roads/rivers), contour+Hough based | `core/objects.py` |
| Bi-temporal change | ORB/RANSAC + ECC image registration, change-vector analysis + Otsu threshold, severity, hotspots | `core/change.py` |
| Evidence + confidence | every answer cites `E#` evidence items with method + confidence | `core/evidence.py` |
| Reports / history | downloadable HTML audit report, reproducible runs (seeded) | `core/report.py`, `core/history.py` |
| Web UI | full workspace: viewer, layer toggles, chat, evidence trail, reports | `static/index.html` |

**In plain words:** think of this as a very good "photo analyst using math,
not AI" — it measures colors/edges/shapes with classical image-processing
algorithms (k-means clustering, edge detection, image alignment) and turns
those measurements into sentences. It's honest (never invents a number it
can't measure) but it isn't yet an *AI model* that learned to see from
training data, and it doesn't yet fuse optical + radar images together.

## 3. What I just built and pushed (this session)

Three of the seven gaps from `resource.md` §3B are now closed or scaffolded.
All of it is committed in this push:

### 3.1 Formal agent registry + audit trace (§3B item "Formal agent registry")
- **[`registry.yaml`](registry.yaml)** — the single source of truth for every
  tool the agent is allowed to pick: name, version, modality
  (optical/sar/any), fixed params, and preconditions (e.g. "needs a second
  image", "needs a NIR band").
- **[`core/registry.py`](core/registry.py)** — loads `registry.yaml`, resolves
  a tool id to its metadata, and checks compatibility (e.g. refuses to run a
  SAR-only tool on an optical-only upload) before the pipeline ever executes it.
- Query execution now returns an **auditable trace**: `{task, models[]
  (name+version), params, outputs, confidence}` — exactly the shape judges
  said they'd score, instead of just "some Python function ran."

**Plain words:** before, the code just called Python functions with no
names. Now every function has an ID card (name, version, what kind of image
it needs) in one YAML file, and the agent has to check that ID card before
using a tool — and shows the ID card in the answer. That's what "agentic
orchestration" means in judge-speak.

### 3.2 Optical–SAR fusion scaffolding (§3B item "Optical–SAR fusion — critical")
- **[`core/sar.py`](core/sar.py)** — SAR-specific preprocessing: log/dB
  backscatter stretch, 5×5 median speckle filtering (SAR images are
  inherently noisy — "speckle" — unlike optical), and a first
  **rule-based joint fusion mask**: pixels are called `water` when
  backscatter is low AND optical texture is low, `built_up` when backscatter
  is high AND optical texture is high. This is the first real "look at both
  images together" logic in the codebase.
- Registered in `registry.yaml` as `sar.preprocess` and `fusion.joint_mask`
  so it shows up in the audit trace like any other tool.
- **Status: scaffolding, not the full mandatory feature.** The PS wants this
  fused with a *learned* model eventually (see §5); right now it's a
  classical rule, which is enough to demo the *concept* end-to-end but not
  enough to fully satisfy "extract complementary info from co-registered
  optical+SAR" at a competitive level.

### 3.3 Grounding upgrade — real per-object boxes (§3B item "Learned RS-VQA + grounding")
- **[`core/render.py`](core/render.py)** now has `mask_to_boxes()` using
  OpenCV's `connectedComponentsWithStats` (8-connectivity) — instead of one
  big bounding box around "the water," LOCATE queries now return **one box
  per water body / built-up patch / vegetation patch**, sorted by area, with
  small-noise regions filtered out.
- `core/query.py`'s LOCATE intent and `app.py`'s overlay renderer were updated
  to carry and draw these per-instance boxes.
- **Status: classical-CV grounding is much better now (real per-instance
  boxes vs one blob), but it's still not a learned grounding model
  (GroundingDINO/SAM) — see §5.**

### 3.4 GeoTIFF hardening (§3B item "Sensor-accurate ingestion")
- `core/ingestion.py` now extracts real ground sample distance (GSD/pixel
  resolution) from GeoTIFF metadata through a fallback cascade:
  **rasterio → tifffile GeoTIFF tags (`ModelPixelScaleTag`) → user-entered
  value**, and records *which* of the three actually supplied the number
  (`gsd_source` field) — so a report never silently claims false precision.
- Added `rasterio>=1.3` and `tifffile>=2024.2.12` to `requirements.txt`
  (soft dependency — ingestion still degrades gracefully to Pillow-only tags
  if either import fails, so a broken install never blocks the app).
- Added a real `samples/geotiff_sample.tif` with proper
  `ModelPixelScaleTag`/`ModelTiepointTag`/`GDAL_NODATA` tags plus
  `tests/test_ingestion.py` (4 passing tests) proving the cascade works.

**Plain words:** satellite files (GeoTIFF) carry real-world scale info in
their metadata — e.g. "each pixel is 10 meters." Before, we mostly asked the
user to type that in by hand. Now the app reads it straight from the file
when possible, and is honest about where the number came from when it can't.

### 3.5 Offline evaluation harness (§3B item "Eval harness")
- **[`eval/`](eval/)** package: `vrsbench.py`, `rsvqa.py`, `cdvqa.py` runners
  + `common.py` shared metric utilities + `run_all.py` orchestrator.
- Local **stand-in JSONL splits** (`eval/data/*.jsonl`, 7+5+2 sample items)
  that match the real VRSBench/RSVQA/CDVQA schema exactly — this means once
  we download the real benchmark splits, we swap the file path and the
  scoring code needs zero changes.
- Wired to `GET /api/eval` (lazy-imported to avoid a circular dependency
  between `eval/` and `app.py`'s analysis pipeline).
- `tests/test_eval.py` — 3 smoke tests confirming the harness runs end to end.
- **Status: the harness (plumbing) is real and working. The *scores* it
  currently produces are meaningless** — they're measured against 14 made-up
  sample questions, not the actual VRSBench/RSVQA/CDVQA benchmark data. Next
  step is downloading real splits (see §5.6).

---

## 4. What's still missing vs. the problem statement

Straight from `resource.md` §3B, with current status after today's push:

| # | Requirement | Status | Why it matters for judging |
|---|---|---|---|
| 1 | **BigEarthNet-adapted vision component** | ❌ Still missing | This is the PS's hard mandatory line ("≥1 visual/VL component fine-tuned/adapted on BigEarthNet"). Nothing in the repo is a trained/adapted model yet — `requirements.txt` has no `torch`/`transformers`. This is the single biggest risk to disqualification. |
| 2 | **Learned VQA / captioning / grounding** (VRSBench, RSVQA) | ❌ Still missing | Current VQA/captioning/grounding is 100% deterministic rules over classical CV features — good demo, but judges will ask "where's the learned model." |
| 3 | **Learned change-VQA** (CDVQA-style) | ❌ Still missing | Change answers are templated from pixel-level deltas, not a trained change-captioning head. |
| 4 | **Optical–SAR fusion** | ⚠️ Partial (was ❌) | Rule-based fusion mask now exists (§3.2) — real progress, but not the learned/attention fusion the PS implies, and there's no true SAR reader yet (calibration from a real `.tif` SAR product, not just any second image). |
| 5 | **Formal agent registry + audit trace** | ✅ Done (was ⚠️) | Closed today — see §3.1. |
| 6 | **Sensor-accurate ingestion** (S1/S2, Cartosat-2S, RISAT band tables) | ⚠️ Partial (improved) | GSD-from-metadata is real now (§3.4); real multi-band S1/S2/Cartosat/RISAT band tables in `core/sensors.py` are still mostly placeholders — only generic RGB/RGB+NIR are verified. |
| 7 | **Eval harness** (public + ISRO benchmark scoring) | ⚠️ Partial (was ❌) | Plumbing done (§3.5), real benchmark data not yet loaded. |

**Bottom line honest read:** we went from *"strong classical demo, fails the
PS on the model-adaptation requirement"* to *"strong classical demo + real
agent framework + fusion/grounding/eval scaffolding, still fails the PS on
the model-adaptation requirement."* Item #1 (BigEarthNet adaptation) is the
one gap that actually blocks passing the mandatory functional scope — items
2–4 are downstream of it (once you have a real trained vision encoder, VQA
and grounding heads bolt onto it far more easily).

## 5. How we're going to close each gap (with open-source shortcuts)

Ordered by leverage — #1 first because everything else compounds on it.

### 5.1 BigEarthNet adaptation — the mandatory one
**Fastest legitimate path: don't train from scratch.** Use an existing
open-source BigEarthNet-pretrained encoder and adapt/fine-tune only its
final layer (a "linear probe") — this *is* fine-tuning/adaptation per the PS
wording, and takes hours not days on a single GPU (or even CPU for a small
subset).
- **[TorchGeo](https://github.com/microsoft/torchgeo)** — Microsoft's
  geospatial-ML library ships BigEarthNet dataloaders *and* several models
  pretrained on BigEarthNet out of the box (`torchgeo.models`,
  `torchgeo.datasets.BigEarthNet`). This is the single biggest shortcut
  available — it turns "train on BigEarthNet" into "load checkpoint, fine-tune
  head, save weights."
- **[BigEarthNet official repo](https://git.tu-berlin.de/rsim/bigearthnet-models-tf)**
  (also mirrored on GitHub) — baseline ResNet/VGG models already trained on
  BigEarthNet-19 labels, usable as a frozen encoder for a CLIP-style
  linear-probe adaptation.
- **[IBM/NASA Prithvi-100M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M)**
  — a geospatial foundation model (HLS satellite imagery) on Hugging Face;
  fine-tuning it on BigEarthNet labels is a defensible "adapted on
  BigEarthNet" story and gives a stronger backbone than training from scratch.
- **Deliverable to ship:** weights file + a model card (what it was trained
  on, accuracy, limitations) + the training script, wired into
  `core/landcover.py`/`core/query.py` as an additional labelled source
  (keep the classical pipeline running alongside it — the PS rewards
  agentic *selection between* tools, not replacing the classical one).

### 5.2 Learned grounding
- **[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)** or
  **[Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)**
  — both are open-weight, text-prompted or click-prompted grounding models
  that can replace/augment the current connected-components boxes (§3.3)
  with actual open-vocabulary detection. GroundingDINO is the more direct
  fit since it takes a *text query* + image and returns boxes — exactly our
  LOCATE intent's shape.

### 5.3 Learned VQA
- **[RSVQA](https://github.com/syvlo/RSVQA)** dataset + baseline model, or
  fine-tune a small VLM (e.g. **[LLaVA](https://github.com/haotian-liu/LLaVA)**
  with a LoRA adapter, or a CLIP linear-probe classifier) on RSVQA-LR/HR.
  Swap this in behind `core/query.py`'s answer() as an additional "learned"
  tool registered in `registry.yaml` (`type: learned`).

### 5.4 Learned change-VQA
- **[CDVQA / ChangeQA](https://github.com/shizenglin/ChangeQA)** — an
  existing open-source change-VQA baseline architecture; even reusing its
  head design over our own `core/change.py` CVA features is faster than
  designing one from scratch.
- Pairing datasets: **LEVIR-CD**, **WHU-CD** for bi-temporal ground truth.

### 5.5 Real optical–SAR fusion
- Current rule-based `fusion.joint_mask` (§3.2) is the honest stopgap.
  Upgrade path: a small feature-concatenation or cross-attention fusion
  head trained on **BigEarthNet's own S1 (SAR) + S2 (optical) pairs** — this
  conveniently reuses the same dataset as §5.1, so the BigEarthNet adapter
  and the fusion head can share a training pipeline.

### 5.6 Real eval data
- Download real splits: **[VRSBench](https://github.com/lx-cpu/VRSBench)**,
  **RSVQA-LR/HR**, **CDVQA**. Because `eval/` already speaks their JSONL
  schema (§3.5), this is a drop-in swap, not new code.

### 5.7 Sensor band tables
- Fill in real Sentinel-1/2, Cartosat-2S, RISAT band definitions in
  `core/sensors.py` (wavelengths, resolution, band order) — mostly reference
  lookup work, no ML needed, cheap to knock out early.

---

## 6. How close are we to fully solving PS 26167?

Judged strictly against the PS's **mandatory functional scope** (§1 above):

| Mandatory item | Status |
|---|---|
| GUI/web app | ✅ Fully met |
| Single-image VQA baseline | ✅ Met (classical) — ⚠️ would score higher learned |
| Captioning / grounding | ✅ Met (classical) — ⚠️ grounding now per-instance, still not learned |
| Bi-temporal change description/VQA | ✅ Met (classical) |
| Cross-modal optical+SAR extraction | ⚠️ Scaffolded today, not complete |
| Agentic orchestration + audit trace | ✅ Met today (registry.yaml + trace) |
| **≥1 component adapted on BigEarthNet** | ❌ **Not met — the one hard blocker** |

**Honest estimate:** the platform is feature-complete on breadth — every
query type the PS lists gets a real, evidence-backed answer, the UI is
demo-ready, and the agent framework is now legitimate (not just a marketing
word for "we called a function"). What's missing is *depth in the ML sense*:
one adapted model. That single item (§5.1) is what stands between "very
strong classical EO tool with agent framing" and "passes the PS's mandatory
scope." It's also the fastest of the five remaining ❌/⚠️ items to close
using the TorchGeo shortcut, because it's a fine-tune, not a from-scratch
train. Realistic estimate: a focused day of work (data loading via
TorchGeo's `BigEarthNet` dataset class, freeze backbone, fine-tune one linear
head on the 19-class labels, export weights + card) closes the single
hardest gate. Items 5.2–5.4 then become "bolt a small head onto that same
backbone," not new research.

---

## 7. How it all works (plain language, then technical)

**Plain language:** You upload a picture (or two). The app first checks the
picture is readable and figures out basic facts about it (size, how many
color channels, whether it has real-world scale info). When you type a
question, a *planner* reads your sentence and decides what kind of question
it is — "how much," "what changed," "where is," etc. Based on that, it picks
one or more specialist *tools* from a fixed catalog (`registry.yaml`) — one
tool is good at finding water, another at counting ships, another at
comparing two dates. Those tools are classical image-processing algorithms
(math on pixels — clustering colors, comparing edges between two photos),
*not* a big AI model reading the image like a human. Every number the tools
produce gets logged as "evidence," so the final answer can always point to
proof — a highlighted region, a percentage, a confidence score — instead of
just asserting something. That's the whole trick: it's an *honest* system
that never says "the water body increased" without a highlighted picture and
a percentage behind it.

**Technical language:** The request flow is
`upload → core/ingestion.py (format/CRS/GSD/checksum) → core/validation.py
(magic-byte hardening) → core/query.py (lexicon/rule-based NLU: intent +
target + operation extraction) → core/registry.py (resolves intent to
one/more tool ids in registry.yaml, checks modality/precondition
compatibility) → tool execution (core/features.py spectral indices,
core/landcover.py k-means+physics-rule classification, core/objects.py
contour/Hough detection, core/change.py ORB/RANSAC/ECC registration +
CVA+Otsu, core/sar.py SAR preprocessing + rule fusion) → core/evidence.py
(assembles E# evidence graph: value + method + confidence + visualization
ref) → core/render.py (renders overlays as data-URI images, including
per-instance boxes via connectedComponentsWithStats) → core/ai_reasoning.py
(offline deterministic template composer turns evidence into prose; no LLM
call happens unless an Ollama provider is explicitly configured) → response
with {answer, evidence[], trace{task, models[], params}, overlays[]}`. All
current "models" are classical/deterministic (`type: classical` in
`registry.yaml`); there are no trained neural network weights in the repo
yet — that's precisely gap §5.1. The FastAPI app (`app.py`) exposes this as
versioned REST routes (v1 preserved for compatibility, v2 adds
`/api/evidence`, `/api/quality`, `/api/report`, `/api/eval`, etc. — full list
in `docs/API.md`), and `static/index.html` is a single-page vanilla-JS
workspace UI (map viewer, layer toggles, chat, evidence trail, before/after
slider, report/history panel) with no frontend framework/build step.

---

## 8. Quick start

```bash
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
python -m pytest tests/ -q          # full test suite
```

Open `http://localhost:8000` → **Explore Demo** (ground-truthed bi-temporal
pair) → ask *"What changed between the two dates?"* → **Evidence** →
**Generate Report** → **History → Reproduce**.

New this session:
```bash
python -m pytest tests/test_ingestion.py -q   # GeoTIFF GSD cascade
python -m pytest tests/test_eval.py -q        # eval harness smoke tests
curl http://localhost:8000/api/eval           # run all eval splits (stand-in data)
```

## 9. Codebase map

```
app.py                  FastAPI server, v1+v2 routes, sessions, pipeline orchestration
core/registry.py        NEW — resolves tool ids to registry.yaml metadata + compat gate
core/sar.py              NEW — SAR speckle/backscatter preprocessing + rule-based fusion mask
core/ingestion.py       IngestionReport + NEW GSD-from-metadata cascade (rasterio/tifffile/manual)
core/validation.py      Upload hardening, magic-byte check
core/sensors.py         BandMapping registry (generic_rgb/rgb_nir verified; rest planned)
core/features.py        Spectral indices (NDVI/NDWI/SAVI/EVI/GNDVI + RGB proxies)
core/landcover.py       k-means + physics rules, adaptive k, quality label, uncertainty
core/objects.py         Water/veg/built regions, vessels, bright targets, linear (Hough)
core/change.py          ORB/RANSAC + ECC registration, CVA+Otsu, transitions, severity, hotspots
core/query.py           Rule/lexicon NLU → intent/target/operation/tools → deterministic answer
core/evidence.py        EvidenceGraph (E# measurement/inference/limitation + method + viz)
core/ai_reasoning.py    OfflineTemplateProvider (deterministic); Ollama stub (unconfigured)
core/render.py          Overlays + NEW mask_to_boxes() per-instance grounding boxes
core/report.py          Self-contained HTML audit report
core/history.py         Persistent versioned records + reproduce
core/config.py          VERSION, PIPELINE_VERSION, ALGO_VERSIONS, limits, severity thresholds
registry.yaml            NEW — agent tool catalog (name/version/modality/params/preconditions)
eval/                    NEW — vrsbench.py / rsvqa.py / cdvqa.py runners + run_all.py + stand-in data
static/index.html       Landing + workspace UI (viewer, layers, chat, evidence, reports, history)
samples/                delta_t1/t2.png + truth pair, + NEW geotiff_sample.tif
tests/                  10 suites incl. NEW test_ingestion.py, test_eval.py
docs/                   ARCHITECTURE, API, ALGORITHMS, EVIDENCE, DEVELOPMENT, SECURITY, LIMITATIONS
resource.md             Living gap-tracker: PS scope, done/left table, datasets, build order
```

## 10. Docs

`docs/ARCHITECTURE.md` · `docs/API.md` · `docs/ALGORITHMS.md` ·
`docs/EVIDENCE.md` · `docs/DEVELOPMENT.md` · `docs/SECURITY.md` ·
`docs/LIMITATIONS.md` · [`resource.md`](resource.md) (PS scope + gap tracker
— update this whenever a ❌ flips to ✅, with the commit/PR link)

## 11. Known limits

RGB-only proxies where NIR/SWIR is absent, inferred (not survey-grade)
land-cover classes, classical (not learned) detectors, GSD-dependent area
estimates, image-space coordinates for non-georeferenced input, eval scores
currently computed against stand-in sample data rather than real benchmark
splits (§5.6). Full list: `docs/LIMITATIONS.md`.
