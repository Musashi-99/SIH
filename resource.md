# SatQuery AI — Resource File (PS-26167)

> Single source of truth: problem statement + codebase status + gaps + datasets + UI reference prompts + roadmap.

---

## 1. Problem Statement

- **ID:** 26167
- **Title:** SatQuery AI - An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries
- **Org:** ISRO / Department of Space | **Category:** Software | **Theme:** Space Technology

**Objective:** Agentic vision-language assistant for single + paired RS images via natural-language queries. Single-image = mandatory baseline; principal focus = joint reasoning over paired cross-modal and multitemporal imagery.

### Defined Input Scope

| Input | Description | Formats |
|---|---|---|
| Single image | 1x optical/multispectral OR SAR — captioning, VQA, grounding | GeoTIFF/TIFF; PNG/JPEG only for benchmark datasets |
| Cross-modal pair | Co-registered optical + SAR, same area — joint extraction | GeoTIFF/TIFF |
| Bi-temporal pair | 2x spatially corresponding images, different times — change detection/description/change-VQA | GeoTIFF/TIFF |

### Mandatory Functional Scope

1. **RS adaptation:** ≥1 visual or VL component fine-tuned/adapted on BigEarthNet (or any open-source training data).
2. **Single-image baseline:** VQA mandatory + (captioning/scene description OR text-guided grounding).
3. **Multi-image change:** Change description OR change-VQA from bi-temporal pair mandatory. Spatial change map optional (where masks available).
4. **Cross-modal analysis:** Extract complementary info from co-registered optical + SAR pair.
5. **Agentic orchestration:** Auto select/sequence/execute specialist models/tools per query + input config.

### Representative Queries

- `Describe the land-cover and major objects visible in this image.`
- `Highlight the water body referred to in the query.`
- `What changed between these two dates, and where did the change occur?`
- `Use the optical and SAR images together to identify built-up and water-covered regions.`
- `Has the built-up area increased, decreased, or remained unchanged?`

### Agentic Orchestration (expected by judges)

- interpret query → classify task
- check number, modality, format, metadata, compatibility of inputs
- select 1+ models/tools from predefined registry
- configure permitted params + execute workflow
- combine textual + spatial outputs, estimate confidence, return visual evidence
- auditable execution summary: selected task, model/tool names, key params

> Only observable execution trace is evaluated (task, models/tools, params, outputs). Internal reasoning text is NOT evaluated.

### Expected Solution

Interactive GUI/web app + agentic RS backend: upload + compat check, RS-adapted VL component, specialists (VQA, caption/ground, change, optical-SAR), agentic controller, visual evidence + confidence + execution summaries + downloadable reports.

### Evaluation

Prescribed public benchmark test subsets + ISRO/SAC eval set (pre-georeferenced co-registered Cartosat-2S optical + RISAT SAR pairs, with reference answers/labels/boxes/masks — not disclosed). Scores normalised before combining.

---

## 2. Codebase Map (current)

```
app.py                  FastAPI server, v1+v2 routes, sessions, pipeline orchestration
core/ingestion.py       IngestionReport: format/dims/bands/CRS-or-unavailable/checksum/sensor map
core/validation.py      Upload hardening, magic-byte check (.tif incl.)
core/sensors.py         BandMapping registry (generic_rgb/rgb_nir verified; rest planned)
core/features.py        Spectral indices (NDVI/NDWI/SAVI/EVI/GNDVI w/ NIR; VARI/ExG/water-proxy on RGB)
core/landcover.py       k-means + physics rules, adaptive k, quality label, uncertainty
core/objects.py         Water/veg/built regions, vessels, bright targets, linear (Hough)
core/change.py          ORB/RANSAC + ECC registration, CVA+Otsu, transitions, severity, hotspots
core/query.py           Rule/lexicon NLU → intent/target/operation/tools → deterministic answer
core/evidence.py        EvidenceGraph (E# measurement/inference/limitation + method + viz)
core/ai_reasoning.py    OfflineTemplateProvider (deterministic); Ollama stub (unconfigured)
core/render.py          Overlays: landcover/class/detections/change/uncertainty → data URIs
core/report.py          Self-contained HTML audit report
core/history.py         Persistent versioned records + reproduce
core/config.py          VERSION, PIPELINE_VERSION, ALGO_VERSIONS, limits, severity thresholds
static/index.html       Landing + workspace UI (viewer, layers, chat, evidence, reports, history)
samples/                delta_t1/t2.png + truth, harbour_port, river_wetland, urban_mixed (.png only)
tests/                  8 suites: api/change/features/landcover/objects/query/regression/validation
docs/                   ARCHITECTURE, API, ALGORITHMS, EVIDENCE, DEVELOPMENT, SECURITY, LIMITATIONS
```

**Key routes:** `/api/analyze`, `/api/query`, `/api/analyze_sample`, `/api/samples`, `/api/health` (v1) + `/api/version`, `/api/explain`, `/api/indices`, `/api/objects`, `/api/evidence`, `/api/quality`, `/api/inspect`, `/api/report`, `/api/history/*`, `/api/benchmark`, `/api/sensors`, `/api/providers` (v2).

---

## 3. Comparison: DONE vs LEFT

### 3A. DONE / Partial (strengths)

| Requirement | Status | File proof |
|---|---|---|
| GUI/web app | ✅ Done | `static/index.html` |
| Upload + compat check | ✅ Done | `core/validation.py`, `core/ingestion.py:131`, `app.py:342` |
| Single-image VQA (classical) | ✅ Partial | `core/query.py:190 answer()` — describe/area/count/locate/presence/spectral/spatial |
| Captioning / scene description | ✅ Partial | `core/query.py:576` scene type + composition + narrative |
| Grounding (classical highlight) | ✅ Partial | `core/query.py:387` + `core/render.py:1` class-mask/detection overlays |
| Bi-temporal change + map + change-QA | ✅ Partial | `core/change.py:82`, `app.py:281` change overlay, severity, hotspots |
| Evidence + confidence + reports | ✅ Done | `core/evidence.py`, `/api/evidence`, `/api/quality`, `/api/report`, `/api/inspect` |
| Basic agent trace | ✅ Partial | `core/query.py:731 _TOOLS_FOR`, `app.py:622 /api/explain` steps+tools |
| TIFF accept (basic) | ✅ Partial | `core/config.py:29`, `core/ingestion.py:72` Pillow + optional rasterio |
| History/reproduce/benchmark | ✅ Done (bonus) | `core/history.py`, `app.py:827 /api/benchmark` |

### 3B. LEFT / Missing (judging blockers)

| Requirement | Status | What to build |
|---|---|---|
| RS adaptation (BigEarthNet fine-tune) | ❌ Missing | Add ≥1 adapted component (e.g. CLIP/SegFormer/U-Net or BLIP/LLaVA-RS LoRA on BigEarthNet multilabel). Ship weights/card + train script + proof. No torch/transformers in `requirements.txt` today; README says "No pretrained weights". |
| Learned RS-VQA (RSVQA) + VRSBench caption/grounding | ❌ Missing | Add VQA + caption or GroundingDINO/SAM head + `eval/` harness on prescribed splits. Current VQA is lexicon rules only. |
| Learned change-VQA (CDVQA) | ❌ Missing | Add change-caption/change-VQA head over bi-temporal features. Current answers are templates from `deltas/transitions`. |
| Optical–SAR fusion (mandatory) | ❌ Missing — critical | `image2` is T2-only (`app.py:376`). Need: mode flag (temporal vs cross-modal), SAR calibrate/speckle branch, co-register check, fusion (feature concat/attention), built-up+water joint head. Add SAR to `core/sensors.py` (now only generic_rgb verified). |
| Formal agent registry + audit summary | ⚠️ Partial | Need: task classifier (vqa/caption/ground/change/fusion), `registry.yaml` (model/tool name+version+params), modality/compat gate, observable trace `{task, models[], params, outputs, confidence}` per query. Current tools are code functions, not named models. |
| Sensor-accurate ingestion (S1/S2, Cartosat-2S, RISAT) | ⚠️ Partial | Need real multi-band + SAR reader (rasterio/tifffile required), GSD-from-metadata, band tables verified. Samples are `.png` only. |
| Eval harness (public + ISRO) | ❌ Missing | Need `eval/vrsbench.py`, `eval/rsvqa.py`, `eval/cdvqa.py` + metrics + downloadable eval report. None exist. |

**One-line verdict:** Strong classical EO demo (single + bi-temporal + evidence UI). Fails PS on (a) BigEarthNet adaptation, (b) optical–SAR fusion, (c) formal agent registry + eval harness.

---

## 4. Datasets & Links

| Dataset | Use | Link |
|---|---|---|
| BigEarthNet (S1 SAR + S2 multispectral + labels) | Primary adaptation / fine-tune | https://git.tu-berlin.de/rsim/bigearthnet-models-tf — arXiv: https://arxiv.org/abs/1906.11794 (PS lists `2603.29630` — verify; use official BigEarthNet) |
| VRSBench | Single-image captioning + grounding + VQA eval | https://github.com/lx-cpu/VRSBench |
| RSVQA (FloodNet/RSVQA-LR/HR) | Single-image VQA eval | https://github.com/ SylvainChagué/rsvqa — HF mirror: search `RSVQA-HR` |
| CDVQA / Change detection sets | Bi-temporal change-VQA eval | CDVQA: https://github.com/shizenglin/ChangeQA — pairs: LEVIR-CD, WHU-CD |
| Cartosat-2S + RISAT (ISRO eval) | Hidden eval — matched optical+SAR | Not public; ensure SAR-ready pipeline + GeoTIFF CRS handling now |

> Supported formats reminder: GeoTIFF/TIFF for all real inputs; PNG/JPEG only for benchmark datasets.

---

## 5. UI Reference — Pinterest / Image Prompts

### Image 1 — CURRENT flow (best output you have today)

**Tagline to search:** `dark satellite analytics dashboard evidence map viewer UI`

**Generation prompt (copy-paste):**
```
dark mode earth observation analytics dashboard, central satellite map viewer with land-cover overlay and change detection heatmap, left sidebar upload T1 T2 panels, right chat panel with grounded Q&A answers and confidence scores, evidence trail cards with E# IDs, HTML report preview, sleek ISRO-style mission control UI, high fidelity web app screenshot
```

**Pinterest searches:**
- `satellite image analysis dashboard UI dark mode`
- `earth observation web app interface`
- `geospatial analytics dashboard design`

**Maps to:** `static/index.html` workspace — viewer + layers + chat + evidence + report.

### Image 2 — FINAL flow (what to add → target)

**Tagline to search:** `multimodal AI agentic remote sensing copilot optical SAR fusion UI`

**Generation prompt (copy-paste):**
```
agentic AI remote sensing copilot web app, three input modes single image / optical-SAR pair / bi-temporal pair selector, side-by-side optical and SAR viewer with fusion layer and built-up water mask, before-after time slider with red change bounding boxes, natural language query bar, agent execution trace timeline showing task classifier to BigEarthNet VQA model to grounding model to change model to fusion model with model names and confidence, final low-flow architecture diagram, futuristic ISRO mission UI
```

**Pinterest searches:**
- `AI agent workflow UI multimodal`
- `SAR optical fusion visualization dashboard`
- `agentic AI pipeline interface`
- `before after image comparison slider UI`

**Maps to missing work:** mode selector + SAR/fusion viewer + agent trace + grounding boxes + eval badge.

---

## 6. Final Target Low-Flow (build order)

```
Upload (1/2 imgs + mode: single | optical-SAR | bi-temporal)
  → Ingest + compat gate (format/modality/CRS/bands/size → accept or honest refuse)
  → Task classifier (vqa / caption / grounding / change-desc / change-vqa / fusion)
  → Registry select (model name + version + params, e.g. bigearthnet-clip-v1, grounding-dino-rs, cdvqa-head-v1, sar-fusion-v1)
  → Execute specialists (classical pipeline KEEPS running as measurement base)
  → Fuse + ground (masks/boxes/change-map overlays)
  → Confidence + limitations
  → Answer + visual evidence + auditable trace {task, models[], params}
  → Downloadable HTML/PDF report + eval log
```

**Priority queue (do in this order):**
1. `registry.yaml` + task classifier + audit trace UI (cheapest judging points).
2. Optical–SAR mode: separate from T2, SAR preprocess + fusion head + dual viewer.
3. BigEarthNet adapter: CLIP-linear-probe or SegFormer multilabel → wire its labels into `landcover`/`query` as adapted source.
4. Grounding head (GroundingDINO/SAM or bbox regressor) → replace class-mask-only highlight.
5. Change-VQA head (CDVQA-style) over `change.py` features.
6. `eval/` harness (VRSBench/RSVQA/CDVQA splits) + metrics page.
7. GeoTIFF hardening: require rasterio, real S2/SAR bands, GSD-from-metadata, GeoTIFF demo pair in `samples/`.

---

## 7. Quick Commands

```bash
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
python -m pytest tests/ -q
```

Demo: open `http://localhost:8000` → Explore Demo (`delta_t1/t2`) → ask `What changed between the two dates?` → Evidence → Generate Report → History → Reproduce.

---

*Generated: 2026-09-27. Update this file when any ❌ above flips to ✅ (link the PR/commit + weight/card).*
