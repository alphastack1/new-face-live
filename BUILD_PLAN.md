# NewFace — Build Guide

> **How to use this file:** Drop it into an empty folder, open it in
> Claude Code, and say *"Read this file and build everything."*
> Requires: Windows 10/11, Python 3.10+, NVIDIA GPU with CUDA, ~2 GB disk for models.
> Double-click `start.bat`. Done.

---

# Part 1: The Big Picture

## What This App Does

A real-time cosmetic surgery preview tool. Point your webcam at your face,
select a reference face with the features you want (a particular nose shape,
lip fullness, jaw structure), pick a facial region — and see the result
blended onto your live video feed instantly. Think "Instagram filter" but
for plastic surgery consultations.

```
 USER EXPERIENCE
 ═══════════════════════════════════════════════════════════════

 First Launch (models auto-download on first use):
 ┌──────────────────────────────────────────────────────────┐
 │  ● Loading...                                            │
 │                                                          │
 │  Downloading face swap model...          265 MB          │
 │  Downloading face parser model...         89 MB          │
 │  Loading InsightFace detection...        330 MB          │
 │                                                          │
 │  One-time setup. ~2 GB total.                            │
 └──────────────────────────────────────────────────────────┘

 After models cached (instant on every future launch):
 ┌──────────────────────────────────────────────────────────┐
 │  NewFace                                    ● Ready      │
 │  ┌─────────────────────────────────────────────────────┐ │
 │  │                                                     │ │
 │  │              LIVE CAMERA FEED                       │ │
 │  │              (zoom: scroll / pinch)                 │ │
 │  │              12-18 FPS with face swap               │ │
 │  │                                                     │ │
 │  ├─────────────────────────────────────────────────────┤ │
 │  │  [▶ Start]              [◉ Snap]        [Export]    │ │
 │  ├─────────────────────────────────────────────────────┤ │
 │  │  Region: [Full] [●Nose] [Chin] [Brow] [Eyes] [Lips]│ │
 │  │  Blend:  ═══════●═══════════════════════════  70%   │ │
 │  ├─────────────────────────────────────────────────────┤ │
 │  │  Ref:  Nose│Lips│Eyes│Brow│Chin                     │ │
 │  │  [img][img][img][img][img][img] →  (scroll)         │ │
 │  └─────────────────────────────────────────────────────┘ │
 │  Live │ Photo │ Settings                                 │
 └──────────────────────────────────────────────────────────┘
```

## Core Pipeline

```
LIVE FRAME PROCESSING (~60-80ms per frame)
═══════════════════════════════════════════════════════════

  Camera frame (960×540)
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │  1. INSIGHTFACE DETECTION                    │
  │     buffalo_l model (det_10g + w600k_r50)    │
  │     → face bounding box + 5-point landmarks  │
  │     → 512-dim ArcFace embedding              │
  │     ~15ms                                    │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │  2. INSWAPPER_128 FACE SWAP                  │
  │     Takes: source embedding + target face    │
  │     Produces: full face swap (paste_back)    │
  │     ~30ms on GPU                             │
  │                                              │
  │     NOTE: inswapper uses the ArcFace         │
  │     embedding (pose-invariant), NOT the      │
  │     reference image pixels. Multi-angle      │
  │     refs don't help — one good front-facing  │
  │     ref is enough per look.                  │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │  3. BISENET FACE PARSING                     │
  │     resnet-34 variant, 512×512 input         │
  │     19 CelebAMask-HQ classes:                │
  │                                              │
  │     0=background  1=skin      2=l_brow       │
  │     3=r_brow      4=l_eye     5=r_eye        │
  │     6=glasses     7=l_ear     8=r_ear        │
  │     9=earring    10=nose     11=mouth         │
  │    12=u_lip      13=l_lip    14=neck          │
  │    15=necklace   16=cloth    17=hair          │
  │    18=hat                                     │
  │                                              │
  │     Produces: pixel-level region mask         │
  │     ~10ms on GPU                             │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │  4. ALPHA BLEND                              │
  │     mask × opacity × swapped_face            │
  │     + (1 - mask × opacity) × original_face   │
  │                                              │
  │     Feathered edges: GaussianBlur on mask    │
  │     proportional to face width (7% feather)  │
  │     ~2ms                                     │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │  5. SHARPNESS (live, every frame)            │
  │     Unsharp mask: addWeighted(orig, blur)    │
  │     Amount = slider / 50 (0-2× range)        │
  │     ~3ms                                     │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  MJPEG frame → browser (JPEG quality 80)
```

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    BROWSER (localhost:5959)                    │
│                                                               │
│   static/index.html — single file, no build step              │
│   ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
│   │ Live View  │  │ Photo View │  │ Settings View      │    │
│   │            │  │            │  │                     │    │
│   │ MJPEG feed │  │ Upload     │  │ GFPGAN toggle      │    │
│   │ Region     │  │ Before/    │  │ GPEN toggles       │    │
│   │ pills      │  │ After      │  │ Mirror toggle      │    │
│   │ Ref picker │  │ Export     │  │ Sharpness slider   │    │
│   │ Zoom/pan   │  │            │  │                     │    │
│   └─────┬──────┘  └─────┬──────┘  └──────────┬─────────┘    │
│         │               │                     │               │
└─────────┼───────────────┼─────────────────────┼───────────────┘
          │               │                     │
          ▼               ▼                     ▼
┌──────────────────────────────────────────────────────────────┐
│                    FLASK SERVER (app.py)                       │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ MJPEG        │  │ /api/        │  │ /api/camera/     │   │
│  │ Generator    │  │ preview      │  │ settings         │   │
│  │              │  │              │  │                   │   │
│  │ Camera loop  │  │ Upload img   │  │ Region, opacity  │   │
│  │ → detect     │  │ → detect     │  │ Mirror, quality  │   │
│  │ → swap       │  │ → swap       │  │ Sharpness        │   │
│  │ → parse      │  │ → parse      │  │                   │   │
│  │ → blend      │  │ → blend      │  │ Sent with 200ms  │   │
│  │ → sharpen    │  │ → enhance    │  │ debounce from UI │   │
│  │ → encode     │  │ → export     │  │                   │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘   │
│         │                 │                    │              │
└─────────┼─────────────────┼────────────────────┼──────────────┘
          │                 │                    │
          ▼                 ▼                    ▼
   ┌────────────┐    ┌────────────┐     ┌────────────┐
   │ InsightFace│    │ inswapper  │     │ BiSeNet    │
   │ buffalo_l  │    │ _128_fp16  │     │ resnet-34  │
   │            │    │            │     │            │
   │ Detection  │    │ Face swap  │     │ Face       │
   │ + ArcFace  │    │ ONNX       │     │ parsing    │
   │ ~330 MB    │    │ 265 MB     │     │ 89 MB      │
   └────────────┘    └────────────┘     └────────────┘
```

## What It Costs

| Item | Cost | Notes |
|------|------|-------|
| Everything | **$0** | All models are free and open source |
| Disk space | ~2 GB | One-time model downloads |
| GPU | NVIDIA with CUDA | Required for real-time (12-18 FPS) |

## What You Need

| Requirement | Notes |
|-------------|-------|
| Windows 10/11 | Linux/Mac work too with minor tweaks |
| Python 3.10+ | From [python.org](https://python.org) — must be in PATH |
| NVIDIA GPU | CUDA-capable. GTX 1060+ or RTX series |
| CUDA toolkit | Matching your onnxruntime-gpu version |
| ~2 GB free disk | For ML models. App itself is tiny |
| A browser | Chrome recommended for camera access |

---

# Part 2: Project Structure

```
new-face-live/
├── start.bat               ← Double-click to launch. Handles everything.
├── app.py                  ← Flask backend. ALL server logic in one file.
├── requirements.txt        ← Python deps
├── .gitignore
├── BUILD_PLAN.md           ← This file.
├── static/
│   └── index.html          ← ENTIRE frontend in one file. HTML + CSS + JS.
│                              No React, no build, no node_modules.
│                              Mobile-first. Dark gold theme.
├── modules/                ← Deep-Live-Cam engine (face detection, swap, enhance)
│   ├── globals.py
│   ├── face_analyser.py
│   └── processors/frame/
│       ├── face_swapper.py
│       ├── face_enhancer.py         (GFPGAN)
│       ├── face_enhancer_gpen256.py
│       └── face_enhancer_gpen512.py
└── references/             ← Pre-loaded AI-generated reference faces
    ├── nose_ref_1.jpg ... nose_ref_6.jpg
    ├── lips_ref_1.jpg ... lips_ref_4.jpg
    ├── chin_ref_1.jpg ... chin_ref_3.jpg
    ├── eyes_ref_1.jpg ... eyes_ref_3.jpg
    └── brow_ref_1.jpg ... brow_ref_3.jpg

Auto-created at runtime (gitignored):
├── venv/                   ← Python virtual environment
├── models/
│   ├── inswapper_128_fp16.onnx     ← 265 MB face swap model
│   └── bisenet_resnet_34.onnx      ← 89 MB face parser
├── uploads/                ← Patient photo uploads
└── results/                ← Generated previews and exports
```

---

# Part 3: The Backend (app.py)

One Python file. ~740 lines. No blueprints, no ORM, no complexity.

## Startup Flow

```
app.py starts
═══════════════════════════════════════════════════════════

  ┌───────────────────────────────────────────────────┐
  │  1. SET EXECUTION PROVIDERS                        │
  │                                                    │
  │  modules.globals.execution_providers =             │
  │    ['CUDAExecutionProvider', 'CPUExecutionProvider']│
  │  modules.globals.headless = True                   │
  │                                                    │
  │  WHY: Deep-Live-Cam defaults to tkinter UI.        │
  │  headless=True prevents it. CUDA first, CPU        │
  │  fallback for BiSeNet if cuDNN version mismatches. │
  └───────────────────────┬───────────────────────────┘
                          │
  ┌───────────────────────▼───────────────────────────┐
  │  2. BACKGROUND ENGINE INIT                         │
  │                                                    │
  │  threading.Thread(target=init_engine, daemon=True) │
  │                                                    │
  │  init_engine() does:                               │
  │  ├── get_face_swapper()  → loads inswapper ONNX    │
  │  ├── _load_bisenet()     → loads/downloads BiSeNet │
  │  └── ENGINE_STATE['ready'] = True                  │
  │                                                    │
  │  Server starts immediately, engine loads in back.  │
  │  Frontend polls /api/status until ready=true.      │
  └───────────────────────┬───────────────────────────┘
                          │
  ┌───────────────────────▼───────────────────────────┐
  │  3. FLASK SERVER                                   │
  │                                                    │
  │  app.run(host='0.0.0.0', port=5959, threaded=True)│
  │                                                    │
  │  0.0.0.0 = accessible from mobile on same LAN     │
  │  threaded=True = multiple concurrent requests      │
  └───────────────────────────────────────────────────┘
```

## BiSeNet Face Parsing (the key differentiator)

```
parse_face(frame, face) → full-frame class map
═══════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────┐
  │  1. PAD BOUNDING BOX 25%                             │
  │     (needs forehead/chin context for good parsing)   │
  │                                                      │
  │     ┌──────────────────┐                             │
  │     │   padding 25%    │                             │
  │     │ ┌──────────────┐ │                             │
  │     │ │              │ │                             │
  │     │ │   face bbox  │ │                             │
  │     │ │              │ │                             │
  │     │ └──────────────┘ │                             │
  │     └──────────────────┘                             │
  └──────────────────┬──────────────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────────────┐
  │  2. PREPROCESS                                       │
  │     Resize crop → 512×512                            │
  │     BGR → RGB                                        │
  │     Normalize to [-1, 1]                             │
  │     NCHW tensor: [1, 3, 512, 512]                   │
  └──────────────────┬──────────────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────────────┐
  │  3. INFERENCE → argmax → class map                   │
  │     Output: [1, 19, 512, 512] → argmax → [512, 512] │
  │     Each pixel = class label (0-18)                  │
  └──────────────────┬──────────────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────────────┐
  │  4. PROJECT BACK TO FULL FRAME                       │
  │     Resize parsing to crop size (INTER_NEAREST)      │
  │     Place into full-frame zeros array                │
  │     → H×W uint8 class map                           │
  └─────────────────────────────────────────────────────┘


create_region_mask(frame_shape, face, parsing, region)
═══════════════════════════════════════════════════════════

  Region class mapping:
  ┌────────────┬─────────────────────────────────────┐
  │ Region     │ BiSeNet Classes                      │
  ├────────────┼─────────────────────────────────────┤
  │ nose       │ [10]                                 │
  │ lips       │ [11, 12, 13]  (mouth, u_lip, l_lip) │
  │ eyes       │ [4, 5]        (l_eye, r_eye)         │
  │ brow       │ [2, 3]        (l_brow, r_brow)       │
  │ chin       │ [1]           (skin, below mouth only)│
  │ full       │ entire face (no parsing needed)       │
  └────────────┴─────────────────────────────────────┘

  Special: chin crops skin mask to below mouth line only
  (uses landmark kps[3], kps[4] mouth corners)

  Feathering: GaussianBlur with kernel = 7% of face width
  Result: float32 mask, 0.0 = original, 1.0 = swapped
```

## Quality Pipeline (snapshot/export only)

```
apply_quality_pipeline(frame, settings)
═══════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────┐
  │  ENHANCERS (mutually exclusive, pick first)   │
  │                                               │
  │  ┌─────────┐  ┌──────────┐  ┌──────────┐    │
  │  │ GFPGAN  │  │ GPEN-512 │  │ GPEN-256 │    │
  │  │ 325 MB  │  │ 272 MB   │  │ 73 MB    │    │
  │  │ no_face │  │ with_face│  │ with_face│    │
  │  │ sig     │  │ sig      │  │ sig      │    │
  │  └─────────┘  └──────────┘  └──────────┘    │
  │                                               │
  │  "no_face" = enhance_face(frame)              │
  │  "with_face" = enhance_face(frame, face_obj)  │
  │                                               │
  │  Lazy-loaded on first use via _get_enhancer() │
  │  Too slow for live stream (~200-500ms)        │
  └───────────────────┬──────────────────────────┘
                      │
  ┌───────────────────▼──────────────────────────┐
  │  SHARPNESS (also applied live)                │
  │                                               │
  │  amount = slider_value / 50.0  (0-2× range)  │
  │  blurred = GaussianBlur(frame, sigma=3)       │
  │  result = addWeighted(frame, 1+amt, blur, -amt)│
  └──────────────────────────────────────────────┘
```

## API Routes

```
┌────────┬──────────────────────────┬────────────────────────────────────┐
│ Method │ Path                     │ What it does                       │
├────────┼──────────────────────────┼────────────────────────────────────┤
│ GET    │ /                        │ Serve static/index.html            │
│ GET    │ /api/status              │ Engine state + camera FPS          │
│ GET    │ /api/references          │ List reference face images         │
│ GET    │ /api/reference-image/:fn │ Serve a reference image            │
│ POST   │ /api/upload-reference    │ Upload custom reference face       │
│ POST   │ /api/upload              │ Upload patient photo               │
│ GET    │ /api/upload-image/:fn    │ Serve uploaded patient photo       │
│ POST   │ /api/preview             │ Static photo: swap + enhance       │
│ GET    │ /api/camera/stream       │ MJPEG live feed with real-time swap│
│ POST   │ /api/camera/start        │ Signal camera ready                │
│ POST   │ /api/camera/stop         │ Stop camera + release              │
│ POST   │ /api/camera/set-reference│ Set active reference face          │
│ POST   │ /api/camera/settings     │ Update region, opacity, toggles    │
│ POST   │ /api/camera/snapshot     │ Capture frame + quality pipeline   │
│ POST   │ /api/export              │ Before/after side-by-side JPEG     │
└────────┴──────────────────────────┴────────────────────────────────────┘
```

## MJPEG Generator (the live loop)

```
generate_frames() — runs in Flask response context
═══════════════════════════════════════════════════════════

  cap = cv2.VideoCapture(0)  ← 960×540 @ 30 FPS
  CAMERA_STATE['active'] = True

  while active:
  ┌──────────────────────────────────────────────┐
  │  1. Read frame from camera                    │
  │  2. Mirror if enabled (cv2.flip horizontal)   │
  │  3. Save raw frame (for snapshot comparison)  │
  │  4. If reference face is set:                 │
  │     ├── detect target face in frame           │
  │     ├── regional_face_swap()                  │
  │     └── (includes BiSeNet + alpha blend)      │
  │  5. Apply live sharpness (unsharp mask)       │
  │  6. Overlay FPS + region text                 │
  │  7. JPEG encode (quality 80)                  │
  │  8. Yield as MJPEG frame                      │
  │                                               │
  │  Frame budget: ~60-80ms = 12-16 FPS           │
  │  Bottleneck: face detection + swap (~45ms)    │
  └──────────────────────────────────────────────┘
```

---

# Part 4: The Frontend (static/index.html)

One HTML file. No framework, no npm, no build step. Everything inline.
Mobile-first responsive design.

## Design System

```
VISUAL LANGUAGE
═══════════════════════════════════════════════════════════

  Theme:      Dark (near-black background, gold accent)
  Fonts:      Inter (UI text) + JetBrains Mono (values/stats)
  Icons:      Inline SVGs
  Layout:     Mobile-first, single column
  Desktop:    768px+ breakpoint, side panel for controls

  Color tokens (CSS variables on :root):
  ┌──────────────────────────────────────────────────┐
  │  Backgrounds          Borders         Text       │
  │  --bg:    #08080c     --border:  gold/12%  --text:  #f0ece4│
  │  --bg2:   #0c0c12     --border2: gold/25%  --text2: #a8a4a0│
  │  --bg3:   #101018                          --muted: #6b6865│
  │  --surface: rgb(16,16,24)/.8                      │
  │                                                    │
  │  Accent               Status                      │
  │  --gold:   #e8af48    --green: #4ade80             │
  │  --gold-l: #feeaa3    --red:   #f87171             │
  │  --gold-d: #c49746                                 │
  │  --gold-dk:#533517                                 │
  │  --glow:   gold/15%                                │
  └──────────────────────────────────────────────────┘

  Key UI patterns:
  ├── Pills for region selection (rounded, gold glow when active)
  ├── Category tabs for reference filtering (horizontal scroll)
  ├── Horizontal scroll for reference thumbnails (60px squares)
  ├── Range sliders with gold thumb + glow
  ├── Toggle switches for enhancer on/off
  ├── Toast notifications (bottom, auto-dismiss 2.2s)
  ├── Camera flash animation on snapshot
  └── Zoom: scroll wheel (desktop) + pinch (mobile) + double-tap reset
```

## Mobile vs Desktop Layout

```
MOBILE (< 768px)                    DESKTOP (768px+)
═══════════════════                 ═══════════════════

┌──────────────────┐               ┌────────────────────────────┐
│ NewFace  ● Ready │ 36px          │ NewFace            ● Ready │ 44px
├──────────────────┤               ├────────────────────────────┤
│ Live│Photo│Settings│ 30px        │    Live │ Photo │ Settings │
├──────────────────┤               ├──────────────────┬─────────┤
│                  │               │                  │ Region  │
│   CAMERA FEED    │               │                  │ pills   │
│   (fills space)  │               │   CAMERA FEED    │         │
│                  │               │   (fills space)  │ Blend   │
├──────────────────┤               │                  │ slider  │
│ [Start] [Snap]   │               │                  │         │
├──────────────────┤               │  [Start] [Snap]  │ Ref     │
│ Region pills     │               │  (overlay bar)   │ tabs +  │
│ Blend slider     │               │                  │ thumbs  │
│ Ref tabs + scroll│               │                  │         │
└──────────────────┘               └──────────────────┴─────────┘

Video priority:                     Side panel: 320px fixed
fills ALL remaining                 Video: fills rest
vertical space
```

## Zoom Implementation

```
ZOOM & PAN
═══════════════════════════════════════════════════════════

  State: { scale: 1, tx: 0, ty: 0, min: 1, max: 5 }

  Desktop (scroll wheel):
  ┌──────────────────────────────────────────────┐
  │  wheel event on .cam-box                      │
  │  deltaY > 0 → zoom out (scale -= 0.15)       │
  │  deltaY < 0 → zoom in  (scale += 0.15)       │
  │  Clamp to [1, 5]                              │
  │  Reset translate when scale ≤ 1               │
  └──────────────────────────────────────────────┘

  Mobile (pinch):
  ┌──────────────────────────────────────────────┐
  │  2 touches → track initial distance           │
  │  touchmove → scale = initial × (dist / dist0) │
  │  1 touch + zoomed → pan (translate offset)    │
  │  Snap to 1.0 when scale < 1.05               │
  └──────────────────────────────────────────────┘

  Reset: double-tap (mobile) / double-click (desktop)

  CSS: img { transform-origin: center; touch-action: none }
  JS:  img.style.transform = `scale(S) translate(TX%, TY%)`
```

## Reference Face Picker

```
CATEGORY TABS + HORIZONTAL SCROLL
═══════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────┐
  │  [Nose 6] [Lips 4] [Eyes 3] [Brow 3] [Chin 3]│  ← tabs
  │                                               │
  │  [img] [img] [img] [img] [img] [img] →        │  ← scroll
  │                        ↑ gold border = selected│
  │                                               │
  │  [Clear]                            [+ Add]   │
  └──────────────────────────────────────────────┘

  Filename convention: {category}_ref_{n}.jpg
  Category extracted from filename prefix (split on '_')
  19 pre-loaded AI-generated faces selected by landmark analysis
  Custom upload: modal with dropzone + name + category select
```

---

# Part 5: The Launcher (start.bat)

```
start.bat
═══════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────┐
  │  1. cd to script directory                   │
  │                                              │
  │  2. Check for venv/                          │
  │     └── If exists → use venv\Scripts\python  │
  │     └── If not → use system python           │
  │                                              │
  │  3. Print banner:                            │
  │     ┌──────────────────────────────────┐     │
  │     │  Starting NewFace...             │     │
  │     └──────────────────────────────────┘     │
  │                                              │
  │  4. Run: python app.py                       │
  │                                              │
  │  5. On exit → pause (so user sees errors)    │
  └─────────────────────────────────────────────┘
```

---

# Part 6: Design Decisions & Gotchas

```
┌─────────────────────────────────────────────────────────────┐
│              WHY IT'S BUILT THIS WAY                         │
│                                                              │
│  Single HTML file (no build step)                            │
│  └── WHY: Zero tooling. Works on any machine. Just serve it.│
│                                                              │
│  BiSeNet for region masking (not landmarks)                  │
│  └── WHY: Hand-crafted ellipse masks from 5-point landmarks │
│      fail at head angles. BiSeNet gives pixel-level accuracy │
│      at any pose. FaceFusion uses the same approach.         │
│                                                              │
│  MJPEG streaming (not WebSocket/WebRTC)                      │
│  └── WHY: Simplest possible streaming. Works in any browser.│
│      No client-side JS needed for video display — just an    │
│      <img> tag with src="/api/camera/stream". Tradeoff:      │
│      higher bandwidth than H.264, but no codec complexity.   │
│                                                              │
│  Reference always sent to backend (even when cam is off)     │
│  └── WHY: If server restarts mid-session, the new server    │
│      has no reference in memory. By always sending on        │
│      selection, the backend state stays in sync.             │
│                                                              │
│  Sharpness applied live (enhancers only on snapshot)         │
│  └── WHY: Unsharp mask is ~3ms (fast enough for live).       │
│      GFPGAN/GPEN are 200-500ms (would drop FPS to 2-3).     │
│      Apply enhancers only when user takes a snapshot.        │
│                                                              │
│  Mobile-first layout                                         │
│  └── WHY: Surgeons show patients previews on iPad/phone.     │
│      Desktop is secondary. Video fills max vertical space.   │
│                                                              │
│  AI-generated reference faces (thispersondoesnotexist.com)   │
│  └── WHY: No consent/licensing issues. StyleGAN faces are    │
│      synthetic. Selected by InsightFace landmark analysis    │
│      for maximum feature distinctiveness per category.       │
└─────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────┐
│              GOTCHAS TO WATCH FOR                             │
│                                                              │
│  ✗ inswapper uses ArcFace EMBEDDING, not pixels              │
│    └── The 512-dim embedding is pose-invariant. Having       │
│        multiple angles of the same reference face does NOT   │
│        improve results. One good front-facing photo is       │
│        enough per "look".                                    │
│                                                              │
│  ✗ BiSeNet needs padded crop (25% bbox expansion)            │
│    └── Without padding, the model has no forehead/chin       │
│        context and parsing quality degrades significantly.   │
│                                                              │
│  ✗ Chin region = skin class BELOW mouth line                 │
│    └── BiSeNet class 1 (skin) covers the entire face skin.  │
│        For chin-only masking, we crop the skin mask to       │
│        below the mouth landmarks (kps[3] + kps[4] avg Y).   │
│                                                              │
│  ✗ Camera release on stop                                    │
│    └── Must call cap.release() or the camera stays locked.   │
│        generate_frames() calls release_camera() when the     │
│        active flag goes false.                               │
│                                                              │
│  ✗ CUDA fallback                                             │
│    └── BiSeNet may fall back to CPU if cuDNN version doesn't │
│        match onnxruntime-gpu. This is fine — 512×512 input   │
│        runs fast enough on CPU (~15ms). The face swapper     │
│        still uses CUDA.                                      │
│                                                              │
│  ✗ Debounce slider sends                                     │
│    └── Without debounce, dragging a slider sends 30+ API     │
│        requests/sec. Opacity debounced at 150ms, settings    │
│        at 200ms.                                             │
└─────────────────────────────────────────────────────────────┘
```

---

# Part 7: Deployment Roadmap

```
CURRENT: LOCAL MVP
═══════════════════════════════════════════════════════════

  User downloads repo → start.bat → localhost:5959
  Camera is local (OpenCV). GPU is local. Everything local.
  Works great for clinic use on a dedicated machine.

FUTURE: WEB APP (newface.live)
═══════════════════════════════════════════════════════════

  Can't deploy to Netlify/Vercel/GitHub Pages — those are
  static hosting only. This app needs Python + GPU.

  Architecture change required:
  ┌──────────────────────────────────────────────────┐
  │  CURRENT         │  WEB VERSION                   │
  ├──────────────────┼───────────────────────────────┤
  │  Camera: OpenCV  │  Camera: browser getUserMedia  │
  │  Stream: MJPEG   │  Stream: WebSocket frames      │
  │  GPU: local      │  GPU: cloud (Railway/AWS/GCP)  │
  │  Latency: ~5ms   │  Latency: ~100-200ms           │
  └──────────────────┴───────────────────────────────┘

  Hosting options for GPU backend:
  ┌──────────────┬─────┬───────────┬──────────────────┐
  │ Provider     │ GPU │ Cost      │ Notes            │
  ├──────────────┼─────┼───────────┼──────────────────┤
  │ Railway      │ T4  │ ~$0.50/hr │ Easy deploy      │
  │ AWS EC2 g4dn│ T4  │ ~$0.52/hr │ Full control     │
  │ GCP Compute  │ T4  │ ~$0.35/hr │ Cheapest         │
  │ Self-hosted  │ Any │ Hardware  │ Best for clinics  │
  └──────────────┴─────┴───────────┴──────────────────┘
```
