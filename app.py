"""
NewFace — Cosmetic Surgery Preview Tool
https://newface.live

Pipeline: detect face (InsightFace) → swap face (inswapper) →
          parse face regions (BiSeNet) → mask + blend → enhance (optional)
"""

import sys
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

# Set execution providers before importing any DLC modules
import modules.globals
modules.globals.execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
modules.globals.headless = True

from flask import Flask, request, jsonify, send_from_directory, send_file, Response
from flask_cors import CORS
import cv2
import numpy as np
import threading
import uuid
import base64
import time
import onnxruntime

from modules.face_analyser import get_one_face
from modules.processors.frame.face_swapper import get_face_swapper

app = Flask(__name__, static_folder='static')
CORS(app)

# Directories — everything is relative to app root now
MODELS_DIR = os.path.join(APP_DIR, 'models')
UPLOAD_DIR = os.path.join(APP_DIR, 'uploads')
REFERENCE_DIR = os.path.join(APP_DIR, 'references')
RESULTS_DIR = os.path.join(APP_DIR, 'results')
for d in (UPLOAD_DIR, REFERENCE_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# ENGINE STATE
# ═══════════════════════════════════════════════════════════

ENGINE_STATE = {
    'ready': False,
    'loading': False,
    'error': None,
}


def init_engine():
    """Load face swap model + BiSeNet in background thread."""
    ENGINE_STATE['loading'] = True
    ENGINE_STATE['error'] = None
    try:
        swapper = get_face_swapper()
        if swapper is None:
            ENGINE_STATE['error'] = 'Failed to load face swapper model'
            return
        print("[NewFace] Face swapper loaded.")

        # Preload BiSeNet so first frame isn't slow
        session = _load_bisenet()
        if session:
            print("[NewFace] BiSeNet face parser loaded.")
        else:
            print("[NewFace] BiSeNet unavailable — regional masks will fall back to full face.")

        ENGINE_STATE['ready'] = True
        print("[NewFace] Engine ready.")
    except Exception as e:
        ENGINE_STATE['error'] = str(e)
        print(f"[NewFace] Engine error: {e}")
    finally:
        ENGINE_STATE['loading'] = False


# ═══════════════════════════════════════════════════════════
# BISENET FACE PARSING
#
# Pixel-level face segmentation (~13MB ONNX model).
# CelebAMask-HQ class labels:
#   0=background  1=skin      2=l_brow    3=r_brow
#   4=l_eye       5=r_eye     6=glasses   7=l_ear
#   8=r_ear       9=earring   10=nose     11=mouth
#   12=u_lip      13=l_lip    14=neck     15=necklace
#   16=cloth      17=hair     18=hat
# ═══════════════════════════════════════════════════════════

VALID_REGIONS = {'full', 'nose', 'chin', 'brow', 'eyes', 'lips'}

REGION_CLASSES = {
    'nose': [10],
    'lips': [11, 12, 13],
    'eyes': [4, 5],
    'brow': [2, 3],
    'chin': [1],  # skin — cropped to lower face only
}

_bisenet_session = None
_bisenet_lock = threading.Lock()
BISENET_SIZE = 512
BISENET_URL = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/bisenet_resnet_34.onnx"
BISENET_FILE = "bisenet_resnet_34.onnx"


def _load_bisenet():
    """Load or download BiSeNet face parsing model."""
    global _bisenet_session
    with _bisenet_lock:
        if _bisenet_session is not None:
            return _bisenet_session

        model_path = os.path.join(MODELS_DIR, BISENET_FILE)

        if not os.path.exists(model_path):
            print(f"[NewFace] Downloading {BISENET_FILE}...")
            try:
                import urllib.request
                urllib.request.urlretrieve(BISENET_URL, model_path)
                print(f"[NewFace] Downloaded to {model_path}")
            except Exception as e:
                print(f"[NewFace] Download failed: {e}")
                return None

        try:
            opts = onnxruntime.SessionOptions()
            opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            _bisenet_session = onnxruntime.InferenceSession(
                model_path, sess_options=opts,
                providers=modules.globals.execution_providers,
            )
            print(f"[NewFace] BiSeNet providers: {_bisenet_session.get_providers()}")
            return _bisenet_session
        except Exception as e:
            print(f"[NewFace] BiSeNet load failed: {e}")
            return None


def parse_face(frame, face):
    """
    Run BiSeNet on a padded face crop.
    Returns full-frame parsing map (H x W, uint8 class labels) or None.
    """
    session = _load_bisenet()
    if session is None:
        return None

    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    # Pad bbox 25% for forehead/chin context
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * 0.25), int(bh * 0.25)
    x1p, y1p = max(0, x1 - px), max(0, y1 - py)
    x2p, y2p = min(w, x2 + px), min(h, y2 + py)

    crop = frame[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        return None

    # Preprocess: resize, BGR→RGB, normalize to [-1,1], NCHW
    resized = cv2.resize(crop, (BISENET_SIZE, BISENET_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[np.newaxis, ...]

    # Inference
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor})[0]
    parsing = np.argmax(output[0], axis=0).astype(np.uint8)

    # Project back to full frame
    crop_h, crop_w = crop.shape[:2]
    parsing_crop = cv2.resize(parsing, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
    full = np.zeros((h, w), dtype=np.uint8)
    full[y1p:y2p, x1p:x2p] = parsing_crop
    return full


def create_region_mask(frame_shape, face, parsing, region_name):
    """
    Build a soft float32 mask from BiSeNet parsing map.
    1.0 = show swapped, 0.0 = show original.
    """
    if region_name == 'full':
        return np.ones(frame_shape[:2], dtype=np.float32)

    if region_name not in VALID_REGIONS or parsing is None:
        return np.ones(frame_shape[:2], dtype=np.float32)

    classes = REGION_CLASSES.get(region_name, [])
    if not classes:
        return np.ones(frame_shape[:2], dtype=np.float32)

    # Binary mask from class labels
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    for cls in classes:
        mask[parsing == cls] = 255

    # Chin special case: skin below mouth line only
    if region_name == 'chin':
        kps = getattr(face, 'kps', None)
        if kps is not None and kps.shape[0] >= 5:
            mouth_y = int((kps[3][1] + kps[4][1]) / 2.0)
            mask[:mouth_y, :] = 0

    # Feather edges — proportional to face size
    face_w = face.bbox[2] - face.bbox[0]
    feather = max(11, int(face_w * 0.07))
    ksize = feather * 2 + 1
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), 0)
    mx = blurred.max()
    return blurred / mx if mx > 0 else blurred


# ═══════════════════════════════════════════════════════════
# FACE SWAP + BLEND
# ═══════════════════════════════════════════════════════════

def regional_face_swap(frame, ref_face, target_face, swapper, region='full', opacity=0.7):
    """
    1. Full face swap via inswapper
    2. BiSeNet parsing for pixel-level region mask
    3. Alpha blend only the target region
    """
    swapped = swapper.get(frame.copy(), target_face, ref_face, paste_back=True)
    if swapped is None:
        return frame
    swapped = np.clip(swapped, 0, 255).astype(np.uint8)

    if region == 'full':
        # Skip parsing — just blend everything
        if opacity >= 1.0:
            return swapped
        mask_3c = np.full(frame.shape, opacity, dtype=np.float32)
    else:
        # Run BiSeNet face parsing
        parsing = parse_face(frame, target_face)
        mask = create_region_mask(frame.shape, target_face, parsing, region)
        mask_3c = np.stack([mask * opacity] * 3, axis=-1)

    result = frame.astype(np.float32) * (1.0 - mask_3c) + swapped.astype(np.float32) * mask_3c
    return np.clip(result, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════
# QUALITY PIPELINE (enhancers + sharpness)
# Applied on snapshot/export only — too slow for live stream.
# ═══════════════════════════════════════════════════════════

_enhancer_cache = {}


def _get_enhancer(name):
    """Lazy-load an enhancer. Returns (signature, function) or None."""
    if name in _enhancer_cache:
        return _enhancer_cache[name]
    try:
        if name == 'gfpgan':
            from modules.processors.frame.face_enhancer import get_face_enhancer, enhance_face
            get_face_enhancer()
            _enhancer_cache[name] = ('no_face', enhance_face)
        elif name == 'gpen512':
            from modules.processors.frame.face_enhancer_gpen512 import get_enhancer, enhance_face
            get_enhancer()
            _enhancer_cache[name] = ('with_face', enhance_face)
        elif name == 'gpen256':
            from modules.processors.frame.face_enhancer_gpen256 import get_enhancer, enhance_face
            get_enhancer()
            _enhancer_cache[name] = ('with_face', enhance_face)
        return _enhancer_cache.get(name)
    except Exception as e:
        print(f"[NewFace] Enhancer '{name}' failed: {e}")
        _enhancer_cache[name] = None
        return None


def apply_quality_pipeline(frame, settings):
    """Apply one enhancer + sharpness to a frame."""
    result = frame

    # Pick first enabled enhancer (they're mutually exclusive)
    for name in ('gfpgan', 'gpen512', 'gpen256'):
        if not settings.get(name):
            continue
        entry = _get_enhancer(name)
        if not entry:
            break
        sig, fn = entry
        try:
            if sig == 'with_face':
                face = get_one_face(result)
                if face is not None:
                    result = fn(result, face)
            else:
                result = fn(result)
        except Exception as e:
            print(f"[NewFace] {name} error: {e}")
        break  # only run one

    # Unsharp mask sharpening
    sharpness = settings.get('sharpness', 0)
    if sharpness > 0:
        amount = sharpness / 50.0  # 0-100 → 0-2x
        blurred = cv2.GaussianBlur(result, (0, 0), 3)
        result = cv2.addWeighted(result, 1.0 + amount, blurred, -amount, 0)
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result


# ═══════════════════════════════════════════════════════════
# LIVE CAMERA
# ═══════════════════════════════════════════════════════════

CAMERA_STATE = {
    'active': False,
    'reference_face': None,
    'reference_filename': None,
    'opacity': 0.7,
    'region': 'nose',
    'mirror': True,
    'cap': None,
    'lock': threading.Lock(),
    'last_frame': None,
    'fps': 0,
    'gfpgan': False,
    'gpen512': False,
    'gpen256': False,
    'sharpness': 50,
}


def get_camera():
    with CAMERA_STATE['lock']:
        if CAMERA_STATE['cap'] is None or not CAMERA_STATE['cap'].isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            cap.set(cv2.CAP_PROP_FPS, 30)
            CAMERA_STATE['cap'] = cap
        return CAMERA_STATE['cap']


def release_camera():
    with CAMERA_STATE['lock']:
        if CAMERA_STATE['cap'] is not None:
            CAMERA_STATE['cap'].release()
            CAMERA_STATE['cap'] = None
        CAMERA_STATE['active'] = False


def generate_frames():
    """MJPEG generator with real-time regional face swap."""
    swapper = get_face_swapper()
    cap = get_camera()
    CAMERA_STATE['active'] = True

    frame_count = 0
    fps_start = time.time()

    while CAMERA_STATE['active']:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        if CAMERA_STATE['mirror']:
            frame = cv2.flip(frame, 1)

        CAMERA_STATE['last_frame'] = frame.copy()

        ref_face = CAMERA_STATE['reference_face']
        opacity = CAMERA_STATE['opacity']
        region = CAMERA_STATE['region']

        if ref_face is not None and swapper is not None and opacity > 0:
            try:
                target_face = get_one_face(frame)
                if target_face is not None:
                    frame = regional_face_swap(
                        frame, ref_face, target_face, swapper,
                        region=region, opacity=opacity,
                    )
            except Exception as e:
                print(f"[NewFace] Live swap error: {e}")

        # FPS counter
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            CAMERA_STATE['fps'] = round(frame_count / elapsed, 1)
            frame_count = 0
            fps_start = time.time()

        # Live sharpness (unsharp mask is fast enough for real-time)
        sharpness = CAMERA_STATE['sharpness']
        if sharpness > 0:
            amount = sharpness / 50.0
            blurred = cv2.GaussianBlur(frame, (0, 0), 3)
            frame = cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'

    release_camera()


# ═══════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# ═══════════════════════════════════════════════════════════
# HEARTBEAT WATCHDOG
# Browser polls /api/status every 2s. If no ping for 20s,
# the user closed the tab — auto-shutdown the server.
# ═══════════════════════════════════════════════════════════

_last_heartbeat = time.time()
HEARTBEAT_TIMEOUT = 20  # seconds


def _watchdog():
    """Kill server if browser stops polling."""
    while True:
        time.sleep(5)
        if time.time() - _last_heartbeat > HEARTBEAT_TIMEOUT:
            print("[NewFace] No heartbeat for 20s — browser closed. Shutting down.")
            release_camera()
            os._exit(0)


@app.route('/api/status')
def status():
    global _last_heartbeat
    _last_heartbeat = time.time()
    return jsonify({
        **ENGINE_STATE,
        'camera_active': CAMERA_STATE['active'],
        'camera_fps': CAMERA_STATE['fps'],
    })


# ─── References ──────────────────────────────────────────

@app.route('/api/references')
def list_references():
    refs = []
    for fname in sorted(os.listdir(REFERENCE_DIR)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            refs.append({
                'filename': fname,
                'name': os.path.splitext(fname)[0].replace('_', ' ').replace('-', ' ').title(),
                'url': f'/api/reference-image/{fname}',
            })
    return jsonify(refs)


@app.route('/api/reference-image/<filename>')
def get_reference_image(filename):
    return send_from_directory(REFERENCE_DIR, filename)


@app.route('/api/upload-reference', methods=['POST'])
def upload_reference():
    if 'photo' not in request.files:
        return jsonify({'error': 'No photo provided'}), 400

    file = request.files['photo']
    name = request.form.get('name', '').strip()
    category = request.form.get('category', 'general').strip()

    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Could not read image'}), 400

    face = get_one_face(img)
    if face is None:
        return jsonify({'error': 'No face detected in reference image'}), 400

    safe_name = name.lower().replace(' ', '_') if name else uuid.uuid4().hex[:8]
    ext = os.path.splitext(file.filename)[1] or '.jpg'
    filename = f"{category}_{safe_name}{ext}"
    filepath = os.path.join(REFERENCE_DIR, filename)
    with open(filepath, 'wb') as f:
        f.write(file_bytes)

    return jsonify({'filename': filename, 'name': name or safe_name,
                    'url': f'/api/reference-image/{filename}'})


# ─── Static Photo Upload + Preview ──────────────────────

@app.route('/api/upload', methods=['POST'])
def upload_photo():
    if 'photo' not in request.files:
        return jsonify({'error': 'No photo provided'}), 400

    file = request.files['photo']
    patient_id = uuid.uuid4().hex[:8]
    ext = os.path.splitext(file.filename)[1] or '.jpg'
    filename = f"{patient_id}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    if img is None:
        return jsonify({'error': 'Could not read image'}), 400

    face = get_one_face(img)
    if face is None:
        return jsonify({'error': 'No face detected in photo'}), 400

    h, w = img.shape[:2]
    return jsonify({
        'patient_id': patient_id, 'filename': filename,
        'width': w, 'height': h,
        'face_bbox': face.bbox.astype(int).tolist(),
        'message': 'Face detected successfully',
    })


@app.route('/api/upload-image/<filename>')
def get_upload_image(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route('/api/result-image/<filename>')
def get_result_image(filename):
    return send_from_directory(RESULTS_DIR, filename)


@app.route('/api/preview', methods=['POST'])
def preview():
    """Static image preview with regional swap + quality pipeline."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    patient_filename = data.get('patient_filename')
    reference_filename = data.get('reference_filename')
    opacity = max(0.0, min(1.0, float(data.get('opacity', 0.7))))
    region = data.get('region', 'full')

    if not patient_filename or not reference_filename:
        return jsonify({'error': 'Missing filenames'}), 400

    patient_path = os.path.join(UPLOAD_DIR, patient_filename)
    reference_path = os.path.join(REFERENCE_DIR, reference_filename)
    if not os.path.exists(patient_path) or not os.path.exists(reference_path):
        return jsonify({'error': 'Images not found'}), 404

    patient_img = cv2.imread(patient_path)
    reference_img = cv2.imread(reference_path)
    if patient_img is None or reference_img is None:
        return jsonify({'error': 'Could not read images'}), 400

    patient_face = get_one_face(patient_img)
    reference_face = get_one_face(reference_img)
    if patient_face is None or reference_face is None:
        return jsonify({'error': 'No face detected'}), 400

    swapper = get_face_swapper()
    if swapper is None:
        return jsonify({'error': 'Engine not ready'}), 503

    result = regional_face_swap(
        patient_img, reference_face, patient_face, swapper,
        region=region, opacity=opacity,
    )

    # Quality pipeline
    qs = {k: data.get(k, CAMERA_STATE.get(k, False)) for k in ('gfpgan', 'gpen512', 'gpen256')}
    qs['sharpness'] = data.get('sharpness', CAMERA_STATE.get('sharpness', 0))
    if any(qs.values()):
        result = apply_quality_pipeline(result, qs)

    result_filename = f"preview_{data.get('patient_id', 'x')}_{region}_{int(opacity*100)}.jpg"
    result_path = os.path.join(RESULTS_DIR, result_filename)
    cv2.imwrite(result_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])

    _, buf = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 92])
    b64 = base64.b64encode(buf).decode('utf-8')
    return jsonify({'image_b64': b64, 'result_filename': result_filename, 'opacity': opacity})


# ─── Live Camera ─────────────────────────────────────────

@app.route('/api/camera/stream')
def camera_stream():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/camera/start', methods=['POST'])
def camera_start():
    if CAMERA_STATE['active']:
        return jsonify({'status': 'already_running'})
    return jsonify({'status': 'ready', 'stream_url': '/api/camera/stream'})


@app.route('/api/camera/stop', methods=['POST'])
def camera_stop():
    CAMERA_STATE['active'] = False
    release_camera()
    return jsonify({'status': 'stopped'})


@app.route('/api/camera/set-reference', methods=['POST'])
def camera_set_reference():
    data = request.get_json()
    filename = data.get('reference_filename')

    if not filename:
        CAMERA_STATE['reference_face'] = None
        CAMERA_STATE['reference_filename'] = None
        return jsonify({'status': 'cleared'})

    ref_path = os.path.join(REFERENCE_DIR, filename)
    if not os.path.exists(ref_path):
        return jsonify({'error': 'Reference not found'}), 404

    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        return jsonify({'error': 'Could not read image'}), 400

    face = get_one_face(ref_img)
    if face is None:
        return jsonify({'error': 'No face detected'}), 400

    CAMERA_STATE['reference_face'] = face
    CAMERA_STATE['reference_filename'] = filename
    return jsonify({'status': 'set', 'filename': filename})


@app.route('/api/camera/settings', methods=['POST'])
def camera_settings():
    """Update live camera settings."""
    data = request.get_json()
    if 'opacity' in data:
        CAMERA_STATE['opacity'] = max(0.0, min(1.0, float(data['opacity'])))
    if 'region' in data and data['region'] in VALID_REGIONS:
        CAMERA_STATE['region'] = data['region']
    if 'mirror' in data:
        CAMERA_STATE['mirror'] = bool(data['mirror'])
    for key in ('gfpgan', 'gpen512', 'gpen256'):
        if key in data:
            CAMERA_STATE[key] = bool(data[key])
    if 'sharpness' in data:
        CAMERA_STATE['sharpness'] = max(0, min(100, int(data['sharpness'])))

    return jsonify({k: CAMERA_STATE[k] for k in
                    ('opacity', 'region', 'mirror', 'gfpgan', 'gpen512', 'gpen256', 'sharpness')})


@app.route('/api/camera/snapshot', methods=['POST'])
def camera_snapshot():
    """Capture current frame with swap + quality pipeline applied."""
    if CAMERA_STATE['last_frame'] is None:
        return jsonify({'error': 'No frame available'}), 400

    frame = CAMERA_STATE['last_frame'].copy()
    ref_face = CAMERA_STATE['reference_face']
    opacity = CAMERA_STATE['opacity']
    region = CAMERA_STATE['region']
    swapper = get_face_swapper()

    if ref_face is not None and swapper is not None and opacity > 0:
        target_face = get_one_face(frame)
        if target_face is not None:
            frame = regional_face_swap(
                frame, ref_face, target_face, swapper,
                region=region, opacity=opacity,
            )

    # Quality pipeline on snapshots
    qs = {k: CAMERA_STATE[k] for k in ('gfpgan', 'gpen512', 'gpen256', 'sharpness')}
    if any(qs.values()):
        frame = apply_quality_pipeline(frame, qs)

    snap_id = uuid.uuid4().hex[:8]
    snap_filename = f"snapshot_{snap_id}.jpg"
    cv2.imwrite(os.path.join(RESULTS_DIR, snap_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    raw_filename = f"snapshot_raw_{snap_id}.jpg"
    cv2.imwrite(os.path.join(RESULTS_DIR, raw_filename), CAMERA_STATE['last_frame'], [cv2.IMWRITE_JPEG_QUALITY, 95])

    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return jsonify({
        'image_b64': base64.b64encode(buf).decode('utf-8'),
        'result_filename': snap_filename,
        'raw_filename': raw_filename,
    })


@app.route('/api/export', methods=['POST'])
def export_comparison():
    data = request.get_json()
    before_filename = data.get('before_filename') or data.get('patient_filename')
    after_filename = data.get('after_filename') or data.get('result_filename')

    if not before_filename or not after_filename:
        return jsonify({'error': 'Missing filenames'}), 400

    before_path = os.path.join(RESULTS_DIR, before_filename)
    if not os.path.exists(before_path):
        before_path = os.path.join(UPLOAD_DIR, before_filename)
    after_path = os.path.join(RESULTS_DIR, after_filename)

    if not os.path.exists(before_path) or not os.path.exists(after_path):
        return jsonify({'error': 'Images not found'}), 404

    original = cv2.imread(before_path)
    result = cv2.imread(after_path)
    if original is None or result is None:
        return jsonify({'error': 'Could not read images'}), 400

    h, w = original.shape[:2]
    result = cv2.resize(result, (w, h))

    gap = 4
    divider = np.ones((h, gap, 3), dtype=np.uint8) * 40
    comparison = np.hstack([original, divider, result])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Before', (20, 40), font, 1.0, (255, 255, 255), 2)
    cv2.putText(comparison, 'After', (w + gap + 20, 40), font, 1.0, (255, 255, 255), 2)
    disclaimer = "SIMULATION ONLY - Not a guarantee of surgical results"
    ts = cv2.getTextSize(disclaimer, font, 0.5, 1)[0]
    cv2.putText(comparison, disclaimer, ((comparison.shape[1] - ts[0]) // 2, h - 20), font, 0.5, (180, 180, 180), 1)

    export_filename = f"comparison_{uuid.uuid4().hex[:8]}.jpg"
    export_path = os.path.join(RESULTS_DIR, export_filename)
    cv2.imwrite(export_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return send_file(export_path, mimetype='image/jpeg', as_attachment=True, download_name=export_filename)


# ─── Main ────────────────────────────────────────────────

if __name__ == '__main__':
    print("[NewFace] Starting engine initialization...")
    threading.Thread(target=init_engine, daemon=True).start()
    threading.Thread(target=_watchdog, daemon=True).start()

    # Auto-open browser after short delay
    import webbrowser
    threading.Timer(1.5, lambda: webbrowser.open('http://localhost:5959')).start()

    print("[NewFace] Server starting on http://localhost:5959")
    app.run(host='0.0.0.0', port=5959, debug=False, threaded=True)
