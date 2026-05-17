"""
ai_slot_gen.py — AI image-to-image generation + slot extraction.

Two providers are supported:
  • Nano Banana Pro  (primary)  — async task API, needs a public image URL
  • Gemini           (fallback) — sync SDK, accepts bytes directly

Flow:
  1. Send the best-confidence camera frame to the AI provider with a prompt
     that fills every empty parking space with a parked car.
  2. Run YOLO on the returned generated image.
  3. Convert each detected car bounding box into a slot quad.
"""

import logging
import time
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image

log = logging.getLogger(__name__)

# ── Nano Banana Pro ───────────────────────────────────────────────────────────
NB_BASE         = "https://api.nanobananaapi.ai"
NB_GENERATE_PRO = "/api/v1/nanobanana/generate-pro"
NB_TASK_DETAIL  = "/api/v1/nanobanana/record-info"    # GET ?taskId=...

GEMINI_MODEL = "gemini-2.5-flash-image-preview"

DEFAULT_PROMPT = (
    "You are editing a real CCTV overhead fisheye image of a parking lot at night. "
    "STRICT RULE: Only place a parked car inside a parking stall that is CLEARLY bounded "
    "by at least two painted white or yellow lines on its sides, OR backed against a yellow "
    "rubber wheel-stop bumper. "
    "A slot must have visible boundary markings — do NOT place cars in any open area that "
    "lacks these markings. "
    "The large open central area and all driving lanes, access roads, and circulation paths "
    "MUST remain completely empty — no cars there under any circumstances. "
    "Replace orange traffic cones or reserve stands only when they sit INSIDE a clearly "
    "marked stall with visible boundary lines. "
    "Cars must be aligned parallel to their stall's orientation (overhead view, sedan/SUV/hatchback). "
    "Do not move, alter, or remove any car that is already parked in the original image. "
    "Preserve all asphalt, road markings, lighting, and shadows exactly as they are. "
    "Output must be indistinguishable from the original surveillance footage."
)

TARGET_CLASSES = [0]   # class 0 = car (matches flask_api.py)


def _blank_cone_regions(frame_bgr: np.ndarray, cone_boxes: list) -> np.ndarray:
    """
    Paint over each (x1,y1,x2,y2) cone bounding box with the median color
    sampled from a 10-px border ring around it, making the slot look like
    bare pavement before the frame is sent to the AI.
    Returns a copy of the frame.
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in cone_boxes:
        pad = 10
        bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
        bx2, by2 = min(w, x2 + pad), min(h, y2 + pad)
        border_mask = np.zeros((h, w), dtype=np.uint8)
        border_mask[by1:by2, bx1:bx2] = 255
        border_mask[y1:y2, x1:x2] = 0
        border_pixels = out[border_mask == 255]
        if len(border_pixels):
            fill_color = np.median(border_pixels, axis=0).astype(np.uint8)
            out[y1:y2, x1:x2] = fill_color
    return out


def generate_filled_lot(
    frame_jpeg: bytes,
    api_key: str,
    prompt: str = None,
    cone_boxes: list | None = None,
) -> bytes:
    """
    Send a JPEG parking lot frame to Gemini and return the AI-generated image bytes
    where all empty spaces are filled with parked cars.

    Raises ValueError if no image part is found in the response.
    """
    if cone_boxes:
        raw = cv2.imdecode(np.frombuffer(frame_jpeg, np.uint8), cv2.IMREAD_COLOR)
        raw = _blank_cone_regions(raw, cone_boxes)
        _, buf = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_jpeg = buf.tobytes()

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    img = Image.open(BytesIO(frame_jpeg))
    used_prompt = prompt or DEFAULT_PROMPT
    log.info(f"[AI] Sending frame to Gemini ({GEMINI_MODEL}) — prompt: {repr(used_prompt[:80])}")

    response = model.generate_content([used_prompt, img])

    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                return part.inline_data.data   # raw image bytes (PNG/JPEG)

    raise ValueError("[AI] Gemini returned no image data in response.")


def generate_filled_lot_nb(
    frame_url: str,
    api_key: str,
    prompt: str = None,
    resolution: str = "2K",
    aspect_ratio: str = "16:9",
    poll_interval: float = 5.0,
    timeout: float = 300.0,
) -> bytes:
    """
    Submit a Nano Banana Pro image generation task and poll until it completes.

    Args:
        frame_url:     Publicly accessible URL of the parking lot frame to edit.
        api_key:       Nano Banana Pro API key.
        prompt:        Generation prompt (defaults to DEFAULT_PROMPT).
        resolution:    "1K" | "2K" | "4K"
        aspect_ratio:  One of the supported aspect ratios (default "16:9").
        poll_interval: Seconds between task status checks.
        timeout:       Total seconds before giving up.

    Returns:
        Raw image bytes of the generated (filled) parking lot.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "prompt":      prompt or DEFAULT_PROMPT,
        "imageUrls":   [frame_url],
        "resolution":  resolution,
        "aspectRatio": aspect_ratio,
    }

    log.info(f"[AI/NB] Submitting Pro task — resolution={resolution}, ratio={aspect_ratio}")
    r = requests.post(f"{NB_BASE}{NB_GENERATE_PRO}", json=payload, headers=headers, timeout=30)
    r.raise_for_status()

    resp_data = r.json()
    log.info(f"[AI/NB] Submission response: {resp_data}")
    data_obj  = resp_data.get("data") or {}

    # Some NB endpoints return the result synchronously in the submission response.
    immediate_url = _extract_image_url(data_obj)
    if immediate_url:
        log.info(f"[AI/NB] Result available immediately — downloading")
        img_resp = requests.get(immediate_url, timeout=60)
        img_resp.raise_for_status()
        return img_resp.content

    task_id = data_obj.get("taskId") or data_obj.get("task_id") or data_obj.get("id")
    if not task_id:
        raise ValueError(f"[AI/NB] No taskId in response: {resp_data}")
    log.info(f"[AI/NB] Task submitted: {task_id}")

    # Poll /record-info until successFlag != 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        sr = requests.get(
            f"{NB_BASE}{NB_TASK_DETAIL}",
            params={"taskId": task_id},
            headers=headers,
            timeout=15,
        )
        sr.raise_for_status()
        task_data   = sr.json().get("data", {})
        flag        = task_data.get("successFlag", 0)
        log.info(f"[AI/NB] Task {task_id} successFlag={flag}")

        if flag == 1:
            result_url = (task_data.get("response") or {}).get("resultImageUrl")
            if not result_url:
                raise ValueError(f"[AI/NB] Task complete but no resultImageUrl: {task_data}")
            log.info(f"[AI/NB] Task {task_id} complete — downloading result")
            img_resp = requests.get(result_url, timeout=60)
            img_resp.raise_for_status()
            return img_resp.content

        if flag in (2, 3):
            raise ValueError(f"[AI/NB] Task {task_id} failed (flag={flag}): {task_data.get('errorMessage')}")

    raise TimeoutError(f"[AI/NB] Task {task_id} did not complete within {timeout}s")


def _extract_image_url(data: dict) -> str | None:
    """Check if a submission response already contains a result image URL (synchronous case)."""
    response_obj = data.get("response") or {}
    candidates = [
        response_obj.get("resultImageUrl"),
        data.get("resultImageUrl"),
        data.get("imageUrl"),
        data.get("url"),
    ]
    return next((c for c in candidates if c and isinstance(c, str) and c.startswith("http")), None)


def extract_slots_from_ai_frame(
    ai_frame_bytes: bytes,
    yolo_model,
    conf: float,
    existing_slot_ids: set,
    cam_frame_bytes: bytes | None = None,
) -> dict:
    """
    Run YOLO on the AI-generated image and convert each detected car bbox to a slot quad.

    Returns a dict of proposed slots:
      { "S01": {"coords": [[x,y],[x,y],[x,y],[x,y]], "source": "ai_generated"}, ... }

    Slot IDs are assigned sequentially, skipping IDs already in existing_slot_ids.
    If cam_frame_bytes is provided the AI image is resized to match the camera resolution
    before YOLO runs, so all output coordinates are already in camera space.
    """
    img = cv2.imdecode(np.frombuffer(ai_frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        log.error("[AI] Failed to decode AI-generated image")
        return {}

    if cam_frame_bytes:
        cam_img = cv2.imdecode(np.frombuffer(cam_frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        if cam_img is not None:
            cam_h, cam_w = cam_img.shape[:2]
            ai_h, ai_w   = img.shape[:2]
            if (ai_w, ai_h) != (cam_w, cam_h):
                log.info(f"[AI] Resizing AI image {ai_w}×{ai_h} → {cam_w}×{cam_h} to match camera")
                img = cv2.resize(img, (cam_w, cam_h), interpolation=cv2.INTER_AREA)

    results = yolo_model(img, conf=conf, classes=TARGET_CLASSES, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2))

    log.info(f"[AI] YOLO found {len(boxes)} cars in generated image")

    slots = {}
    counter = 1
    for x1, y1, x2, y2 in boxes:
        while True:
            candidate = f"S{counter:02d}"
            if candidate not in existing_slot_ids and candidate not in slots:
                break
            counter += 1
        slots[candidate] = {
            "coords": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "source": "ai_generated",
        }
        counter += 1

    return slots
