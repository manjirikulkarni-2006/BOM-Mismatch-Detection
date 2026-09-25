from fastapi import FastAPI, UploadFile, File

from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
import cv2
import io

import numpy as np

import torch

import easyocr

from PIL import Image

from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# ============================================================

# 1. APPLICATION SETUP

# ============================================================

app = FastAPI(
    title="BOM Mismatch Detection API",
    description="Deep Learning-Based BOM Mismatch Detection System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================

# 2. MODEL CONFIGURATION

# ============================================================

BASE_MODEL = "microsoft/layoutlmv3-base"

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "layoutlmv3_bom_corrected_best.pth"
)

LABELS = [
    "O",
    "B-PART_NO",
    "I-PART_NO",
    "B-DESCRIPTION",
    "I-DESCRIPTION",
    "B-MATERIAL",
    "I-MATERIAL",
    "B-UOM",
    "I-UOM",
    "B-QTY",
    "I-QTY",
]

label2id = {label: idx for idx, label in enumerate(LABELS)}

id2label = {idx: label for idx, label in enumerate(LABELS)}

# ============================================================

# 3. DEVICE

# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================

# 4. LOAD LAYOUTLM PROCESSOR

# ============================================================

processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL, apply_ocr=False)

# ============================================================

# 5. LOAD LAYOUTLM MODEL

# ============================================================

model = LayoutLMv3ForTokenClassification.from_pretrained(
    BASE_MODEL, num_labels=len(LABELS), id2label=id2label, label2id=label2id
)

# ============================================================

# 6. LOAD TRAINED WEIGHTS

# ============================================================

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)

model.eval()

# ============================================================

# 7. LOAD EASYOCR

# ============================================================

reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

def run_bom_ocr(bom_crop):
    """Run improved EasyOCR preprocessing and preserve original-crop coordinates."""
    image_array = np.array(bom_crop)
    scale = 2.0

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    upscaled = cv2.resize(
        gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )
    denoised = cv2.fastNlMeansDenoising(
        upscaled, None, h=10, templateWindowSize=7, searchWindowSize=21
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11
    )
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    processed = cv2.filter2D(binary, -1, kernel)

    ocr_results = reader.readtext(
        processed, detail=1, paragraph=False,
        text_threshold=0.6, low_text=0.3, link_threshold=0.3,
        mag_ratio=1.5, width_ths=0.4, height_ths=0.4,
        contrast_ths=0.05, adjust_contrast=0.7
    )

    corrected_results = []
    for bbox, text, confidence in ocr_results:
        corrected_bbox = [
            [float(point[0]) / scale, float(point[1]) / scale]
            for point in bbox
        ]
        corrected_results.append((corrected_bbox, text, confidence))

    return corrected_results


# ============================================================

# 8. OCR ROW RECONSTRUCTION

# ============================================================


def reconstruct_ocr_rows(ocr_results, y_tolerance=8):
    """

    Convert EasyOCR detections into rows.

    EasyOCR returns:

        (bbox, text, confidence)

    Each row is sorted from left to right.

    """

    detections = []

    for detection in ocr_results:

        bbox, text, confidence = detection

        bbox = [[float(point[0]), float(point[1])] for point in bbox]

        text = str(text).strip()

        if not text:

            continue

        xs = [point[0] for point in bbox]

        ys = [point[1] for point in bbox]

        center_x = sum(xs) / len(xs)

        center_y = sum(ys) / len(ys)

        height = max(ys) - min(ys)

        detections.append(
            {
                "text": text,
                "bbox": bbox,
                "confidence": float(confidence),
                "center_x": center_x,
                "center_y": center_y,
                "height": height,
            }
        )

    # ---------------------------------------------------------

    # Sort detections top-to-bottom

    # ---------------------------------------------------------

    detections.sort(key=lambda item: item["center_y"])

    # ---------------------------------------------------------

    # Group detections into rows

    # ---------------------------------------------------------

    rows = []

    for detection in detections:

        placed = False

        for row in rows:

            row_center_y = sum(item["center_y"] for item in row) / len(row)

            if abs(detection["center_y"] - row_center_y) <= y_tolerance:

                row.append(detection)

                placed = True

                break

        if not placed:

            rows.append([detection])

    # ---------------------------------------------------------

    # Sort each row left-to-right

    # ---------------------------------------------------------

    rows.sort(key=lambda row: min(item["center_y"] for item in row))

    reconstructed_rows = []

    for row_id, row in enumerate(rows):

        row.sort(key=lambda item: item["center_x"])

        reconstructed_rows.append(
            {
                "row_id": row_id,
                "text": " | ".join(item["text"] for item in row),
                "tokens": row,
            }
        )

    return reconstructed_rows


# ============================================================

# 9. LAYOUTLMV3 BOM FIELD EXTRACTION

# ============================================================


def extract_bom_fields(rows, bom_crop):
    """

    Uses the trained LayoutLMv3 model to assign

    OCR words to BOM fields.

    """

    extracted_rows = []

    crop_width, crop_height = bom_crop.size

    for row in rows:

        tokens = row["tokens"]

        if not tokens:

            continue

        # ----------------------------------------------------

        # Prepare OCR words and bounding boxes

        # ----------------------------------------------------

        words = [token["text"] for token in tokens]

        boxes = [token["bbox"] for token in tokens]

        normalized_boxes = []

        for bbox in boxes:

            x_coordinates = [point[0] for point in bbox]

            y_coordinates = [point[1] for point in bbox]

            x0 = min(x_coordinates)

            y0 = min(y_coordinates)

            x1 = max(x_coordinates)

            y1 = max(y_coordinates)

            # Normalize coordinates to LayoutLM range 0-1000

            x0 = int((x0 / crop_width) * 1000)

            y0 = int((y0 / crop_height) * 1000)

            x1 = int((x1 / crop_width) * 1000)

            y1 = int((y1 / crop_height) * 1000)

            normalized_boxes.append(
                [
                    max(0, min(1000, x0)),
                    max(0, min(1000, y0)),
                    max(0, min(1000, x1)),
                    max(0, min(1000, y1)),
                ]
            )

        # ----------------------------------------------------

        # LayoutLMv3 encoding

        # ----------------------------------------------------

        encoding = processor(
            images=bom_crop,
            text=words,
            boxes=normalized_boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256,
        )

        # Keep the original BatchEncoding object.

        # This is required for encoding.word_ids().

        model_inputs = {
            key: value.to(device)
            for key, value in encoding.items()
            if isinstance(value, torch.Tensor)
        }

        # ----------------------------------------------------

        # Model inference

        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(**model_inputs)

        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

        # Map subword tokens back to original OCR words

        word_ids = encoding.word_ids(batch_index=0)

        # ----------------------------------------------------

        # Store predicted fields

        # ----------------------------------------------------

        field_values = {
            "PART_NO": [],
            "DESCRIPTION": [],
            "MATERIAL": [],
            "UOM": [],
            "QTY": [],
        }

        previous_word_id = None

        for token_index, word_id in enumerate(word_ids):

            if word_id is None:

                continue

            # Ignore additional subword tokens

            if word_id == previous_word_id:

                continue

            previous_word_id = word_id

            if word_id >= len(words):

                continue

            predicted_label = id2label[predictions[token_index]]

            word = words[word_id]

            if predicted_label in ["B-PART_NO", "I-PART_NO"]:

                field_values["PART_NO"].append(word)

            elif predicted_label in ["B-DESCRIPTION", "I-DESCRIPTION"]:

                field_values["DESCRIPTION"].append(word)

            elif predicted_label in ["B-MATERIAL", "I-MATERIAL"]:

                field_values["MATERIAL"].append(word)

            elif predicted_label in ["B-UOM", "I-UOM"]:

                field_values["UOM"].append(word)

            elif predicted_label in ["B-QTY", "I-QTY"]:

                field_values["QTY"].append(word)

        # ----------------------------------------------------

        # Create structured BOM row

        # ----------------------------------------------------

        structured_row = {
            "row_id": row["row_id"],
            "PART_NO": " ".join(field_values["PART_NO"]).strip(),
            "DESCRIPTION": " ".join(field_values["DESCRIPTION"]).strip(),
            "MATERIAL": " ".join(field_values["MATERIAL"]).strip(),
            "UOM": " ".join(field_values["UOM"]).strip(),
            "QTY": " ".join(field_values["QTY"]).strip(),
        }

        extracted_rows.append(structured_row)

    return extracted_rows


# ============================================================

# 10. BOM OUTPUT CLEANUP

# ============================================================


def clean_structured_bom(structured_bom):
    """

    Removes obvious title/header rows and empty rows.

    """

    cleaned_bom = []

    for row in structured_bom:

        values = {
            key: str(row.get(key, "")).strip()
            for key in ["PART_NO", "DESCRIPTION", "MATERIAL", "UOM", "QTY"]
        }

        combined_text = " ".join(values.values()).upper()

        # Remove obvious title/header rows

        if (
            "BILL OF MATERIALS" in combined_text
            or (
                "DESCRIPTION" in combined_text
                and ("MATERIAL" in combined_text or "MATL" in combined_text)
            )
            or ("PART NO" in combined_text and "QTY" in combined_text)
        ):

            continue

        # Remove completely empty rows

        if not any(values.values()):

            continue

        cleaned_bom.append({"row_id": row["row_id"], **values})

    return cleaned_bom


# ============================================================

# PART NUMBER NORMALIZATION

# ============================================================

import re

from difflib import SequenceMatcher


def normalize_part_number(value, reference_part_numbers, threshold=0.85):
    """

    Normalize an OCR-extracted PART_NO against a

    caller-supplied PART_NO vocabulary.

    Returns the original value when no sufficiently

    strong match exists.

    NOTE (UPDATED): this is a LEGACY helper. It is NO LONGER CALLED
    anywhere in the pipeline. The old comparison stage in
    /analyze-blueprint passed it the CURRENT reference BOM's PART_NO
    values, which let the evaluation reference influence the predicted
    PART_NO (test-reference leakage). It is kept only so that no working
    function is removed. Never call it with a reference/test BOM.
    The training-only replacement is correct_part_number() in
    section 10B below.
    """

    value = str(value).strip()

    if not value:

        return value

    def canonical(text):

        return re.sub(r"[^A-Z0-9]", "", str(text).upper())

    candidate = canonical(value)

    if not candidate:

        return value

    best_match = None

    best_score = 0.0

    for reference_part in reference_part_numbers:

        reference_canonical = canonical(reference_part)

        if not reference_canonical:

            continue

        score = SequenceMatcher(None, candidate, reference_canonical).ratio()

        if score > best_score:

            best_score = score

            best_match = reference_part

    if best_match is not None and best_score >= threshold:

        return best_match

    return value


# ===== OCR IMPROVEMENT START =====
#
# ============================================================
# 10B. TRAINING-ONLY OCR CORRECTION (PART_NO / DESCRIPTION / MATERIAL)
# ============================================================
#
# GOAL
# ----
# Correct obvious OCR/LayoutLMv3 extraction noise in PART_NO, DESCRIPTION
# and MATERIAL using vocabularies built ONLY from the training ground
# truth, never from the reference/ground-truth row of the blueprint
# currently being analyzed (the "test" blueprint). QTY/UOM are untouched.
#
# LEAKAGE GUARD
# -------------
# build_training_vocabularies() reads ONLY
#   data/ground_truth/blueprint_ground_truth.csv
# and explicitly EXCLUDES any row that belongs to the blueprint currently
# being processed: (a) the row with the same filename stem, and (b) rows
# that are augmentation variants of the same underlying blueprint
# (e.g. "<id>" vs "<id>_noisy"), see _blueprint_base_id(). It NEVER reads
# the uploaded --reference_bom CSV, and it never receives the
# "reference"/"reference_rows" variables used later for comparison. This
# is the only data source for correction.
#
# CONSERVATIVE MATCHING
# ----------------------
# 1. Canonicalize (uppercase, strip separators for comparison only).
# 2. Exact canonical match against the training vocabulary -> confidence 1.0.
# 3. Otherwise, fuzzy match against the training vocabulary. A correction
#    is only ever applied if the resulting similarity score clears
#    `threshold`. Nothing is substituted "blindly" -- a real vocabulary
#    entry must still be found.
# 4. If no confident match exists, the value is returned UNCHANGED
#    (this preserves genuinely new/unknown part numbers, descriptions,
#    and materials instead of forcing them onto the nearest vocabulary
#    entry).
#
# PART_NO (UPDATED) uses a CONFUSION-WEIGHTED EDIT DISTANCE instead of
# the earlier "single-character confusion variant + difflib ratio" probe.
# The old probe could only forgive ONE confusable character per value, so
# a PART_NO with several independent OCR confusions (A/4, Z/2, S/5 in the
# same string) could never clear the threshold. The new method
# (correct_part_number) charges a small cost for every confusable
# substitution and a full cost for any other edit, so any NUMBER of
# confusable characters is tolerated as long as the whole string stays
# very close to a single training PART_NO, and the match is unambiguous.
# DESCRIPTION and MATERIAL still use the earlier logic below, unchanged.
#
# DESCRIPTION uses two tiers: first a whole-phrase match against training
# DESCRIPTION strings (catches full catalog phrases such as
# "Lithonia LED Troffer 2x4"), and only if that is not confident enough,
# a conservative WORD-BY-WORD correction against a training vocabulary of
# individual words (catches things like "AwG" -> "AWG" inside an
# otherwise-correct or novel description, without touching numbers/sizes
# such as "2x4" or "1'").
#
# All thresholds below are deliberately conservative starting points.
# They are ordinary keyword arguments / module constants -- raise them if
# you see any false corrections in your own dataset, lower them if
# genuinely-correct OCR is failing to match an obvious training vocabulary
# entry.

GROUND_TRUTH_CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "ground_truth"
    / "blueprint_ground_truth.csv"
)

# Conservative single-character OCR confusion table for blueprint text.
# Used only to WIDEN the vocabulary search, never to directly rewrite text.
# (Still used by DESCRIPTION / MATERIAL correction. PART_NO now uses the
# symmetric pair table further below.)
_CHAR_CONFUSIONS = {
    "O": "0",
    "0": "O",
    "I": "1",
    "1": "I",
    "L": "1",
    "S": "5",
    "5": "S",
    "B": "8",
    "8": "B",
    "Z": "2",
    "2": "Z",
    "G": "6",
    "6": "G",
    "T": "7",
    "7": "T",
}

# ------------------------------------------------------------
# PART_NO correction settings (added)
# ------------------------------------------------------------

# Symmetric OCR confusion pairs used by the confusion-weighted distance.
# A pair is "confusable" in BOTH directions (A<->4 as well as 4<->A).
_OCR_CONFUSION_PAIRS = [
    ("A", "4"),
    ("Z", "2"),
    ("O", "0"),
    ("Q", "0"),
    ("D", "0"),
    ("I", "1"),
    ("L", "1"),
    ("S", "5"),
    ("B", "8"),
    ("G", "6"),
    ("T", "7"),
]

_CONFUSABLE_PAIRS = {frozenset(pair) for pair in _OCR_CONFUSION_PAIRS}

# Cost of substituting one confusable character for another.
# Any other substitution / insertion / deletion costs 1.0.
CONFUSION_SUB_COST = 0.30

# Minimum similarity (1 - weighted_distance / longer_length) required
# before a PART_NO is changed.
PART_NO_CONFIDENCE_THRESHOLD = 0.90

# The best training candidate must beat the runner-up by at least this
# much, otherwise the correction is considered ambiguous and skipped.
PART_NO_AMBIGUITY_MARGIN = 0.03

# Very short values carry too little evidence for fuzzy correction
# (exact training matches are still applied).
PART_NO_MIN_LENGTH = 5

# ------------------------------------------------------------
# Train/test separation settings (added)
# ------------------------------------------------------------

# The ground-truth CSV holds every project blueprint, so the blueprint
# being analyzed is excluded from the correction vocabulary. When True,
# augmentation variants of the same underlying blueprint (same name plus
# a suffix such as "_noisy") are excluded too, because a clean/noisy pair
# shares the same BOM and would otherwise leak it into the vocabulary.
EXCLUDE_AUGMENTATION_SIBLINGS = True

_AUGMENTATION_SUFFIX_TOKENS = {
    "noisy",
    "noise",
    "blur",
    "blurred",
    "rotated",
    "rotate",
    "skew",
    "skewed",
    "scan",
    "scanned",
    "aug",
    "augmented",
    "copy",
    "dark",
    "faded",
    "lowres",
    "compressed",
    "clean",
    "test",
}


def _blueprint_base_id(filename_stem):
    """
    Reduce a filename stem to the underlying blueprint identity by
    stripping trailing augmentation tokens, e.g.
        "<id>_noisy" -> "<id>",  "<id>" -> "<id>".
    Used only to keep sibling variants of the CURRENT blueprint out of the
    training vocabulary. It does not look at any BOM content.
    """
    tokens = [
        token
        for token in re.split(r"[_\-\s]+", str(filename_stem or "").strip().lower())
        if token
    ]

    while len(tokens) > 1 and tokens[-1] in _AUGMENTATION_SUFFIX_TOKENS:
        tokens.pop()

    return "_".join(tokens)


def _canonical(text):
    """Uppercase and strip everything except letters/digits, for comparison only."""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def _confusion_variants(canonical_text):
    """
    Generate conservative SINGLE-character-substitution variants of an
    already-canonicalized string, using known blueprint OCR confusions.
    Only one character is changed per variant (never combinatorial /
    blind across the whole string). Always includes the original text.
    """
    variants = {canonical_text}
    for i, ch in enumerate(canonical_text):
        replacement = _CHAR_CONFUSIONS.get(ch)
        if replacement:
            variants.add(canonical_text[:i] + replacement + canonical_text[i + 1:])
    return variants


def _best_vocab_match(candidate_canonical, vocab_canonical_map, use_confusion=True):
    """
    candidate_canonical: canonical form of the OCR value being corrected.
    vocab_canonical_map: dict {canonical_form: original_training_value}.
    Returns (best_original_training_value_or_None, best_score).
    """
    if not candidate_canonical or not vocab_canonical_map:
        return None, 0.0

    search_forms = (
        _confusion_variants(candidate_canonical)
        if use_confusion
        else {candidate_canonical}
    )

    best_value = None
    best_score = 0.0

    for form in search_forms:
        for vocab_canonical, vocab_original in vocab_canonical_map.items():
            score = SequenceMatcher(None, form, vocab_canonical).ratio()
            if score > best_score:
                best_score = score
                best_value = vocab_original

    return best_value, best_score


def _confusion_weighted_distance(a, b):
    """
    Levenshtein-style edit distance where substituting two characters that
    OCR commonly confuses (see _OCR_CONFUSION_PAIRS) costs
    CONFUSION_SUB_COST, while every other substitution, insertion and
    deletion costs 1.0. Operates on canonical (upper-case alphanumeric)
    strings.
    """
    len_a = len(a)
    len_b = len(b)

    previous = [float(j) for j in range(len_b + 1)]

    for i in range(1, len_a + 1):

        current = [float(i)] + [0.0] * len_b

        for j in range(1, len_b + 1):

            char_a = a[i - 1]
            char_b = b[j - 1]

            if char_a == char_b:
                substitution_cost = 0.0
            elif frozenset((char_a, char_b)) in _CONFUSABLE_PAIRS:
                substitution_cost = CONFUSION_SUB_COST
            else:
                substitution_cost = 1.0

            current[j] = min(
                previous[j] + 1.0,
                current[j - 1] + 1.0,
                previous[j - 1] + substitution_cost,
            )

        previous = current

    return previous[len_b]


def _confusion_similarity(a, b):
    """Similarity in [0, 1]: 1 - weighted_distance / length_of_longer_string."""
    longest = max(len(a), len(b))

    if longest == 0:
        return 0.0

    return max(0.0, 1.0 - _confusion_weighted_distance(a, b) / longest)


def _print_stage_banner(title):
    """Terminal marker separating pipeline stages (model prediction /
    OCR correction / final reference comparison) for viva explanation."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _print_bom_snapshot(rows):
    """Compact one-line-per-row dump of a structured BOM for debugging."""
    for row in rows:
        print(
            f"  row {row.get('row_id')}: "
            f"{row.get('PART_NO', '')} | {row.get('DESCRIPTION', '')} | "
            f"{row.get('MATERIAL', '')} | {row.get('UOM', '')} | "
            f"{row.get('QTY', '')}"
        )


def _print_correction_debug(field_name, raw, normalized, corrected, source, confidence):
    """Debug trace: RAW -> NORMALIZED -> CORRECTED -> FINAL, per the academic
    write-up requirement, so PART_NO/DESCRIPTION/MATERIAL corrections can be
    explained during a viva."""
    print(f"--- {field_name} ---")
    print(f"RAW: {raw}")
    print(f"NORMALIZED: {normalized}")
    if source == "none":
        print("CORRECTION: none")
    else:
        print(f"CORRECTED: {corrected}")
        print(f"SOURCE: {source}")
        print(f"CONFIDENCE: {confidence:.2f}")
    print(f"FINAL: {corrected}")
    print("-" * 50)


def _print_part_no_debug(
    raw,
    normalized,
    corrected,
    source,
    confidence,
    best_candidate,
    best_score,
    runner_up,
    runner_up_score,
    note,
):
    """PART_NO-specific debug trace (OCR-correction stage only)."""
    print("--- PART_NO (OCR correction, training vocabulary only) ---")
    print(f"RAW OCR: {raw}")
    print(f"NORMALIZED: {normalized}")
    if source == "none":
        print("CORRECTED: none")
        print("SOURCE: none")
        print("CONFIDENCE: n/a")
    else:
        print(f"CORRECTED: {corrected}")
        print(f"SOURCE: {source}")
        print(f"CONFIDENCE: {confidence:.2f}")
    if best_candidate is not None:
        print(f"BEST TRAINING CANDIDATE: {best_candidate} (score {best_score:.2f})")
    if runner_up is not None:
        print(f"RUNNER-UP: {runner_up} (score {runner_up_score:.2f})")
    if note:
        print(f"NOTE: {note}")
    print(f"FINAL: {corrected}")
    print("-" * 50)


def build_training_vocabularies(exclude_filename_stem):
    """
    Build PART_NO / DESCRIPTION / MATERIAL correction vocabularies from the
    TRAINING ground truth only (data/ground_truth/blueprint_ground_truth.csv),
    explicitly excluding the row(s) belonging to the blueprint currently
    being analyzed (`exclude_filename_stem`) so the test blueprint can never
    leak into its own OCR correction.

    Excluded rows:
      1. the row whose filename stem equals `exclude_filename_stem`;
      2. when EXCLUDE_AUGMENTATION_SIBLINGS is True, rows that share the
         same underlying blueprint identity (same name after stripping
         augmentation suffixes such as "_noisy").

    Known limitation: the CSV does not say which blueprints the model was
    trained on, so every OTHER row is treated as training data. A
    dedicated train-split list would be needed for a stricter guarantee.
    """

    part_no_vocab = set()
    description_phrase_vocab = set()
    description_word_vocab = {}
    material_vocab = set()

    if not GROUND_TRUTH_CSV_PATH.exists():
        return {
            "part_no_vocab": part_no_vocab,
            "description_phrase_vocab": description_phrase_vocab,
            "description_word_vocab": description_word_vocab,
            "material_vocab": material_vocab,
        }

    import pandas as pd
    import json

    gt_df = pd.read_csv(GROUND_TRUTH_CSV_PATH)

    exclude_stem = str(exclude_filename_stem or "").strip().lower()
    exclude_base = _blueprint_base_id(exclude_stem) if exclude_stem else ""

    excluded_rows = []
    used_rows = 0

    for _, row in gt_df.iterrows():

        gt_filename = Path(str(row.get("filename", ""))).stem.strip().lower()

        # ----- LEAKAGE GUARD -----
        # Skip the ground-truth row(s) belonging to the blueprint that is
        # currently being processed. Everything else in this CSV is treated
        # as "training data" for vocabulary purposes.
        if exclude_stem:

            if gt_filename == exclude_stem:
                excluded_rows.append(gt_filename)
                continue

            if (
                EXCLUDE_AUGMENTATION_SIBLINGS
                and exclude_base
                and _blueprint_base_id(gt_filename) == exclude_base
            ):
                excluded_rows.append(gt_filename)
                continue

        try:
            data = json.loads(row["json_data"])
        except (ValueError, TypeError, KeyError):
            continue

        used_rows += 1

        for item in data.get("bill_of_materials", []):

            part_no = str(item.get("part_no", "")).strip()
            description = str(item.get("description", "")).strip()
            material = str(item.get("material", "")).strip()

            if part_no:
                part_no_vocab.add(part_no)

            if description:
                description_phrase_vocab.add(description)
                for word in description.split():
                    word_key = re.sub(r"[^A-Za-z0-9]", "", word).upper()
                    if word_key:
                        description_word_vocab.setdefault(word_key, word)

            if material:
                material_vocab.add(material)

    print(
        f"[TRAINING VOCABULARY] excluded ground-truth rows for "
        f"'{exclude_stem}': {excluded_rows if excluded_rows else 'none found'}"
    )
    print(
        f"[TRAINING VOCABULARY] rows used: {used_rows} | "
        f"PART_NO: {len(part_no_vocab)} | "
        f"DESCRIPTION phrases: {len(description_phrase_vocab)} | "
        f"MATERIAL: {len(material_vocab)}"
    )

    return {
        "part_no_vocab": part_no_vocab,
        "description_phrase_vocab": description_phrase_vocab,
        "description_word_vocab": description_word_vocab,
        "material_vocab": material_vocab,
    }


def _normalize_separators(text):
    """Light, format-preserving normalization: uppercase, collapse
    accidental whitespace, and tidy spaces around '-' and '/'."""
    text = str(text or "").strip().upper()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s*/\s*", "/", text)
    return text.strip()


def correct_whole_value(raw_value, vocab, threshold, field_name, use_confusion=True):
    """
    Generic whole-value corrector used for MATERIAL (a closed/near-closed
    vocabulary of catalog materials). PART_NO now uses
    correct_part_number() below.
    """
    raw_value = str(raw_value or "")
    normalized = _normalize_separators(raw_value)

    corrected = normalized
    source = "none"
    confidence = 0.0

    candidate_canonical = _canonical(normalized)

    if candidate_canonical and vocab:

        vocab_canonical_map = {_canonical(v): v for v in vocab}

        if candidate_canonical in vocab_canonical_map:
            corrected = vocab_canonical_map[candidate_canonical]
            source = "training vocabulary (exact)"
            confidence = 1.0
        else:
            best_value, best_score = _best_vocab_match(
                candidate_canonical, vocab_canonical_map, use_confusion=use_confusion
            )
            if best_value is not None and best_score >= threshold:
                corrected = best_value
                source = "training vocabulary (fuzzy)"
                confidence = best_score

    _print_correction_debug(
        field_name, raw_value, normalized, corrected, source, confidence
    )

    return corrected


def correct_part_number(
    raw_value,
    vocab,
    threshold=PART_NO_CONFIDENCE_THRESHOLD,
    margin=PART_NO_AMBIGUITY_MARGIN,
    min_length=PART_NO_MIN_LENGTH,
):
    """
    Training-vocabulary-only PART_NO correction.

    `vocab` must come from build_training_vocabularies(); the reference /
    test BOM is never an input to this function.

    Steps:
      1. Normalize (uppercase, tidy separators) and canonicalize
         (letters/digits only) the OCR value.
      2. Exact canonical match in the training vocabulary -> use the
         training spelling, confidence 1.0.
      3. Otherwise score EVERY training PART_NO with a confusion-weighted
         edit distance (confusable OCR substitutions such as A/4, Z/2,
         O/0, I/1, L/1, S/5, B/8, G/6, Q/0, D/0 cost CONFUSION_SUB_COST;
         all other edits cost 1.0), converted to a similarity in [0, 1].
      4. Correct only if:
           - the best similarity >= threshold, AND
           - the best candidate beats the runner-up by at least `margin`
             (otherwise the match is ambiguous), AND
           - the value has at least `min_length` characters.
      5. Otherwise keep the OCR value (only case/separator-normalized).
    """
    raw_value = str(raw_value or "")
    normalized = _normalize_separators(raw_value)
    candidate = _canonical(normalized)

    corrected = normalized
    source = "none"
    confidence = 0.0

    best_value = None
    best_score = 0.0
    runner_up_value = None
    runner_up_score = 0.0
    note = ""

    if candidate and vocab:

        vocab_canonical_map = {}

        for vocab_value in vocab:
            vocab_key = _canonical(vocab_value)
            if vocab_key:
                vocab_canonical_map.setdefault(vocab_key, vocab_value)

        if candidate in vocab_canonical_map:

            corrected = vocab_canonical_map[candidate]
            source = "training vocabulary (exact)"
            confidence = 1.0
            best_value = corrected
            best_score = 1.0

        elif len(candidate) >= min_length and vocab_canonical_map:

            scored = sorted(
                (
                    (_confusion_similarity(candidate, vocab_key), vocab_value)
                    for vocab_key, vocab_value in vocab_canonical_map.items()
                ),
                key=lambda item: (-item[0], item[1]),
            )

            best_score, best_value = scored[0]

            if len(scored) > 1:
                runner_up_score, runner_up_value = scored[1]

            if best_score < threshold:

                note = (
                    f"best score {best_score:.2f} is below threshold "
                    f"{threshold:.2f}; OCR value preserved"
                )

            elif (
                runner_up_value is not None
                and (best_score - runner_up_score) < margin
            ):

                note = (
                    f"ambiguous: best and runner-up scores differ by less "
                    f"than {margin:.2f}; OCR value preserved"
                )

            else:

                corrected = best_value
                source = "training vocabulary (confusion-weighted fuzzy)"
                confidence = best_score

        elif candidate:

            note = (
                f"value shorter than {min_length} characters; only exact "
                f"training matches are applied"
            )

    elif candidate:

        note = "training PART_NO vocabulary is empty; OCR value preserved"

    _print_part_no_debug(
        raw_value,
        normalized,
        corrected,
        source,
        confidence,
        best_value,
        best_score,
        runner_up_value,
        runner_up_score,
        note,
    )

    return corrected


def correct_description(
    raw_value,
    phrase_vocab,
    word_vocab,
    phrase_threshold=0.85,
    word_threshold=0.80,
):
    """
    Two-tier DESCRIPTION correction:
      Tier 1: whole-phrase match against training DESCRIPTION strings.
      Tier 2: conservative word-by-word correction against a training
              word vocabulary, only replacing individual words that
              clear `word_threshold`; unmatched/unknown words (including
              sizes like "2x4" or "1'") are preserved untouched.
    """
    raw_value = str(raw_value or "")
    normalized = re.sub(r"\s+", " ", raw_value.strip())

    corrected = normalized
    source = "none"
    confidence = 0.0

    if normalized:

        # ---- Tier 1: whole-phrase match ----
        phrase_canonical = _canonical(normalized)

        if phrase_canonical and phrase_vocab:

            phrase_vocab_map = {_canonical(p): p for p in phrase_vocab}

            if phrase_canonical in phrase_vocab_map:
                corrected = phrase_vocab_map[phrase_canonical]
                source = "training vocabulary (phrase, exact)"
                confidence = 1.0

            else:
                best_phrase, best_phrase_score = _best_vocab_match(
                    phrase_canonical, phrase_vocab_map, use_confusion=False
                )
                if best_phrase is not None and best_phrase_score >= phrase_threshold:
                    corrected = best_phrase
                    source = "training vocabulary (phrase, fuzzy)"
                    confidence = best_phrase_score

        # ---- Tier 2: conservative word-level fallback ----
        if source == "none" and word_vocab:

            words = normalized.split(" ")
            corrected_words = []
            matched_scores = []
            any_word_corrected = False

            for word in words:

                word_key = re.sub(r"[^A-Za-z0-9]", "", word).upper()

                if not word_key:
                    corrected_words.append(word)
                    continue

                if word_key in word_vocab:
                    corrected_words.append(word_vocab[word_key])
                    continue

                best_word_value, best_word_score = _best_vocab_match(
                    word_key, word_vocab, use_confusion=True
                )

                if best_word_value is not None and best_word_score >= word_threshold:
                    corrected_words.append(best_word_value)
                    matched_scores.append(best_word_score)
                    any_word_corrected = True
                else:
                    # Preserve genuinely unknown/new words untouched.
                    corrected_words.append(word)

            if any_word_corrected:
                corrected = " ".join(corrected_words)
                source = "training vocabulary (word-level)"
                confidence = min(matched_scores)

    _print_correction_debug(
        "DESCRIPTION", raw_value, normalized, corrected, source, confidence
    )

    return corrected


def apply_ocr_corrections(structured_bom, vocabularies):
    """
    Applies training-vocabulary-only corrections to PART_NO, DESCRIPTION
    and MATERIAL for every row. QTY and UOM are passed through unchanged.
    """
    part_no_vocab = vocabularies["part_no_vocab"]
    description_phrase_vocab = vocabularies["description_phrase_vocab"]
    description_word_vocab = vocabularies["description_word_vocab"]
    material_vocab = vocabularies["material_vocab"]

    corrected_bom = []

    for row in structured_bom:

        new_row = dict(row)

        print(f"\n===== OCR CORRECTION (row_id={row.get('row_id')}) =====")

        new_row["PART_NO"] = correct_part_number(
            row.get("PART_NO", ""),
            part_no_vocab,
        )

        new_row["DESCRIPTION"] = correct_description(
            row.get("DESCRIPTION", ""),
            description_phrase_vocab,
            description_word_vocab,
            phrase_threshold=0.85,
            word_threshold=0.80,
        )

        new_row["MATERIAL"] = correct_whole_value(
            row.get("MATERIAL", ""),
            material_vocab,
            threshold=0.82,
            field_name="MATERIAL",
            use_confusion=True,
        )

        corrected_bom.append(new_row)

    return corrected_bom


# ===== OCR IMPROVEMENT END =====


# ============================================================

# 11. ROOT ENDPOINT

# ============================================================


@app.get("/")
def root():

    return {
        "message": "BOM Mismatch Detection API is running",
        "model": "LayoutLMv3",
        "device": str(device),
        "model_loaded": True,
        "ocr_loaded": True,
    }


# ============================================================

# 12. MODEL STATUS

# ============================================================


@app.get("/model-status")
def model_status():

    return {
        "model_loaded": True,
        "ocr_loaded": True,
        "device": str(device),
        "model_path": str(MODEL_PATH),
        "labels": LABELS,
    }


# ============================================================

# 13. BLUEPRINT UPLOAD

# ============================================================


@app.post("/upload-blueprint")
async def upload_blueprint(file: UploadFile = File(...)):

    # --------------------------------------------------------

    # 1. Read uploaded image

    # --------------------------------------------------------

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    width, height = image.size

    # --------------------------------------------------------

    # 2. Crop BOM region

    #

    # Current project-layout assumption:

    # BOM is located in the lower-right region.

    # --------------------------------------------------------

    x1 = 760

    y1 = 515

    x2 = min(1245, width)

    y2 = min(760, height)

    bom_crop = image.crop((x1, y1, x2, y2))

   
# --------------------------------------------------------
# 3. Improved EasyOCR
# --------------------------------------------------------

    ocr_results = run_bom_ocr(bom_crop)

    # --------------------------------------------------------

    # 5. Convert OCR results

    # --------------------------------------------------------

    detections = []

    for bbox, text, confidence in ocr_results:

        detections.append(
            {
                "text": text,
                "bbox": [[int(point[0]), int(point[1])] for point in bbox],
                "confidence": float(confidence),
            }
        )

    # --------------------------------------------------------

    # 6. Reconstruct OCR rows

    # --------------------------------------------------------

    rows = reconstruct_ocr_rows(ocr_results)

    # --------------------------------------------------------

    # 7. LayoutLMv3 field extraction

    # --------------------------------------------------------

    structured_bom = extract_bom_fields(rows, bom_crop)

    # --------------------------------------------------------

    # 8. Clean BOM output

    # --------------------------------------------------------

    structured_bom = clean_structured_bom(structured_bom)

    _print_stage_banner("[STAGE 1] MODEL PREDICTION (LayoutLMv3, before any correction)")
    _print_bom_snapshot(structured_bom)

    # ===== OCR IMPROVEMENT START =====
    # 8B. Training-vocabulary-only OCR correction (PART_NO/DESCRIPTION/MATERIAL)
    _print_stage_banner("[STAGE 2] OCR CORRECTION (training vocabulary only)")

    uploaded_filename_stem = (
        Path(file.filename).stem.strip().lower() if file.filename else ""
    )

    training_vocabularies = build_training_vocabularies(
        exclude_filename_stem=uploaded_filename_stem
    )

    structured_bom = apply_ocr_corrections(structured_bom, training_vocabularies)

    _print_stage_banner("[STAGE 3] FINAL PREDICTED BOM (after OCR correction)")
    _print_bom_snapshot(structured_bom)
    # ===== OCR IMPROVEMENT END =====

    # --------------------------------------------------------

    # 9. Return result

    # --------------------------------------------------------

    return {
        "status": "success",
        "filename": file.filename,
        "image_width": width,
        "image_height": height,
        "bom_crop": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": x2 - x1,
            "height": y2 - y1,
        },
        "ocr_detections": len(detections),
        "ocr_rows": len(rows),
        "bom": structured_bom,
    }


# ============================================================

# 14. BOM MISMATCH DETECTION

# ============================================================

from pydantic import BaseModel

from typing import List


class BOMItem(BaseModel):

    PART_NO: str

    DESCRIPTION: str

    MATERIAL: str

    UOM: str

    QTY: str


class BOMComparisonRequest(BaseModel):

    reference_bom: List[BOMItem]

    detected_bom: List[BOMItem]


@app.post("/check-bom")
def check_bom(request: BOMComparisonRequest):

    reference = {
        item.PART_NO.strip(): item
        for item in request.reference_bom
        if item.PART_NO.strip()
    }

    detected = {
        item.PART_NO.strip(): item
        for item in request.detected_bom
        if item.PART_NO.strip()
    }

    results = []

    # --------------------------------------------------------

    # Compare detected BOM against reference BOM

    # --------------------------------------------------------

    for part_no, detected_item in detected.items():

        # Extra part

        if part_no not in reference:

            results.append(
                {
                    "PART_NO": part_no,
                    "FIELD": "ROW",
                    "EXPECTED": "Part should exist in reference BOM",
                    "DETECTED": "Extra part",
                    "STATUS": "MISMATCH",
                }
            )

            continue

        expected_item = reference[part_no]

        fields = ["DESCRIPTION", "MATERIAL", "UOM", "QTY"]

        for field in fields:

            expected_value = str(getattr(expected_item, field)).strip()

            detected_value = str(getattr(detected_item, field)).strip()

            status = (
                "MATCH" if expected_value == detected_value else "MISMATCH"
            )

            results.append(
                {
                    "PART_NO": part_no,
                    "FIELD": field,
                    "EXPECTED": expected_value,
                    "DETECTED": detected_value,
                    "STATUS": status,
                }
            )

    # --------------------------------------------------------

    # Detect missing parts

    # --------------------------------------------------------

    for part_no in reference:

        if part_no not in detected:

            results.append(
                {
                    "PART_NO": part_no,
                    "FIELD": "ROW",
                    "EXPECTED": "Part exists in reference BOM",
                    "DETECTED": "Missing part",
                    "STATUS": "MISMATCH",
                }
            )

    # --------------------------------------------------------

    # Calculate summary

    # --------------------------------------------------------

    total_fields = len(results)

    matching_fields = sum(
        1 for result in results if result["STATUS"] == "MATCH"
    )

    mismatched_fields = sum(
        1 for result in results if result["STATUS"] == "MISMATCH"
    )

    overall_status = "MATCH" if mismatched_fields == 0 else "MISMATCH DETECTED"

    return {
        "status": overall_status,
        "summary": {
            "total_fields_checked": total_fields,
            "matching_fields": matching_fields,
            "mismatched_fields": mismatched_fields,
        },
        "results": results,
    }


# ============================================================

# 15. COMPLETE BLUEPRINT ANALYSIS

# ============================================================


@app.post("/analyze-blueprint")
async def analyze_blueprint(
    blueprint: UploadFile = File(...),
    reference_bom: UploadFile | None = File(None),
):
    """

    Analyze an uploaded engineering blueprint.

    Reference BOM priority:

    1. User-uploaded reference BOM

    2. Project ground-truth BOM for known dataset blueprints

    3. No comparison for unknown blueprints

    The reference BOM is used ONLY in the final comparison/evaluation
    stage. It is never used for OCR correction or to modify predictions.

    """

    # ---------------------------------------------------------

    # 1. Read uploaded blueprint

    # ---------------------------------------------------------

    image_bytes = await blueprint.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    width, height = image.size

    # ---------------------------------------------------------

    # 2. Select BOM crop

    # ---------------------------------------------------------

    dynamic_crop_dir = (
        Path(__file__).resolve().parent.parent
        / "outputs"
        / "dynamic_bom_crops"
    )

    uploaded_stem = Path(blueprint.filename).stem.strip().lower()

    dynamic_crop_path = dynamic_crop_dir / f"{uploaded_stem}_dynamic_bom.png"

    if dynamic_crop_path.exists():

        bom_crop = Image.open(dynamic_crop_path).convert("RGB")

        crop_source = "saved_dynamic_crop"

        x1 = 0

        y1 = 0

        x2 = bom_crop.width

        y2 = bom_crop.height

    else:

        # Fallback for an unseen blueprint

        x1 = 760

        y1 = 515

        x2 = min(1245, width)

        y2 = min(760, height)

        bom_crop = image.crop((x1, y1, x2, y2))

        crop_source = "fallback_crop"

    # ---------------------------------------------------------

    # 3. OCR

    # ---------------------------------------------------------

    # Improved EasyOCR
    ocr_results = run_bom_ocr(bom_crop)

    rows = reconstruct_ocr_rows(ocr_results)

    # ---------------------------------------------------------

    # 4. LayoutLMv3 BOM extraction

    # ---------------------------------------------------------

    extracted_bom = extract_bom_fields(rows, bom_crop)

    extracted_bom = clean_structured_bom(extracted_bom)

    _print_stage_banner("[STAGE 1] MODEL PREDICTION (LayoutLMv3, before any correction)")
    _print_bom_snapshot(extracted_bom)

    # ===== OCR IMPROVEMENT START =====
    # 4B. Training-vocabulary-only OCR correction (PART_NO/DESCRIPTION/MATERIAL)
    #
    # This runs BEFORE the reference BOM (user-uploaded or project ground
    # truth) is even loaded below, and uses ONLY
    # data/ground_truth/blueprint_ground_truth.csv with the current
    # blueprint's own row (and its augmentation siblings) excluded. It
    # therefore cannot see, and cannot be influenced by, whatever
    # reference/ground-truth is used later for comparison -- preventing
    # test-set leakage.
    _print_stage_banner("[STAGE 2] OCR CORRECTION (training vocabulary only)")

    training_vocabularies = build_training_vocabularies(
        exclude_filename_stem=uploaded_stem
    )

    extracted_bom = apply_ocr_corrections(extracted_bom, training_vocabularies)

    _print_stage_banner("[STAGE 3] FINAL PREDICTED BOM (after OCR correction)")
    _print_bom_snapshot(extracted_bom)
    # ===== OCR IMPROVEMENT END =====

    # ---------------------------------------------------------

    # 5. Determine reference BOM

    reference = None
    reference_source = None
    comparison = None

    # Use user-uploaded reference BOM when provided
    if reference_bom is not None and reference_bom.filename:

        reference_bytes = await reference_bom.read()
        reference_text = reference_bytes.decode("utf-8-sig")

        import csv

        csv_reader = csv.DictReader(io.StringIO(reference_text))

        reference = []

        for row in csv_reader:
            reference.append(
                {
                    "PART_NO": str(row.get("PART_NO", "")).strip(),
                    "DESCRIPTION": str(row.get("DESCRIPTION", "")).strip(),
                    "MATERIAL": str(row.get("MATERIAL", "")).strip(),
                    "UOM": str(row.get("UOM", "")).strip(),
                    "QTY": str(row.get("QTY", "")).strip(),
                }
            )

        reference_source = "user_uploaded"

    # Otherwise use project ground truth when the uploaded
    # filename belongs to the project dataset
    else:

        ground_truth_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "ground_truth"
            / "blueprint_ground_truth.csv"
        )

        print("GROUND TRUTH PATH:", ground_truth_path)
        print("GROUND TRUTH EXISTS:", ground_truth_path.exists())
        print("UPLOADED FILENAME:", blueprint.filename)

        if ground_truth_path.exists():

            import pandas as pd
            import json

            gt_df = pd.read_csv(ground_truth_path)

            uploaded_filename = Path(blueprint.filename).stem.strip().lower()

            matching_files = gt_df[
                gt_df["filename"]
                .astype(str)
                .apply(
                    lambda x: Path(x).stem.strip().lower() == uploaded_filename
                )
            ]["filename"].tolist()

            print("MATCHES:", matching_files)

            for _, row in gt_df.iterrows():

                gt_filename = Path(str(row["filename"])).stem.strip().lower()

                if gt_filename != uploaded_filename:
                    continue

                data = json.loads(row["json_data"])

                reference = []

                for item in data["bill_of_materials"]:
                    reference.append(
                        {
                            "PART_NO": str(item.get("part_no", "")).strip(),
                            "DESCRIPTION": str(
                                item.get("description", "")
                            ).strip(),
                            "MATERIAL": str(item.get("material", "")).strip(),
                            "UOM": str(item.get("uom", "")).strip(),
                            "QTY": str(item.get("qty", "")).strip(),
                        }
                    )

                reference_source = "project_ground_truth"

                print("REFERENCE ROWS LOADED:", len(reference))

                break

        # NOTE: the old "PART_NO NORMALIZATION DIAGNOSTIC" block that used
        # to live here has been REMOVED. It built its lookup vocabulary
        # from this blueprint's own ground-truth `reference` rows, which is
        # exactly the test-set leakage this update is meant to eliminate.
        # The equivalent (and now leakage-safe) debug output is produced
        # above by apply_ocr_corrections(), using build_training_vocabularies()
        # instead of `reference`.

    if reference is not None:

        _print_stage_banner(
            "[STAGE 4] FINAL REFERENCE COMPARISON "
            "(reference used for evaluation only; predictions are not modified)"
        )

        # -----------------------------------------------------

        # 6. BOM comparison

        # -----------------------------------------------------

        def normalize_compare_text(value):
            """Normalize OCR formatting for comparison only."""

            value = str(value or "").upper().strip()

            return re.sub(r"\s+", "", value)

        def normalize_compare_qty(value):
            """Normalize quantity strings such as '476 . 0' and '476.0'."""

            value = str(value or "").strip()

            value = re.sub(r"\s+", "", value)

            value = value.replace(",", "")

            try:

                return float(value)

            except (ValueError, TypeError):

                return None

        def is_bom_header(row):

            values = [
                str(row.get("PART_NO", "")).strip().upper(),
                str(row.get("DESCRIPTION", "")).strip().upper(),
                str(row.get("MATERIAL", "")).strip().upper(),
                str(row.get("UOM", "")).strip().upper(),
                str(row.get("QTY", "")).strip().upper(),
            ]

            combined = " ".join(values)

            return (
                "BILL OF MATERIALS" in combined
                or ("PART NO" in combined and "DESCRIPTION" in combined)
                or ("DESCRIPTION" in combined and "MATERIAL" in combined)
            )

        # Remove header/empty rows before comparison.

        reference_rows = [
            row
            for row in reference
            if not is_bom_header(row)
            and any(
                str(row.get(field, "")).strip()
                for field in [
                    "PART_NO",
                    "DESCRIPTION",
                    "MATERIAL",
                    "UOM",
                    "QTY",
                ]
            )
        ]

        detected_rows = [
            row
            for row in extracted_bom
            if not is_bom_header(row)
            and any(
                str(row.get(field, "")).strip()
                for field in [
                    "PART_NO",
                    "DESCRIPTION",
                    "MATERIAL",
                    "UOM",
                    "QTY",
                ]
            )
        ]

        # LEAKAGE FIX: the previous version built `reference_part_numbers`
        # from the reference BOM and used normalize_part_number() to
        # rewrite each DETECTED PART_NO toward the reference vocabulary.
        # That let the evaluation reference change the prediction, so it
        # has been removed. Both sides now keep their own PART_NO
        # untouched; the detected value is the FINAL PREDICTED value from
        # the training-only correction stage above.

        for row in reference_rows:

            row["_MATCH_PART_NO"] = str(row.get("PART_NO", "")).strip()

        for row in detected_rows:

            row["_MATCH_PART_NO"] = str(row.get("PART_NO", "")).strip()

        comparison_rows = []

        used_reference_indices = set()

        fields = ["DESCRIPTION", "MATERIAL", "UOM", "QTY"]

        def part_key(value):

            return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())

        def find_reference_match(detected_item):
            # Evaluation-side ROW ALIGNMENT only. It decides which
            # reference row a predicted row is compared with; it never
            # changes any predicted value.

            detected_part = detected_item.get("_MATCH_PART_NO", "")

            detected_key = part_key(detected_part)

            candidates = []

            for idx, ref_item in enumerate(reference_rows):

                if idx in used_reference_indices:

                    continue

                ref_part = ref_item.get("_MATCH_PART_NO", "")

                ref_key = part_key(ref_part)

                if detected_key and detected_key == ref_key:

                    candidates.append(idx)

            # For duplicate PART_NO occurrences, use quantity

            # to select the corresponding occurrence when possible.

            if len(candidates) > 1:

                detected_qty = normalize_compare_qty(
                    detected_item.get("QTY", "")
                )

                if detected_qty is not None:

                    qty_matches = []

                    for idx in candidates:

                        ref_qty = normalize_compare_qty(
                            reference_rows[idx].get("QTY", "")
                        )

                        if (
                            ref_qty is not None
                            and abs(ref_qty - detected_qty) < 1e-6
                        ):

                            qty_matches.append(idx)

                    if qty_matches:

                        return qty_matches[0]

                return candidates[0]

            if len(candidates) == 1:

                return candidates[0]

            # Fuzzy fallback only when exact canonical PART_NO

            # matching did not find a reference row. Used for alignment
            # only: a row aligned this way is still reported with its
            # PART_NO field as a MISMATCH below.

            best_idx = None

            best_score = 0.0

            for idx, ref_item in enumerate(reference_rows):

                if idx in used_reference_indices:

                    continue

                ref_key = part_key(ref_item.get("_MATCH_PART_NO", ""))

                if not detected_key or not ref_key:

                    continue

                score = SequenceMatcher(None, detected_key, ref_key).ratio()

                if score > best_score:

                    best_score = score

                    best_idx = idx

            if best_idx is not None and best_score >= 0.80:

                return best_idx

            return None

        # One-to-one occurrence-aware matching.

        for detected_item in detected_rows:

            detected_part = str(detected_item.get("PART_NO", "")).strip()

            match_idx = find_reference_match(detected_item)

            if match_idx is None:

                comparison_rows.append(
                    {
                        "PART_NO": detected_part,
                        "FIELD": "ROW",
                        "EXPECTED": ("Part should exist in " "reference BOM"),
                        "DETECTED": "Extra part",
                        "STATUS": "MISMATCH",
                    }
                )

                continue

            used_reference_indices.add(match_idx)

            expected_item = reference_rows[match_idx]

            expected_part = str(expected_item.get("PART_NO", "")).strip()

            # PART_NO status is now computed honestly. Rows aligned by
            # canonical PART_NO equality still report MATCH exactly as
            # before; a row that was only aligned by the fuzzy fallback
            # reports MISMATCH instead of being silently forgiven.

            part_no_is_match = part_key(expected_part) == part_key(
                detected_part
            )

            comparison_rows.append(
                {
                    "PART_NO": expected_part,
                    "FIELD": "PART_NO",
                    "EXPECTED": expected_part,
                    "DETECTED": detected_part,
                    "STATUS": ("MATCH" if part_no_is_match else "MISMATCH"),
                }
            )

            for field in fields:

                expected_value = str(expected_item.get(field, "")).strip()

                detected_value = str(detected_item.get(field, "")).strip()

                if field == "QTY":

                    expected_qty = normalize_compare_qty(expected_value)

                    detected_qty = normalize_compare_qty(detected_value)

                    if expected_qty is not None and detected_qty is not None:

                        is_match = abs(expected_qty - detected_qty) < 1e-6

                    else:

                        is_match = normalize_compare_text(
                            expected_value
                        ) == normalize_compare_text(detected_value)

                else:

                    is_match = normalize_compare_text(
                        expected_value
                    ) == normalize_compare_text(detected_value)

                comparison_rows.append(
                    {
                        "PART_NO": expected_part,
                        "FIELD": field,
                        "EXPECTED": expected_value,
                        "DETECTED": detected_value,
                        "STATUS": ("MATCH" if is_match else "MISMATCH"),
                    }
                )

        # Reference rows not used above are missing after

        # one-to-one matching.

        for idx, expected_item in enumerate(reference_rows):

            if idx in used_reference_indices:

                continue

            part_no = str(expected_item.get("PART_NO", "")).strip()

            comparison_rows.append(
                {
                    "PART_NO": part_no,
                    "FIELD": "ROW",
                    "EXPECTED": ("Part exists in " "reference BOM"),
                    "DETECTED": "Missing part",
                    "STATUS": "MISMATCH",
                }
            )

        total_fields = len(comparison_rows)

        matching_fields = sum(
            1 for result in comparison_rows if result["STATUS"] == "MATCH"
        )

        mismatched_fields = sum(
            1 for result in comparison_rows if result["STATUS"] == "MISMATCH"
        )

        comparison = {
            "status": (
                "MATCH" if mismatched_fields == 0 else "MISMATCH DETECTED"
            ),
            "summary": {
                "total_fields_checked": (total_fields),
                "matching_fields": (matching_fields),
                "mismatched_fields": (mismatched_fields),
                "reference_rows_compared": (len(reference_rows)),
                "detected_rows_compared": (len(detected_rows)),
            },
            "results": comparison_rows,
        }

    # 7. Final response

    # ---------------------------------------------------------

    return {
        "filename": blueprint.filename,
        "image": {"width": width, "height": height},
        "bom_crop": {
            "source": crop_source,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": bom_crop.width,
            "height": bom_crop.height,
        },
        "ocr_detections": len(ocr_results),
        "ocr_rows": len(rows),
        "extracted_bom_rows": len(extracted_bom),
        "extracted_bom": extracted_bom,
        "reference_available": (reference is not None),
        "reference_source": reference_source,
        "reference_bom_rows": (len(reference) if reference is not None else 0),
        "comparison_available": (comparison is not None),
        "comparison": comparison,
        }