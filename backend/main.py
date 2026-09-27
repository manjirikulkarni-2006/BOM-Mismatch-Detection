import io
import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PPStructureV3


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="BOM Mismatch Detection API",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

GROUND_TRUTH_PATH = (
    BASE_DIR
    / "data"
    / "ground_truth"
    / "blueprint_ground_truth.csv"
)

LEGACY_MODEL_PATH = (
    BASE_DIR
    / "model"
    / "layoutlmv3_bom_corrected_best.pth"
)


# ============================================================
# PP-STRUCTURE
# ============================================================

print("=" * 70)
print("INITIALIZING PP-STRUCTUREV3")
print("=" * 70)

pipeline = PPStructureV3(lang="en")

print("PP-StructureV3 initialized successfully.")
print()


# ============================================================
# STANDARD BOM SCHEMA
# ============================================================

BOM_FIELDS = [
    "PART_NO",
    "DESCRIPTION",
    "QTY",
    "UOM",
    "MATERIAL",
]


# ============================================================
# FIELD ALIASES
# ============================================================

FIELD_ALIASES = {
    "PART_NO": {
        "partno",
        "part_no",
        "partnumber",
        "part_number",
        "part",
        "itemno",
        "item_no",
        "itemnumber",
        "item_number",
        "item",
        "componentno",
        "component_no",
        "componentnumber",
        "component_number",
        "stockno",
        "stock_no",
        "stocknumber",
        "stock_number",
    },
    "DESCRIPTION": {
        "description",
        "desc",
        "itemdescription",
        "item_description",
        "partdescription",
        "part_description",
        "componentdescription",
        "component_description",
        "name",
        "itemname",
        "item_name",
    },
    "QTY": {
        "qty",
        "quantity",
        "count",
        "amount",
        "requiredqty",
        "required_qty",
    },
    "UOM": {
        "uom",
        "unit",
        "units",
        "unitofmeasure",
        "unit_of_measure",
        "measure",
    },
    "MATERIAL": {
        "material",
        "matl",
        "mat",
        "materialtype",
        "material_type",
        "materialgrade",
        "material_grade",
        "grade",
        "specification",
        "spec",
    },
}


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_string(value: Any) -> str:
    """
    Convert any value into a clean string.
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()

    if text.lower() in {
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
    }:
        return ""

    return text


def normalize_text(value: Any) -> str:
    """
    General text normalization.
    """
    text = safe_string(value)

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_header(value: Any) -> str:
    """
    Normalize a table header so aliases can be recognized.

    Example:
        "PART NO." -> "partno"
        "Part Number" -> "partnumber"
        "MATERIAL / GRADE" -> "materialgrade"
    """
    text = normalize_text(value).lower()

    text = text.replace("&", "and")

    # Remove common punctuation.
    text = re.sub(r"[^a-z0-9]+", "", text)

    return text


def canonical_field_from_header(header: Any) -> Optional[str]:
    """
    Convert an arbitrary table header into one of our
    standard BOM fields.
    """
    normalized = normalize_header(header)

    if not normalized:
        return None

    for field, aliases in FIELD_ALIASES.items():
        if normalized in aliases:
            return field

    # More flexible matching for unusual headers.
    if "part" in normalized and (
        "no" in normalized
        or "number" in normalized
        or normalized == "part"
    ):
        return "PART_NO"

    if "item" in normalized and (
        "no" in normalized
        or "number" in normalized
    ):
        return "PART_NO"

    if "desc" in normalized:
        return "DESCRIPTION"

    if "quantity" in normalized or normalized == "qty":
        return "QTY"

    if "unit" in normalized and (
        "measure" in normalized
        or normalized in {"unit", "units"}
    ):
        return "UOM"

    if normalized in {"uom", "unit"}:
        return "UOM"

    if "material" in normalized:
        return "MATERIAL"

    if normalized in {"matl", "mat"}:
        return "MATERIAL"

    if "grade" in normalized:
        return "MATERIAL"

    return None


# ============================================================
# DATA VALIDATION
# ============================================================

def is_numeric(value: Any) -> bool:
    """
    Check whether a value represents a numeric quantity.
    """
    text = safe_string(value)

    if not text:
        return False

    text = text.replace(",", "")
    text = text.replace(" ", "")

    # Allow values such as:
    # 10
    # 10.5
    # 10.5kg
    # 10 EA
    match = re.fullmatch(
        r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?:[a-zA-Z%]+)?",
        text,
    )

    return match is not None


def is_uom(value: Any) -> bool:
    """
    Detect common engineering BOM units.
    """
    text = normalize_text(value).upper()

    if not text:
        return False

    common_units = {
        "EA",
        "EACH",
        "PCS",
        "PC",
        "UNIT",
        "UNITS",
        "LF",
        "FT",
        "M",
        "MM",
        "CM",
        "IN",
        "KG",
        "G",
        "LB",
        "LBS",
        "TON",
        "T",
        "GAL",
        "L",
        "LOT",
        "SET",
        "SETS",
        "BOX",
        "BAG",
        "ROLL",
        "PAIR",
        "PR",
    }

    if text in common_units:
        return True

    # Short alphabetic unit-like strings.
    if 1 <= len(text) <= 5 and re.fullmatch(r"[A-Z0-9/.-]+", text):
        return True

    return False


def looks_like_part_number(value: Any) -> bool:
    text = normalize_text(value)

    if not text:
        return False

    if len(text) > 80:
        return False

    # A part number normally contains alphanumeric content.
    if not re.search(r"[A-Za-z0-9]", text):
        return False

    # Reject obvious long sentences.
    if len(text.split()) > 8:
        return False

    return True


def looks_like_description(value: Any) -> bool:
    text = normalize_text(value)

    if not text:
        return False

    if len(text) < 2 or len(text) > 250:
        return False

    return True


def looks_like_material(value: Any) -> bool:
    text = normalize_text(value)

    if not text:
        return False

    if len(text) > 150:
        return False

    # Typical engineering material/specification patterns.
    material_patterns = [
        r"\bASTM\b",
        r"\bA\d{2,4}\b",
        r"\bSS\b",
        r"\bSTAINLESS\b",
        r"\bSTEEL\b",
        r"\bALUMINUM\b",
        r"\bALUMINIUM\b",
        r"\bPVC\b",
        r"\bHDPE\b",
        r"\bCPVC\b",
        r"\bBRASS\b",
        r"\bCOPPER\b",
        r"\bIRON\b",
        r"\bGR\b",
        r"\bGRADE\b",
        r"\bSCH\b",
        r"\bGALV\b",
    ]

    upper = text.upper()

    if any(re.search(pattern, upper) for pattern in material_patterns):
        return True

    # Some datasets may simply contain short material names.
    if len(text.split()) <= 6:
        return True

    return False


def looks_like_bom_row(row: List[Any]) -> bool:
    """
    A row is considered a BOM row if the important fields
    contain plausible values.

    Material is NOT mandatory because real BOM tables may
    legitimately leave it blank.
    """
    if len(row) < 5:
        return False

    values = [normalize_text(x) for x in row]

    part_no = values[0]
    description = values[1]
    qty = values[2]
    uom = values[3]
    material = values[4]

    core_valid = (
        looks_like_part_number(part_no)
        and looks_like_description(description)
        and is_numeric(qty)
        and is_uom(uom)
    )

    if not core_valid:
        return False

    # Material is optional.
    if material:
        return looks_like_material(material)

    return True

# ============================================================
# TABLE EXTRACTION
# ============================================================

def html_to_variants(html: str) -> List[List[List[str]]]:
    """
    Convert PP-StructureV3 HTML into row matrices.

    We intentionally parse the original HTML directly instead
    of using pandas.read_html().

    This preserves the cell structure produced by
    PP-StructureV3 more faithfully.
    """
    if not html:
        return []

    try:
        soup = BeautifulSoup(
            html,
            "html.parser"
        )
    except Exception:
        return []

    variants = []

    # PP-StructureV3 normally gives one table, but support
    # multiple HTML tables just in case.
    html_tables = soup.find_all("table")

    if not html_tables:
        html_tables = [soup]

    for html_table in html_tables:

        rows = []

        for tr in html_table.find_all("tr"):

            cells = tr.find_all(
                ["th", "td"]
            )

            values = [
                normalize_text(
                    cell.get_text(
                        " ",
                        strip=True
                    )
                )
                for cell in cells
            ]

            if any(values):
                rows.append(values)

        if rows:
            variants.append(rows)

    return variants


def get_table_variants(
    table_result: Dict[str, Any]
) -> List[List[List[str]]]:
    """
    Return all row matrices contained in one
    PP-StructureV3 table result.
    """
    html = table_result.get(
        "pred_html",
        ""
    )

    return html_to_variants(html)


def get_table_rows(
    table_result: Dict[str, Any]
) -> List[List[str]]:
    """
    Backward-compatible helper.
    """
    variants = get_table_variants(
        table_result
    )

    if not variants:
        return []

    return max(
        variants,
        key=lambda rows: len(rows)
    )


# ============================================================
# HEADER DETECTION
# ============================================================

def analyze_header(
    row: List[Any]
) -> Dict[str, Any]:
    """
    Analyze a possible BOM header row.
    """
    mapping = {}
    recognized_fields = []

    for index, value in enumerate(row):

        field = canonical_field_from_header(
            value
        )

        if field:

            recognized_fields.append(
                field
            )

            # Keep the first occurrence.
            if field not in mapping:
                mapping[field] = index

    unique_count = len(
        set(recognized_fields)
    )

    required_core = {
        "PART_NO",
        "DESCRIPTION",
    }

    has_core = required_core.issubset(
        set(recognized_fields)
    )

    has_quantity_or_uom = (
        "QTY" in recognized_fields
        or "UOM" in recognized_fields
    )

    strong = (
        has_core
        and has_quantity_or_uom
        and unique_count >= 3
    )

    duplicate_count = (
        len(recognized_fields)
        - unique_count
    )

    blank_count = sum(
        1
        for value in row
        if not normalize_text(value)
    )

    score = (
        unique_count * 2
        + (3 if has_core else 0)
        + (2 if has_quantity_or_uom else 0)
        - duplicate_count * 1.5
        - blank_count * 0.15
    )

    return {
        "mapping": mapping,
        "recognized_fields": recognized_fields,
        "unique_count": unique_count,
        "duplicate_count": duplicate_count,
        "blank_count": blank_count,
        "strong": strong,
        "score": score,
    }


def find_header(
    rows: List[List[Any]]
) -> Dict[str, Any]:
    """
    Search the first part of the table for a BOM header.
    """
    best = {
        "index": -1,
        "header": [],
        "mapping": {},
        "recognized_fields": [],
        "unique_count": 0,
        "duplicate_count": 0,
        "blank_count": 0,
        "strong": False,
        "score": 0,
    }

    max_scan = min(
        len(rows),
        15
    )

    for index in range(max_scan):

        analysis = analyze_header(
            rows[index]
        )

        if analysis["score"] > best["score"]:

            best = {
                "index": index,
                "header": rows[index],
                **analysis,
            }

    return best


# ============================================================
# HEADER-BASED MAPPING
# ============================================================

def get_header_mapping(
    header_info: Dict[str, Any]
) -> Dict[str, int]:

    return dict(
        header_info.get(
            "mapping",
            {}
        )
    )


# ============================================================
# DATA-DRIVEN COLUMN ALIGNMENT
# ============================================================

def field_value_score(
    field: str,
    value: Any
) -> float:
    """
    Score one cell according to the expected BOM field.

    This is used to recover from table HTML where header
    cells and data cells have slightly different positions
    because of merged/blank cells.
    """
    text = normalize_text(value)

    if field == "PART_NO":
        return (
            1.0
            if looks_like_part_number(text)
            else 0.0
        )

    if field == "DESCRIPTION":
        return (
            1.0
            if looks_like_description(text)
            else 0.0
        )

    if field == "QTY":
        return (
            1.0
            if is_numeric(text)
            else 0.0
        )

    if field == "UOM":
        return (
            1.0
            if is_uom(text)
            else 0.0
        )

    if field == "MATERIAL":

        if not text:
            # Blank material is valid.
            return 0.50

        return (
            1.0
            if looks_like_material(text)
            else 0.0
        )

    return 0.0


def mapping_row_score(
    row: List[Any],
    mapping: Dict[str, int]
) -> float:
    """
    Score one row against one mapping.
    """
    try:
        values = {
            field: normalize_text(
                row[mapping[field]]
            )
            for field in BOM_FIELDS
        }

    except (IndexError, KeyError):
        return 0.0

    score = 0.0

    score += field_value_score(
        "PART_NO",
        values["PART_NO"]
    ) * 1.0

    score += field_value_score(
        "DESCRIPTION",
        values["DESCRIPTION"]
    ) * 1.0

    score += field_value_score(
        "QTY",
        values["QTY"]
    ) * 1.5

    score += field_value_score(
        "UOM",
        values["UOM"]
    ) * 1.5

    score += field_value_score(
        "MATERIAL",
        values["MATERIAL"]
    ) * 0.5

    return score / 5.5


def score_mapping(
    rows: List[List[Any]],
    header_index: int,
    mapping: Dict[str, int]
) -> Tuple[float, int]:
    """
    Score a mapping over actual data rows.
    """
    data_rows = rows[
        header_index + 1:
    ]

    if not data_rows:
        return 0.0, 0

    sample_rows = data_rows[:100]

    scores = []

    for row in sample_rows:

        score = mapping_row_score(
            row,
            mapping
        )

        if score >= 0.60:
            scores.append(score)

    if not scores:
        return 0.0, 0

    return (
        sum(scores) / len(scores),
        len(scores)
    )


def mapping_valid_row_count(
    rows: List[List[Any]],
    header_index: int,
    mapping: Dict[str, int]
) -> int:
    """
    Count how many actual BOM rows are valid under
    the supplied mapping.

    Duplicate rows are intentionally counted separately.
    """
    count = 0

    for row in rows[
        header_index + 1:
    ]:

        if not row:
            continue

        try:
            values = [
                normalize_text(
                    row[mapping[field]]
                )
                for field in BOM_FIELDS
            ]

        except (IndexError, KeyError):
            continue

        if looks_like_bom_row(values):
            count += 1

    return count


def generate_nearby_mappings(
    rows: List[List[Any]],
    header_index: int,
    header_mapping: Dict[str, int]
) -> List[Dict[str, int]]:
    """
    Generate a small number of generalized mappings around
    the detected header positions.

    PP-StructureV3 can produce blank/merged header cells,
    so the actual data column can be slightly left or right
    of the header position.

    We search only a small neighborhood rather than thousands
    of arbitrary combinations.
    """
    if not header_mapping:
        return []

    max_columns = max(
        len(row)
        for row in rows
    )

    # Small shifts are enough for the type of HTML alignment
    # issue seen in the noisy blueprint tables.
    shifts = [-2, -1, 0, 1, 2]

    candidates = []

    # --------------------------------------------------------
    # Build possible positions for every recognized field.
    # --------------------------------------------------------

    field_positions = {}

    for field in BOM_FIELDS:

        if field not in header_mapping:
            continue

        header_position = (
            header_mapping[field]
        )

        positions = []

        for shift in shifts:

            position = (
                header_position
                + shift
            )

            if (
                0 <= position
                < max_columns
            ):
                positions.append(
                    position
                )

        # Remove duplicates.
        field_positions[field] = list(
            dict.fromkeys(
                positions
            )
        )

    # We need all five BOM fields.
    if any(
        field not in field_positions
        for field in BOM_FIELDS
    ):
        return []

    # --------------------------------------------------------
    # Generate combinations.
    #
    # Only five fields are involved and each field has at
    # most five nearby positions.
    # --------------------------------------------------------

    candidates = []

    def build(
        field_index: int,
        current: Dict[str, int],
        used: set
    ):
        if field_index == len(
            BOM_FIELDS
        ):
            candidates.append(
                dict(current)
            )
            return

        field = BOM_FIELDS[
            field_index
        ]

        for position in field_positions[
            field
        ]:

            if position in used:
                continue

            current[field] = position

            used.add(position)

            build(
                field_index + 1,
                current,
                used
            )

            used.remove(position)

            current.pop(
                field,
                None
            )

    build(
        0,
        {},
        set()
    )

    # --------------------------------------------------------
    # Remove duplicate mappings.
    # --------------------------------------------------------

    unique = []

    seen = set()

    for mapping in candidates:

        key = tuple(
            mapping[field]
            for field in BOM_FIELDS
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(mapping)

    return unique


def choose_mapping(
    rows: List[List[Any]],
    header_info: Dict[str, Any]
):
    """
    Choose the best generalized BOM mapping.

    Strategy:

    1. Start from the detected BOM header.
    2. Test the exact header positions.
    3. Test only nearby positions.
    4. Evaluate mappings using actual BOM row patterns.
    5. Prefer mappings that recover the greatest number
       of valid rows.
    6. Use mapping quality as the tie breaker.

    This is generalized and does not depend on any specific
    blueprint or part number.
    """

    header_index = header_info[
        "index"
    ]

    header_mapping = (
        get_header_mapping(
            header_info
        )
    )

    if not header_mapping:
        return (
            None,
            "",
            0.0,
            0
        )

    candidates = []

    # Exact header mapping.
    candidates.append(
        (
            header_mapping,
            "HEADER"
        )
    )

    # Nearby data-driven mappings.
    nearby_mappings = (
        generate_nearby_mappings(
            rows,
            header_index,
            header_mapping
        )
    )

    for mapping in nearby_mappings:

        key = tuple(
            mapping[field]
            for field in BOM_FIELDS
        )

        existing_keys = {
            tuple(
                existing_mapping[field]
                for field in BOM_FIELDS
            )
            for existing_mapping, _
            in candidates
        }

        if key not in existing_keys:

            candidates.append(
                (
                    mapping,
                    "DATA_ALIGNED"
                )
            )

    # --------------------------------------------------------
    # Evaluate mappings.
    # --------------------------------------------------------

    best_mapping = None
    best_source = ""
    best_score = float("-inf")
    best_valid_rows = -1

    for mapping, source in candidates:

        valid_rows = (
            mapping_valid_row_count(
                rows,
                header_index,
                mapping
            )
        )

        mapping_score, scored_rows = (
            score_mapping(
                rows,
                header_index,
                mapping
            )
        )

        # Prefer mappings that recover more actual rows.
        #
        # Mapping score only breaks ties.
        final_score = (
            valid_rows * 10.0
            + mapping_score
        )

        # Exact header gets a very small preference only
        # when the actual row count and quality are equal.
        if source == "HEADER":
            final_score += 0.001

        if (
            valid_rows > best_valid_rows
            or (
                valid_rows == best_valid_rows
                and final_score > best_score
            )
        ):

            best_mapping = mapping
            best_source = source
            best_score = final_score
            best_valid_rows = valid_rows

    return (
        best_mapping,
        best_source,
        round(
            best_score,
            4
        ),
        best_valid_rows,
    )


# ============================================================
# ROW EXTRACTION
# ============================================================

def extract_rows_from_mapping(
    rows: List[List[Any]],
    header_index: int,
    mapping: Dict[str, int]
) -> List[Dict[str, str]]:
    """
    Extract BOM rows using the selected mapping.

    IMPORTANT:
    No deduplication is performed.

    If the same part number appears three times, all three
    rows remain separate.
    """
    result = []

    if not mapping:
        return result

    for row in rows[
        header_index + 1:
    ]:

        if not row:
            continue

        try:

            bom_row = {
                field: normalize_text(
                    row[mapping[field]]
                )
                for field in BOM_FIELDS
            }

        except (IndexError, KeyError):
            continue

        # Ignore repeated header rows.
        if (
            canonical_field_from_header(
                bom_row["PART_NO"]
            ) == "PART_NO"
            and canonical_field_from_header(
                bom_row["DESCRIPTION"]
            ) == "DESCRIPTION"
        ):
            continue

        # Ignore completely empty rows.
        if (
            not bom_row["PART_NO"]
            and not bom_row["DESCRIPTION"]
        ):
            continue

        values = [
            bom_row["PART_NO"],
            bom_row["DESCRIPTION"],
            bom_row["QTY"],
            bom_row["UOM"],
            bom_row["MATERIAL"],
        ]

        if looks_like_bom_row(values):

            # Do NOT deduplicate.
            result.append(
                bom_row
            )

    return result


# ============================================================
# TABLE SELECTION
# ============================================================

def select_bom_table(
    table_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Search every PP-StructureV3 table and select the table
    that contains the strongest BOM structure.

    Selection is based on:
      - BOM header quality
      - number of valid BOM rows
      - mapping quality

    Duplicate BOM rows are preserved.
    """
    candidates = []

    for table_index, table_result in enumerate(
        table_results
    ):

        variants = get_table_variants(
            table_result
        )

        for variant_index, rows in enumerate(
            variants
        ):

            if not rows:
                continue

            header_info = find_header(
                rows
            )

            if header_info["index"] < 0:
                continue

            # Require at least three recognizable BOM fields.
            if (
                header_info[
                    "unique_count"
                ] < 3
            ):
                continue

            (
                mapping,
                mapping_source,
                mapping_score,
                valid_rows,
            ) = choose_mapping(
                rows,
                header_info
            )

            if not mapping:
                continue

            bom_rows = (
                extract_rows_from_mapping(
                    rows,
                    header_info["index"],
                    mapping
                )
            )

            if not bom_rows:
                continue

            # ------------------------------------------------
            # Table selection score.
            # ------------------------------------------------

            selection_score = 0.0

            # Header quality.
            selection_score += (
                header_info[
                    "unique_count"
                ] * 20
            )

            if header_info["strong"]:
                selection_score += 30

            # Actual extracted rows are very important.
            selection_score += (
                min(
                    len(bom_rows),
                    100
                ) * 10
            )

            # Mapping quality.
            selection_score += (
                mapping_score
            )

            # Small preference for exact header alignment.
            if mapping_source == "HEADER":
                selection_score += 2

            candidate = {
                "table_index": table_index,
                "variant_index": variant_index,
                "rows": rows,
                "header": header_info[
                    "header"
                ],
                "header_index": header_info[
                    "index"
                ],
                "mapping": mapping,
                "mapping_source": mapping_source,
                "mapping_score": round(
                    mapping_score,
                    4
                ),
                "valid_rows": len(
                    bom_rows
                ),
                "bom": bom_rows,
                "selection_score": round(
                    selection_score,
                    2
                ),
                "header_info": header_info,
            }

            candidates.append(
                candidate
            )

    if not candidates:
        return {
            "selected": None,
            "candidates": [],
        }

    candidates.sort(
        key=lambda x: (
            x["valid_rows"],
            x["selection_score"],
        ),
        reverse=True
    )

    selected = candidates[0]

    print()
    print("=" * 70)
    print("SELECTED BOM TABLE")
    print("=" * 70)

    print(
        f"Table index: "
        f"{selected['table_index']}"
    )

    print(
        f"Variant index: "
        f"{selected['variant_index']}"
    )

    print(
        f"Header: "
        f"{selected['header']}"
    )

    print(
        f"Mapping: "
        f"{selected['mapping']}"
    )

    print(
        f"Mapping source: "
        f"{selected['mapping_source']}"
    )

    print(
        f"Mapping score: "
        f"{selected['mapping_score']}"
    )

    print(
        f"Valid BOM rows: "
        f"{selected['valid_rows']}"
    )

    print(
        f"Selection score: "
        f"{selected['selection_score']}"
    )

    print("=" * 70)
    print()

    return {
        "selected": selected,
        "candidates": candidates,
    }




# ============================================================
# PP-STRUCTURE BOM EXTRACTION
# ============================================================

def extract_bom_with_ppstructure(
    image_path: str
) -> Dict[str, Any]:
    """
    Run PP-StructureV3 and generalized BOM extraction.
    """
    print()
    print("=" * 70)
    print("PP-STRUCTUREV3 BOM EXTRACTION")
    print("=" * 70)
    print(f"Image: {image_path}")
    print()

    try:
        output = pipeline.predict(
            image_path
        )
    except Exception as exc:
        raise RuntimeError(
            f"PP-StructureV3 failed: {exc}"
        ) from exc

    # Convert generator/list result into a list.
    if not isinstance(output, list):
        output = list(output)

    table_results = []

    for result in output:
        if not isinstance(result, dict):
            continue

        tables = result.get(
            "table_res_list",
            []
        )

        if tables:
            table_results.extend(
                tables
            )

    if not table_results:
        raise RuntimeError(
            "No tables were detected by PP-StructureV3."
        )

    selection = select_bom_table(
        table_results
    )

    selected = selection["selected"]

    if selected is None:
        raise RuntimeError(
            "PP-StructureV3 detected tables, "
            "but no valid BOM table could be identified."
        )

    return {
        "bom": selected["bom"],
        "table_index": selected["table_index"],
        "variant_index": selected["variant_index"],
        "header": selected["header"],
        "mapping": selected["mapping"],
        "mapping_source": selected["mapping_source"],
        "mapping_score": selected["mapping_score"],
        "selection_score": selected["selection_score"],
        "valid_rows": selected["valid_rows"],
        "candidate_count": len(
            selection["candidates"]
        ),
        "raw_output": output,
    }


# ============================================================
# BLUEPRINT ID
# ============================================================

def extract_blueprint_id(filename: str) -> Optional[str]:
    """
    Extract the six-digit blueprint ID.

    Example:
        blueprint_300447_noisy.jpg
        -> 300447
    """
    name = Path(filename).name

    match = re.search(
        r"(?<!\d)(\d{6})(?!\d)",
        name
    )

    if match:
        return match.group(1)

    return None


# ============================================================
# GROUND TRUTH PARSING
# ============================================================

def standardize_bom_item(
    item: Dict[str, Any]
) -> Dict[str, str]:
    """
    Convert a ground-truth BOM dictionary into
    the application's standard schema.
    """
    normalized_keys = {
        normalize_header(key): key
        for key in item.keys()
    }

    def get_value(
        aliases: List[str]
    ) -> Any:
        for alias in aliases:
            key = normalized_keys.get(
                normalize_header(alias)
            )

            if key is not None:
                return item[key]

        return ""

    return {
        "PART_NO": normalize_text(
            get_value([
                "part_no",
                "part number",
                "part_no.",
                "part",
                "item_no",
                "item number",
            ])
        ),
        "DESCRIPTION": normalize_text(
            get_value([
                "description",
                "desc",
                "item description",
            ])
        ),
        "QTY": normalize_text(
            get_value([
                "qty",
                "quantity",
            ])
        ),
        "UOM": normalize_text(
            get_value([
                "uom",
                "unit",
                "unit of measure",
            ])
        ),
        "MATERIAL": normalize_text(
            get_value([
                "material",
                "matl",
                "material grade",
                "grade",
            ])
        ),
    }


def load_reference_bom(
    blueprint_id: Optional[str]
) -> List[Dict[str, str]]:
    """
    Load the exact BOM for a blueprint from the 500-image
    ground-truth CSV.

    The dataset stores BOM data under:
        json_data -> bill_of_materials
    """
    if not blueprint_id:
        return []

    if not GROUND_TRUTH_PATH.exists():
        print(
            f"Ground truth file not found: "
            f"{GROUND_TRUTH_PATH}"
        )
        return []

    try:
        df = pd.read_csv(
            GROUND_TRUTH_PATH
        )
    except Exception as exc:
        print(
            f"Failed to read ground truth CSV: {exc}"
        )
        return []

    for _, record in df.iterrows():
        filename = safe_string(
            record.get("filename", "")
        )

        record_id = extract_blueprint_id(
            filename
        )

        # Exact blueprint ID matching.
        if record_id != blueprint_id:
            continue

        json_data = record.get(
            "json_data",
            ""
        )

        try:
            data = json.loads(
                json_data
            )
        except Exception as exc:
            print(
                f"Could not parse json_data for "
                f"{filename}: {exc}"
            )
            return []

        bom_items = data.get(
            "bill_of_materials",
            []
        )

        if not isinstance(
            bom_items,
            list
        ):
            return []

        reference = []

        for item in bom_items:
            if not isinstance(
                item,
                dict
            ):
                continue

            standardized = standardize_bom_item(
                item
            )

            if any(
                standardized.values()
            ):
                reference.append(
                    standardized
                )

        print(
            f"Reference BOM loaded for "
            f"{blueprint_id}: "
            f"{len(reference)} rows"
        )

        return reference

    print(
        f"No ground truth found for blueprint "
        f"{blueprint_id}"
    )

    return []


# ============================================================
# OCR / TEXT CORRECTION
# ============================================================

def canonical_compare_text(
    value: Any
) -> str:
    """
    Normalize values for comparison.
    """
    text = normalize_text(value).upper()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def similarity(
    a: Any,
    b: Any
) -> float:
    return SequenceMatcher(
        None,
        canonical_compare_text(a),
        canonical_compare_text(b)
    ).ratio()

def normalize_description_for_comparison(
    value: Any
) -> str:
    """
    Normalize engineering descriptions for comparison.

    This handles common OCR formatting differences such as:
    - missing spaces
    - quotation marks
    - apostrophes

    It does not remove meaningful separators such as
    '/', '-', or '_'.
    """
    text = canonical_compare_text(value)

    # Remove quote and apostrophe characters that are
    # commonly lost during OCR.
    text = text.replace('"', "")
    text = text.replace("'", "")

    # Ignore spacing differences.
    text = re.sub(r"\s+", "", text)

    return text

def build_ground_truth_vocab(
    exclude_blueprint_id: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Build optional vocabulary from the ground truth dataset.

    This is used only as a correction aid.
    It is NOT required for the core table extraction.
    """
    vocab = {
        "PART_NO": set(),
        "DESCRIPTION": set(),
        "MATERIAL": set(),
    }

    if not GROUND_TRUTH_PATH.exists():
        return {
            key: []
            for key in vocab
        }

    try:
        df = pd.read_csv(
            GROUND_TRUTH_PATH
        )
    except Exception:
        return {
            key: []
            for key in vocab
        }

    for _, record in df.iterrows():
        filename = safe_string(
            record.get("filename", "")
        )

        record_id = extract_blueprint_id(
            filename
        )

        # Avoid using the current blueprint as its own
        # correction source.
        if (
            exclude_blueprint_id
            and record_id == exclude_blueprint_id
        ):
            continue

        try:
            data = json.loads(
                record.get(
                    "json_data",
                    ""
                )
            )
        except Exception:
            continue

        items = data.get(
            "bill_of_materials",
            []
        )

        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue

            standardized = standardize_bom_item(
                item
            )

            for field in vocab:
                value = standardized[field]

                if value:
                    vocab[field].add(value)

    return {
        key: list(values)
        for key, values in vocab.items()
    }


def correct_with_vocab(
    value: str,
    vocabulary: List[str],
    threshold: float = 0.88
) -> str:
    """
    Correct obvious OCR errors only when a close
    ground-truth vocabulary match exists.
    """
    value = normalize_text(value)

    if not value:
        return value

    if not vocabulary:
        return value

    best_value = value
    best_score = 0.0

    for candidate in vocabulary:
        score = similarity(
            value,
            candidate
        )

        if score > best_score:
            best_score = score
            best_value = candidate

    if best_score >= threshold:
        return best_value

    return value


def correct_bom_ocr(
    bom: List[Dict[str, str]],
    blueprint_id: Optional[str]
) -> List[Dict[str, str]]:
    """
    Apply conservative vocabulary-based correction.
    """
    vocab = build_ground_truth_vocab(
        blueprint_id
    )

    corrected = []

    for row in bom:
        new_row = dict(row)

        for field in [
            "PART_NO",
            "DESCRIPTION",
            "MATERIAL",
        ]:
            new_row[field] = correct_with_vocab(
                new_row[field],
                vocab.get(field, [])
            )

        corrected.append(
            new_row
        )

    return corrected


# ============================================================
# BOM COMPARISON
# ============================================================

def values_match(
    field: str,
    expected: Any,
    detected: Any
) -> bool:
    """
    Field-aware comparison.
    """
    expected_text = canonical_compare_text(
        expected
    )
    detected_text = canonical_compare_text(
        detected
    )

    if field == "DESCRIPTION":
        expected_description = (
            normalize_description_for_comparison(
                expected
            )
        ) 

        detected_description = (
            normalize_description_for_comparison(
                detected
            )
        )

    # First check exact match after OCR-format
    # normalization, then allow a small residual
    # OCR difference.
        return (
            expected_description == detected_description
            or similarity(
                expected_description,
                detected_description
            ) >= 0.90
        )


    if field == "MATERIAL":
        # Keep material comparison unchanged.
        return similarity(
            expected_text,
            detected_text
        ) >= 0.90
    
    if field == "QTY":
        try:
            expected_number = float(
                expected_text.replace(",", "")
            )
            detected_number = float(
                detected_text.replace(",", "")
            )

            return abs(
                expected_number
                - detected_number
            ) < 1e-6
        except Exception:
            return expected_text == detected_text

    return expected_text == detected_text


def part_numbers_match(
    expected: str,
    detected: str
) -> bool:
    """
    Slightly stricter part-number matching.
    """
    expected_text = canonical_compare_text(
        expected
    )
    detected_text = canonical_compare_text(
        detected
    )

    if expected_text == detected_text:
        return True

    return similarity(
        expected_text,
        detected_text
    ) >= 0.94


def compare_bom_rows(
    reference_bom: List[Dict[str, str]],
    detected_bom: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Occurrence-aware BOM comparison.

    Important:
    Duplicate part numbers are allowed.

    Example:
        W10x33
        W10x33
        W10x33

    are treated as three separate BOM rows.
    """
    unmatched_detected = list(
        range(len(detected_bom))
    )

    matched_pairs = []
    missing_rows = []
    extra_rows = []

    # --------------------------------------------------------
    # MATCH REFERENCE ROWS TO DETECTED ROWS
    # --------------------------------------------------------

    for ref_index, reference in enumerate(
        reference_bom
    ):
        best_detected_index = None
        best_score = -1.0

        for detected_index in unmatched_detected:
            detected = detected_bom[
                detected_index
            ]

            # Part number is the primary key.
            if part_numbers_match(
                reference["PART_NO"],
                detected["PART_NO"]
            ):
                score = 1.0

                for field in BOM_FIELDS:
                    if values_match(
                        field,
                        reference[field],
                        detected[field]
                    ):
                        score += 0.2

                if score > best_score:
                    best_score = score
                    best_detected_index = (
                        detected_index
                    )

        # If no part number match, use a secondary
        # fuzzy row comparison.
        if best_detected_index is None:
            for detected_index in unmatched_detected:
                detected = detected_bom[
                    detected_index
                ]

                score = 0.0

                for field in BOM_FIELDS:
                    if field == "PART_NO":
                        score += (
                            similarity(
                                reference[field],
                                detected[field]
                            ) * 2
                        )
                    else:
                        score += similarity(
                            reference[field],
                            detected[field]
                        )

                score /= (
                    len(BOM_FIELDS) + 1
                )

                if score > best_score:
                    best_score = score
                    best_detected_index = (
                        detected_index
                    )

        # Only accept fuzzy matching if sufficiently strong.
        if (
            best_detected_index is not None
            and best_score >= 1.25
        ):
            unmatched_detected.remove(
                best_detected_index
            )

            matched_pairs.append({
                "reference": reference,
                "detected": detected_bom[
                    best_detected_index
                ],
                "reference_index": ref_index,
                "detected_index": best_detected_index,
            })

        else:
            missing_rows.append({
                "reference": reference,
                "reference_index": ref_index,
            })

    # Whatever remains was not found in reference.
    for detected_index in unmatched_detected:
        extra_rows.append({
            "detected": detected_bom[
                detected_index
            ],
            "detected_index": detected_index,
        })

    return {
        "matched_pairs": matched_pairs,
        "missing_rows": missing_rows,
        "extra_rows": extra_rows,
    }


# ============================================================
# FRONTEND COMPARISON FORMAT
# ============================================================

def build_frontend_comparison(
    reference_bom: List[Dict[str, str]],
    detected_bom: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Convert the internal row-level comparison into the
    field-level structure expected by App.js.
    """
    internal = compare_bom_rows(
        reference_bom,
        detected_bom
    )

    results = []

    matching_fields = 0
    mismatched_fields = 0
    missing_fields = 0
    extra_fields = 0

    # --------------------------------------------------------
    # MATCHED ROWS
    # --------------------------------------------------------

    for pair in internal["matched_pairs"]:
        reference = pair["reference"]
        detected = pair["detected"]

        for field in BOM_FIELDS:
            expected = reference.get(
                field,
                ""
            )

            actual = detected.get(
                field,
                ""
            )

            if field == "PART_NO":
                match = part_numbers_match(
                    expected,
                    actual
                )
            else:
                match = values_match(
                    field,
                    expected,
                    actual
                )

            status = (
                "MATCH"
                if match
                else "MISMATCH"
            )

            if match:
                matching_fields += 1
            else:
                mismatched_fields += 1

            results.append({
                "PART_NO": reference.get(
                    "PART_NO",
                    ""
                ),
                "FIELD": field,
                "EXPECTED": expected,
                "DETECTED": actual,
                "STATUS": status,
            })

    # --------------------------------------------------------
    # MISSING ROWS
    # --------------------------------------------------------

    for item in internal["missing_rows"]:
        reference = item["reference"]

        for field in BOM_FIELDS:
            expected = reference.get(
                field,
                ""
            )

            if not expected:
                continue

            missing_fields += 1

            results.append({
                "PART_NO": reference.get(
                    "PART_NO",
                    ""
                ),
                "FIELD": field,
                "EXPECTED": expected,
                "DETECTED": "",
                "STATUS": "MISSING",
            })

    # --------------------------------------------------------
    # EXTRA ROWS
    # --------------------------------------------------------

    for item in internal["extra_rows"]:
        detected = item["detected"]

        for field in BOM_FIELDS:
            actual = detected.get(
                field,
                ""
            )

            if not actual:
                continue

            extra_fields += 1

            results.append({
                "PART_NO": detected.get(
                    "PART_NO",
                    ""
                ),
                "FIELD": field,
                "EXPECTED": "",
                "DETECTED": actual,
                "STATUS": "EXTRA",
            })

    total_fields_checked = (
        matching_fields
        + mismatched_fields
        + missing_fields
        + extra_fields
    )

    missing_rows = len(
        internal["missing_rows"]
    )

    extra_rows = len(
        internal["extra_rows"]
    )

    matched_rows = len(
        internal["matched_pairs"]
    )

    # A row mismatch includes any field mismatch.
    mismatched_rows = 0

    for pair in internal["matched_pairs"]:
        reference = pair["reference"]
        detected = pair["detected"]

        row_has_mismatch = False

        for field in BOM_FIELDS:
            if field == "PART_NO":
                match = part_numbers_match(
                    reference[field],
                    detected[field]
                )
            else:
                match = values_match(
                    field,
                    reference[field],
                    detected[field]
                )

            if not match:
                row_has_mismatch = True
                break

        if row_has_mismatch:
            mismatched_rows += 1

    return {
        "results": results,
        "summary": {
            "total_fields_checked": total_fields_checked,
            "matching_fields": matching_fields,
            "mismatched_fields": mismatched_fields,
            "missing_fields": missing_fields,
            "extra_fields": extra_fields,
            "matched_rows": matched_rows,
            "mismatched_rows": mismatched_rows,
            "missing_rows": missing_rows,
            "extra_rows": extra_rows,
        },
        # Keep internal information available for debugging.
        "internal": internal,
    }


# ============================================================
# LEGACY / MODEL STATUS
# ============================================================

@app.get("/")
def root():
    return {
        "message": "BOM Mismatch Detection API",
        "version": "3.0",
        "status": "running",
        "extraction_engine": "PP-StructureV3",
        "ground_truth": str(
            GROUND_TRUTH_PATH
        ),
    }


@app.get("/model-status")
def model_status():
    return {
        "pp_structure_v3": True,
        "legacy_layoutlm_model": (
            LEGACY_MODEL_PATH.exists()
        ),
        "legacy_model_path": str(
            LEGACY_MODEL_PATH
        ),
        "ground_truth_available": (
            GROUND_TRUTH_PATH.exists()
        ),
    }


# ============================================================
# UPLOAD BLUEPRINT
# ============================================================

@app.post("/upload-blueprint")
async def upload_blueprint(
    file: UploadFile = File(...)
):
    """
    Extract BOM only.
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename supplied."
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use JPG, JPEG, PNG, BMP, WEBP, "
                "TIF, or TIFF."
            )
        )

    temp_dir = BASE_DIR / "output"

    temp_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_path = (
        temp_dir
        / f"_analysis_{file.filename}"
    )

    try:
        content = await file.read()

        with open(
            temp_path,
            "wb"
        ) as f:
            f.write(content)

        extraction = (
            extract_bom_with_ppstructure(
                str(temp_path)
            )
        )

        blueprint_id = (
            extract_blueprint_id(
                file.filename
            )
        )

        corrected_bom = correct_bom_ocr(
            extraction["bom"],
            blueprint_id
        )

        return {
            "success": True,
            "filename": file.filename,
            "blueprint_id": blueprint_id,
            "extraction_method": (
                "PP-StructureV3 + generalized parser"
            ),
            "bom": corrected_bom,
            "extracted_bom": corrected_bom,
            "bom_row_count": len(
                corrected_bom
            ),
            "extracted_bom_rows": len(
                corrected_bom
            ),
            "table_index": extraction[
                "table_index"
            ],
            "variant_index": extraction[
                "variant_index"
            ],
            "header": extraction[
                "header"
            ],
            "mapping": extraction[
                "mapping"
            ],
            "mapping_source": extraction[
                "mapping_source"
            ],
            "mapping_score": extraction[
                "mapping_score"
            ],
            "selection_score": extraction[
                "selection_score"
            ],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


# ============================================================
# CHECK BOM
# ============================================================

@app.post("/check-bom")
async def check_bom(
    payload: Dict[str, Any]
):
    """
    Compare two BOM arrays supplied directly as JSON.
    """
    reference_bom = payload.get(
        "reference_bom",
        []
    )

    detected_bom = payload.get(
        "detected_bom",
        []
    )

    if not isinstance(
        reference_bom,
        list
    ):
        raise HTTPException(
            status_code=400,
            detail="reference_bom must be a list."
        )

    if not isinstance(
        detected_bom,
        list
    ):
        raise HTTPException(
            status_code=400,
            detail="detected_bom must be a list."
        )

    comparison = build_frontend_comparison(
        reference_bom,
        detected_bom
    )

    summary = comparison["summary"]

    return {
        "success": True,
        "comparison": comparison,
        "summary": {
            "matches": summary[
                "matching_fields"
            ],
            "mismatches": summary[
                "mismatched_fields"
            ],
            "missing": summary[
                "missing_rows"
            ],
            "extra": summary[
                "extra_rows"
            ],
        },
    }


# ============================================================
# ANALYZE BLUEPRINT
# ============================================================

@app.post("/analyze-blueprint")
async def analyze_blueprint(
    file: UploadFile = File(...),
    reference_bom_file: Optional[
        UploadFile
    ] = File(None),
):
    """
    Complete pipeline:

        Blueprint image
              ↓
        PP-StructureV3
              ↓
        Generalized table parser
              ↓
        BOM extraction
              ↓
        Optional OCR correction
              ↓
        Blueprint ID
              ↓
        Ground-truth lookup
              ↓
        Field-level comparison
              ↓
        Frontend-ready JSON
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename supplied."
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format."
            )
        )

    temp_dir = BASE_DIR / "output"

    temp_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_path = (
        temp_dir
        / f"_analysis_{file.filename}"
    )

    try:
        # ----------------------------------------------------
        # SAVE IMAGE
        # ----------------------------------------------------

        content = await file.read()

        with open(
            temp_path,
            "wb"
        ) as f:
            f.write(content)

        # ----------------------------------------------------
        # 1. EXTRACT BOM
        # ----------------------------------------------------

        extraction = (
            extract_bom_with_ppstructure(
                str(temp_path)
            )
        )

        raw_bom = extraction[
            "bom"
        ]

        # ----------------------------------------------------
        # 2. BLUEPRINT ID
        # ----------------------------------------------------

        blueprint_id = (
            extract_blueprint_id(
                file.filename
            )
        )

        # ----------------------------------------------------
        # 3. OCR CORRECTION
        # ----------------------------------------------------

        corrected_bom = correct_bom_ocr(
            raw_bom,
            blueprint_id
        )

        # ----------------------------------------------------
        # 4. LOAD REFERENCE BOM
        # ----------------------------------------------------

        reference_bom = []

        # Optional uploaded reference JSON/CSV can be added
        # later. For the current project, the 500-image
        # ground-truth CSV is the default reference.
        if (
            reference_bom_file is not None
            and reference_bom_file.filename
        ):
            reference_bom = (
                await parse_uploaded_reference(
                    reference_bom_file,
                    blueprint_filename=file.filename,
                )
            )

        if not reference_bom:
            reference_bom = load_reference_bom(
                blueprint_id
            )

        # ----------------------------------------------------
        # 5. COMPARE
        # ----------------------------------------------------

        comparison = build_frontend_comparison(
            reference_bom,
            corrected_bom
        )

        comparison_summary = (
            comparison["summary"]
        )

        # ----------------------------------------------------
        # 6. LEGACY SUMMARY
        # ----------------------------------------------------

        legacy_summary = {
            "detected_rows": len(
                corrected_bom
            ),
            "reference_rows": len(
                reference_bom
            ),
            "matches": comparison_summary[
                "matching_fields"
            ],
            "mismatches": comparison_summary[
                "mismatched_fields"
            ],
            "missing": comparison_summary[
                "missing_rows"
            ],
            "extra": comparison_summary[
                "extra_rows"
            ],
        }

        # ----------------------------------------------------
        # 7. FRONTEND RESPONSE
        # ----------------------------------------------------

        response = {
            "success": True,

            "filename": file.filename,

            "blueprint_id": blueprint_id,

            "extraction_method": (
                "PP-StructureV3 + generalized parser"
            ),

            # New frontend contract.
            "extracted_bom": corrected_bom,

            "extracted_bom_rows": len(
                corrected_bom
            ),

            "reference_bom": reference_bom,

            "reference_bom_rows": len(
                reference_bom
            ),

            # Current extraction does not expose individual
            # OCR boxes because the application uses
            # PP-StructureV3 table extraction.
            "ocr_detections": None,

            "ocr_rows": len(
                corrected_bom
            ),

            "comparison": comparison,

            "summary": legacy_summary,

            "table_information": {
                "table_index": extraction[
                    "table_index"
                ],
                "variant_index": extraction[
                    "variant_index"
                ],
                "header": extraction[
                    "header"
                ],
                "mapping": extraction[
                    "mapping"
                ],
                "mapping_source": extraction[
                    "mapping_source"
                ],
                "mapping_score": extraction[
                    "mapping_score"
                ],
                "selection_score": extraction[
                    "selection_score"
                ],
                "candidate_count": extraction[
                    "candidate_count"
                ],
            },

            # Keep old key too, in case another part of
            # the application still expects it.
            "bom": corrected_bom,
            "bom_row_count": len(
                corrected_bom
            ),
        }

        return response

    except HTTPException:
        raise

    except Exception as exc:
        print()
        print("=" * 70)
        print("ANALYSIS ERROR")
        print("=" * 70)
        print(str(exc))
        print("=" * 70)
        print()

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


# ============================================================
# OPTIONAL REFERENCE FILE SUPPORT
# ============================================================

async def parse_uploaded_reference(
    uploaded_file: UploadFile,
    blueprint_filename: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Parse a user-provided reference file.

    Supports:
      1. Simple BOM CSV:
         PART_NO, DESCRIPTION, QTY, UOM, MATERIAL

      2. Dataset ground-truth CSV:
         filename, document_type, json_data

      3. JSON BOM files.
    """

    filename = uploaded_file.filename or ""

    extension = Path(filename).suffix.lower()

    content = await uploaded_file.read()

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if extension == ".csv":
        try:
            df = pd.read_csv(
                io.BytesIO(content)
            )

            # ------------------------------------------------
            # FORMAT 1:
            # Simple BOM CSV
            # ------------------------------------------------

            required_columns = {
                "PART_NO",
                "DESCRIPTION",
                "QTY",
                "UOM",
                "MATERIAL",
            }

            if required_columns.issubset(
                set(df.columns)
            ):
                rows = []

                for _, row in df.iterrows():
                    item = {
                        field: normalize_text(
                            row.get(field, "")
                        )
                        for field in BOM_FIELDS
                    }

                    if any(item.values()):
                        rows.append(item)

                return rows

            # ------------------------------------------------
            # FORMAT 2:
            # Dataset ground-truth CSV
            #
            # filename | document_type | json_data
            # ------------------------------------------------

            if {
                "filename",
                "json_data",
            }.issubset(set(df.columns)):

                if not blueprint_filename:
                    return []

                target_filename = (
                    Path(blueprint_filename)
                    .name
                    .strip()
                    .lower()
                )

                for _, record in df.iterrows():

                    record_filename = (
                        safe_string(
                            record.get(
                                "filename",
                                ""
                            )
                        )
                        .strip()
                        .lower()
                    )

                    if (
                        record_filename
                        != target_filename
                    ):
                        continue

                    json_data = safe_string(
                        record.get(
                            "json_data",
                            ""
                        )
                    )

                    try:
                        data = json.loads(
                            json_data
                        )
                    except Exception:
                        return []

                    bom_items = data.get(
                        "bill_of_materials",
                        []
                    )

                    if not isinstance(
                        bom_items,
                        list
                    ):
                        return []

                    rows = []

                    for item in bom_items:

                        if not isinstance(
                            item,
                            dict
                        ):
                            continue

                        standardized = (
                            standardize_bom_item(
                                item
                            )
                        )

                        if any(
                            standardized.values()
                        ):
                            rows.append(
                                standardized
                            )

                    return rows

                # Blueprint was not found
                return []

            return []

        except Exception:
            return []

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    if extension == ".json":
        try:
            data = json.loads(
                content.decode("utf-8")
            )

            if isinstance(data, dict):
                data = data.get(
                    "bill_of_materials",
                    data.get(
                        "bom",
                        data.get(
                            "items",
                            []
                        )
                    )
                )

            if not isinstance(
                data,
                list
            ):
                return []

            rows = []

            for item in data:

                if not isinstance(
                    item,
                    dict
                ):
                    continue

                rows.append(
                    standardize_bom_item(
                        item
                    )
                )

            return rows

        except Exception:
            return []

    return []


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )