import csv
import hashlib
import io
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path


LANGUAGES = ("sinhala", "singlish", "tamil", "tanglish")
ANNOTATION_FIELDS = [
    "group_id",
    "label",
    "english_text",
    "sinhala_text",
    "sinhala_status",
    "sinhala_reviewer",
    "singlish_text",
    "singlish_status",
    "singlish_reviewer",
    "tamil_text",
    "tamil_status",
    "tamil_reviewer",
    "tanglish_text",
    "tanglish_status",
    "tanglish_reviewer",
    "review_notes",
]
DRIFT_FIELDS = [
    "group_id",
    "text",
    "label",
    "language",
    "pattern",
    "review_status",
    "reviewer",
]
MESSAGE_FIELDS = [
    "group_id",
    "label",
    "original_labels",
    "source_refs",
    "text",
    "url_present",
    "email_present",
    "phone_present",
]


def clean_text(text):
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def label_category(label):
    normalized = (label or "").strip().casefold()
    if normalized == "ham":
        return "legitimate"
    if normalized in {"spam", "smishing"}:
        return "scam"
    raise ValueError("Unrecognized or empty source label: {!r}".format(label))


def group_id_for(text):
    normalized = clean_text(text).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _find_column(header, names):
    positions = {name.strip().casefold(): index for index, name in enumerate(header)}
    for name in names:
        if name in positions:
            return positions[name]
    raise ValueError(
        "Required column {!r} was not found in CSV header: {}".format(
            names[0], header
        )
    )


def _load_csv(path, source_name):
    records = []
    raw = Path(path).read_bytes()
    fallback_count = 0
    try:
        source_text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        source_text = raw.decode("utf-8-sig", errors="surrogateescape")
        decoded = []
        undecodable_count = 0
        for character in source_text:
            codepoint = ord(character)
            if 0xDC80 <= codepoint <= 0xDCFF:
                fallback_count += 1
                try:
                    decoded.append(bytes([codepoint - 0xDC00]).decode("cp1252"))
                except UnicodeDecodeError:
                    decoded.append("\ufffd")
                    undecodable_count += 1
            else:
                decoded.append(character)
        source_text = "".join(decoded)
        print(
            "Warning: {} contains {} non-UTF-8 byte(s); decoded those bytes as Windows-1252.".format(
                path, fallback_count
            ),
            file=sys.stderr,
        )
        if undecodable_count:
            print(
                "Warning: {} contains {} byte(s) undefined by Windows-1252; retained as replacement characters.".format(
                    path, undecodable_count
                ),
                file=sys.stderr,
            )
    replacement_count = source_text.count("\ufffd")
    if replacement_count:
        print(
            "Warning: {} contains {} literal replacement character(s); their original text cannot be recovered.".format(
                path, replacement_count
            ),
            file=sys.stderr,
        )

    with io.StringIO(source_text, newline="") as csv_file:
        reader = csv.reader(csv_file)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("Dataset is empty: {}".format(path))

        if source_name == "spam.csv":
            label_index = _find_column(header, ("v1",))
            text_index = _find_column(header, ("v2",))
        else:
            label_index = _find_column(header, ("label",))
            text_index = _find_column(header, ("text",))

        for row_number, row in enumerate(reader, start=2):
            if not row or not any(cell.strip() for cell in row):
                continue
            if max(label_index, text_index) >= len(row):
                raise ValueError(
                    "Malformed row {} in {}: expected label and text columns".format(
                        row_number, path
                    )
                )
            text = clean_text(row[text_index])
            if not text:
                continue
            label = row[label_index].strip()
            records.append(
                {
                    "group_id": group_id_for(text),
                    "label": label_category(label),
                    "original_label": label,
                    "source_ref": "{}:{}".format(source_name, row_number),
                    "text": text,
                    "url_present": bool(
                        re.search(r"(?:https?://|www\.)\S+", text, re.IGNORECASE)
                    ),
                    "email_present": bool(
                        re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text)
                    ),
                    "phone_present": bool(re.search(r"\+?\d[\d\s().-]{6,}\d", text)),
                }
            )
    return records


def load_and_clean_sources(source_paths):
    grouped = {}
    duplicate_count = 0
    for path in source_paths:
        path = Path(path)
        source_records = _load_csv(path, path.name)
        for record in source_records:
            key = clean_text(record["text"]).casefold()
            existing = grouped.get(key)
            if existing is None:
                grouped[key] = {
                    "group_id": record["group_id"],
                    "label": record["label"],
                    "original_labels": {record["original_label"]},
                    "source_refs": {record["source_ref"]},
                    "text": record["text"],
                    "url_present": record["url_present"],
                    "email_present": record["email_present"],
                    "phone_present": record["phone_present"],
                }
                continue
            if existing["label"] != record["label"]:
                raise ValueError(
                    "Conflicting labels for duplicate message group {}: {} vs {}".format(
                        record["group_id"], existing["label"], record["label"]
                    )
                )
            duplicate_count += 1
            existing["original_labels"].add(record["original_label"])
            existing["source_refs"].add(record["source_ref"])
            for feature in ("url_present", "email_present", "phone_present"):
                existing[feature] = existing[feature] or record[feature]

    messages = []
    for item in grouped.values():
        messages.append(
            {
                "group_id": item["group_id"],
                "label": item["label"],
                "original_labels": "|".join(sorted(item["original_labels"])),
                "source_refs": "|".join(sorted(item["source_refs"])),
                "text": item["text"],
                "url_present": str(item["url_present"]).lower(),
                "email_present": str(item["email_present"]).lower(),
                "phone_present": str(item["phone_present"]).lower(),
            }
        )
    messages.sort(key=lambda item: item["group_id"])
    return messages, duplicate_count


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_translation_sample(messages, per_label=100, seed=42):
    randomizer = random.Random(seed)
    by_label = defaultdict(list)
    for message in messages:
        by_label[message["label"]].append(message)

    sample = []
    for label in ("legitimate", "scam"):
        candidates = list(by_label[label])
        randomizer.shuffle(candidates)
        for message in candidates[:per_label]:
            sample.append(
                {
                    "group_id": message["group_id"],
                    "label": label,
                    "english_text": message["text"],
                    "sinhala_text": "",
                    "sinhala_status": "pending",
                    "sinhala_reviewer": "",
                    "singlish_text": "",
                    "singlish_status": "pending",
                    "singlish_reviewer": "",
                    "tamil_text": "",
                    "tamil_status": "pending",
                    "tamil_reviewer": "",
                    "tanglish_text": "",
                    "tanglish_status": "pending",
                    "tanglish_reviewer": "",
                    "review_notes": "",
                }
            )
    sample.sort(key=lambda item: (item["label"], item["group_id"]))
    return sample


def load_verified_translations(path, messages):
    known = {message["group_id"]: message for message in messages}
    translations = []
    if not Path(path).exists():
        return translations

    with Path(path).open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        # Legacy annotation CSVs can be upgraded without losing reviewed work.
        legacy_optional = {f"{lang}_{suffix}" for lang in ("tamil", "tanglish") for suffix in ("text", "status", "reviewer")}
        missing = set(ANNOTATION_FIELDS) - set(reader.fieldnames or ()) - legacy_optional
        if missing:
            raise ValueError(
                "Translation file is missing columns: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        for row_number, row in enumerate(reader, start=2):
            group_id = (row.get("group_id") or "").strip()
            if group_id not in known:
                raise ValueError(
                    "Unknown translation group_id at row {}: {}".format(
                        row_number, group_id
                    )
                )
            if row.get("label") != known[group_id]["label"]:
                raise ValueError(
                    "Translation label mismatch at row {} for {}".format(
                        row_number, group_id
                    )
                )
            for language in LANGUAGES:
                text = clean_text(row.get(language + "_text"))
                status = (row.get(language + "_status") or "").strip().casefold()
                reviewer = (row.get(language + "_reviewer") or "").strip()
                if status != "verified":
                    continue
                if not text or not reviewer:
                    raise ValueError(
                        "Verified {} translation at row {} requires text and reviewer".format(
                            language, row_number
                        )
                    )
                translations.append(
                    {
                        "group_id": group_id,
                        "label": known[group_id]["label"],
                        "language": language,
                        "text": text,
                    }
                )

    unique = set()
    for translation in translations:
        key = (translation["group_id"], translation["language"])
        if key in unique:
            raise ValueError(
                "Duplicate verified {} translation for group {}".format(*key)
            )
        unique.add(key)
    return translations


def build_training_rows(messages, translations):
    rows = [
        {
            "group_id": message["group_id"],
            "label": message["label"],
            "language": "english",
            "text": message["text"],
        }
        for message in messages
    ]
    rows.extend(translations)
    return rows
