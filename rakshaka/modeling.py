import random
import re
import unicodedata
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


def preprocess_text(text):
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = re.sub(r"\s+", " ", text).strip()
    markers = []
    if re.search(r"(?:https?://|www\.)\S+", text):
        markers.append(" urltoken ")
    if re.search(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", text):
        markers.append(" emailtoken ")
    if re.search(r"\+?\d[\d\s().-]{6,}\d", text):
        markers.append(" phonetoken ")
    return text + "".join(markers)


def split_by_group(rows, test_size=0.15, validation_size=0.15, seed=42):
    labels_by_group = {}
    for row in rows:
        group = row["group_id"]
        label = row["label"]
        if group in labels_by_group and labels_by_group[group] != label:
            raise ValueError("Group {} has conflicting labels".format(group))
        labels_by_group[group] = label

    groups = sorted(labels_by_group)
    labels = [labels_by_group[group] for group in groups]
    class_counts = defaultdict(int)
    for label in labels:
        class_counts[label] += 1
    if len(class_counts) < 2 or min(class_counts.values()) < 3:
        raise ValueError("Need at least three message groups in each label")

    train_val_groups, test_groups = train_test_split(
        groups,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    train_val_labels = [labels_by_group[group] for group in train_val_groups]
    val_fraction = validation_size / (1.0 - test_size)
    train_groups, validation_groups = train_test_split(
        train_val_groups,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_val_labels,
    )
    return {
        "train": set(train_groups),
        "validation": set(validation_groups),
        "test": set(test_groups),
    }


def balance_training_groups(rows, seed=42):
    groups_by_label = defaultdict(set)
    for row in rows:
        groups_by_label[row["label"]].add(row["group_id"])
    if len(groups_by_label) < 2 or not groups_by_label:
        raise ValueError("Training data must contain at least two labels")
    target_size = min(len(groups) for groups in groups_by_label.values())
    randomizer = random.Random(seed)
    selected_groups = set()
    for groups in groups_by_label.values():
        selected_groups.update(randomizer.sample(sorted(groups), target_size))
    return [row for row in rows if row["group_id"] in selected_groups]


def make_classifier():
    features = FeatureUnion(
        [
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=250000,
                ),
            ),
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=100000,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            ("classifier", LogisticRegression(max_iter=1500, random_state=42)),
        ]
    )


def evaluate_classifier(classifier, rows):
    texts = [preprocess_text(row["text"]) for row in rows]
    actual = [row["label"] for row in rows]
    predicted = classifier.predict(texts)
    return {
        "rows": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
        "confusion_matrix_labels": ["legitimate", "scam"],
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=["legitimate", "scam"]
        ).tolist(),
        "per_class": classification_report(
            actual,
            predicted,
            labels=["legitimate", "scam"],
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_by_language(classifier, rows):
    languages = sorted({row["language"] for row in rows})
    results = {}
    for language in languages:
        language_rows = [row for row in rows if row["language"] == language]
        if {row["label"] for row in language_rows} != {"legitimate", "scam"}:
            results[language] = {
                "evaluable": False,
                "rows": len(language_rows),
                "groups": len({row["group_id"] for row in language_rows}),
                "reason": "requires both legitimate and scam examples",
            }
            continue
        results[language] = evaluate_classifier(classifier, language_rows)
        results[language]["evaluable"] = True
    return results


def char_ngram_oov_ratio(classifier, text):
    features = classifier.named_steps["features"]
    char_vectorizer = dict(features.transformer_list)["char"]
    analyzer = char_vectorizer.build_analyzer()
    ngrams = analyzer(preprocess_text(text))
    if not ngrams:
        return 1.0
    vocabulary = char_vectorizer.vocabulary_
    return sum(ngram not in vocabulary for ngram in ngrams) / len(ngrams)
