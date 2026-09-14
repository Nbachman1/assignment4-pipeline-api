"""Fits the search pipeline on the quote corpus and dumps pipeline.joblib.

Run once locally (or any time the corpus/pipeline changes):
    python build_pipeline.py
The resulting artifact is what serve.py / modal_serve.py load — nothing is
ever rebuilt at API boot time.
"""
import datetime
import sys

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from corpus import QUOTES
from pipeline_def import TextCleaner

ARTIFACT_PATH = "pipeline.joblib"


def build() -> dict:
    documents = [q["text"] for q in QUOTES]

    pipeline = Pipeline(
        steps=[
            ("cleaner", TextCleaner(lowercase=True, remove_stopwords=True)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    matrix = pipeline.fit_transform(documents)

    bundle = {
        "pipeline": pipeline,
        "matrix": matrix,
        "documents": QUOTES,
        "metadata": {
            "steps": [name for name, _ in pipeline.steps],
            "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "python_version": sys.version.split()[0],
            "n_documents": len(documents),
            "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
        },
    }
    return bundle


if __name__ == "__main__":
    bundle = build()
    joblib.dump(bundle, ARTIFACT_PATH)
    print(f"Wrote {ARTIFACT_PATH}")
    for k, v in bundle["metadata"].items():
        print(f"  {k}: {v}")
