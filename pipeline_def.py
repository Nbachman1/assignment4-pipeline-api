"""Custom transformer used inside the fitted search pipeline.

Imported by both the build script (to fit the pipeline) and serve.py /
modal_serve.py (to unpickle it). Must live in its own module so joblib can
resolve the class by import path at load time.
"""
import re
import unicodedata

from sklearn.base import BaseEstimator, TransformerMixin

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_PUNCT_RE = re.compile(r"[^\w\s']")
_WS_RE = re.compile(r"\s+")

# Small fixed stopword list (kept local so we don't depend on nltk downloads
# at build/serve time — important since Modal's container has no internet
# access to fetch corpora at runtime).
_STOPWORDS = frozenset(
    """
    a an the and or but if while is are was were be been being to of in on
    for with as by at from this that these those it its it's i you he she
    they we our your their his her not no do does did doing have has had
    """.split()
)


class TextCleaner(BaseEstimator, TransformerMixin):
    """Normalizes raw text before vectorization.

    Lowercases, strips URLs/punctuation/accents, collapses whitespace, and
    optionally drops a small fixed stopword list. Stateless (fit is a no-op)
    but still a real pipeline step, so its exact settings travel with the
    fitted pipeline instead of living as ad-hoc preprocessing in the API.
    """

    def __init__(self, lowercase: bool = True, remove_stopwords: bool = True):
        # __init__ only assigns its arguments (sklearn estimator contract).
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._clean(doc) for doc in X]

    def _clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = _URL_RE.sub(" ", text)
        if self.lowercase:
            text = text.lower()
        text = _PUNCT_RE.sub(" ", text)
        text = _WS_RE.sub(" ", text).strip()
        if self.remove_stopwords:
            words = [w for w in text.split(" ") if w and w not in _STOPWORDS]
            text = " ".join(words)
        return text
