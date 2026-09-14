# Assignment 4 — Submission Details

**Modal API URL:** https://nbachman1--cs-quotes-search-api-fastapi-app.modal.run
**API /docs URL:** https://nbachman1--cs-quotes-search-api-fastapi-app.modal.run/docs
**Vercel URL:** _(fill in after `vercel --prod`)_

## Description (3–5 sentences)

This API serves a fitted scikit-learn `Pipeline` that does semantic search over a
corpus of 63 well-known programming/CS quotes. The pipeline has two steps: a
custom `TextCleaner` transformer (in `pipeline_def.py`) that normalizes and
strips stopwords from text, followed by a `TfidfVectorizer` fitted on the
corpus. A query is run through the same fitted pipeline at request time and
ranked against the pre-computed document matrix by cosine similarity, so the
vocabulary and IDF weights are genuine learned state — a freshly-rebuilt,
unfitted pipeline would return wrong (empty) results. Custom transformer:
`TextCleaner`. scikit-learn version: `1.9.1`.

## Postman

Collection: `postman_collection.json` (base_url variable points at the Modal
URL). Verified with `newman run postman_collection.json` — 5 requests, 11/11
assertions passing (health 200, info 200, valid search 200, two invalid
searches 422). Screenshots of these requests run in the Postman app still
need to be captured and attached for the Canvas submission.

## Repo contents

- `pipeline_def.py` — custom `TextCleaner` transformer
- `corpus.py` — the 63-quote corpus (build-time only, not shipped to Modal)
- `build_pipeline.py` — fits and dumps `pipeline.joblib`
- `pipeline.joblib` — the fitted artifact (pipeline + matrix + documents + metadata)
- `serve.py` — FastAPI app (`/health`, `/info`, `/search`)
- `modal_serve.py` — Modal deployment wrapper (`modal deploy modal_serve.py`)
- `requirements.txt` — local dev dependencies
- `postman_collection.json` — Postman collection with assertions
- `frontend/index.html` — static frontend calling the live Modal API
