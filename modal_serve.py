"""Modal deployment for the CS Quotes Search API.

Ships exactly three files into the image: serve.py, pipeline_def.py, and
the fitted pipeline.joblib artifact. scikit-learn is pinned to the exact
version recorded in the artifact's metadata so unpickling is safe.

Deploy:
    modal deploy modal_serve.py
"""
import modal

app = modal.App("cs-quotes-search-api")

# Keep this pinned to the sklearn_version printed by build_pipeline.py /
# GET /info. A mismatch here is a common source of silent unpickling bugs.
SKLEARN_VERSION = "1.9.1"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        f"scikit-learn=={SKLEARN_VERSION}",
        "fastapi",
        "joblib",
        "numpy",
        "scipy",
    )
    .add_local_file("pipeline_def.py", "/root/pipeline_def.py")
    .add_local_file("pipeline.joblib", "/root/pipeline.joblib")
    .add_local_file("serve.py", "/root/serve.py")
)


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    import sys

    sys.path.insert(0, "/root")

    # Imported inside the function so it only needs to succeed inside the
    # deployed container, not at local `modal deploy` time.
    from serve import app as web_app

    return web_app
