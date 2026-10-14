"""Optional tracking: use the evaluator's identity and never interrupt training."""

import os
from contextlib import contextmanager


@contextmanager
def tracking_run():
    enabled = False
    warned = False
    context = None

    def warn(exc):
        nonlocal enabled, warned
        enabled = False
        if not warned:
            print(f"MLflow tracking unavailable; continuing untracked: {exc}", flush=True)
            warned = True

    def log(method, *args, **kwargs):
        if enabled:
            try:
                getattr(mlflow, method)(*args, synchronous=False, **kwargs)
            except Exception as exc:
                warn(exc)

    if os.environ.get("MLFLOW_TRACKING_URI"):
        try:
            import mlflow

            context = mlflow.start_run()
            context.__enter__()
            enabled = True
        except Exception as exc:
            context = None
            warn(exc)
    try:
        yield log
    finally:
        if context is not None:
            try:
                context.__exit__(None, None, None)
            except Exception as exc:
                warn(exc)
