"""Best-effort MLflow logging using the evaluator-provided run identity."""

from contextlib import contextmanager
import sys


@contextmanager
def tracked_run():
    enabled = False
    warned = False
    active_run = None
    mlflow = None

    def disable(exc):
        nonlocal enabled, warned
        enabled = False
        if not warned:
            print(f"Warning: MLflow tracking unavailable: {exc}", flush=True)
            warned = True

    try:
        import mlflow
        active_run = mlflow.start_run()
        active_run.__enter__()
        enabled = True
    except Exception as exc:
        disable(exc)

    def log(method, *args, **kwargs):
        if enabled:
            try:
                getattr(mlflow, method)(*args, **kwargs)
            except Exception as exc:
                disable(exc)

    try:
        yield log
    finally:
        if active_run is not None:
            try:
                active_run.__exit__(*sys.exc_info())
            except Exception as exc:
                disable(exc)
