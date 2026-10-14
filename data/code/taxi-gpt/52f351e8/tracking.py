"""Best-effort MLflow logging using the run identity supplied by the environment."""

from contextlib import contextmanager
import sys


@contextmanager
def tracking_run():
    manager = None
    enabled = False
    warned = False

    def disable(exc):
        nonlocal enabled, warned
        enabled = False
        if not warned:
            print(f"Warning: MLflow tracking unavailable ({exc}); training continues.", flush=True)
            warned = True

    try:
        import mlflow

        manager = mlflow.start_run()
        manager.__enter__()
        enabled = True
    except Exception as exc:
        manager = None
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
        if manager is not None:
            try:
                manager.__exit__(*sys.exc_info())
            except Exception as exc:
                disable(exc)
