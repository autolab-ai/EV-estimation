"""Best-effort logging to the MLflow run configured by the environment."""

from contextlib import contextmanager
import sys


@contextmanager
def tracking_run():
    enabled = True
    context = None

    def disable(error):
        nonlocal enabled
        if enabled:
            print(f"Warning: MLflow tracking disabled: {error}", flush=True)
        enabled = False

    try:
        import mlflow

        candidate = mlflow.start_run()
        candidate.__enter__()
        context = candidate
    except Exception as error:
        disable(error)

    def log(method, *args, **kwargs):
        if enabled:
            try:
                getattr(mlflow, method)(*args, **kwargs)
            except Exception as error:
                disable(error)

    try:
        yield log
    finally:
        if context is not None:
            try:
                context.__exit__(*sys.exc_info())
            except Exception as error:
                disable(error)
