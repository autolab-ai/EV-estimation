"""Best-effort MLflow logging using the evaluator-provided run identity."""

from contextlib import contextmanager
import sys


@contextmanager
def tracking():
    backend = None
    run = None
    warned = False

    def disable(error):
        nonlocal backend, warned
        backend = None
        if not warned:
            print(f"Warning: MLflow tracking unavailable: {error}", file=sys.stderr)
            warned = True

    try:
        import mlflow

        run = mlflow.start_run()
        run.__enter__()
        backend = mlflow
    except Exception as error:
        disable(error)

    def log(method, *args, **kwargs):
        if backend is not None:
            try:
                getattr(backend, method)(*args, **kwargs)
            except Exception as error:
                disable(error)

    try:
        yield log
    finally:
        if run is not None:
            try:
                run.__exit__(*sys.exc_info())
            except Exception as error:
                disable(error)
