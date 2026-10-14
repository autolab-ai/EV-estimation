"""Best-effort MLflow logging using the evaluator's preconfigured run."""

from contextlib import contextmanager
import sys


@contextmanager
def tracking():
    enabled = False
    warned = False
    run = None

    def warn(exc):
        nonlocal warned
        if not warned:
            print(f"Warning: MLflow tracking unavailable: {exc}", file=sys.stderr)
            warned = True

    try:
        import mlflow

        run = mlflow.start_run()
        enabled = True
    except Exception as exc:
        warn(exc)

    def log(method, *args, **kwargs):
        nonlocal enabled
        if enabled:
            try:
                getattr(mlflow, method)(*args, **kwargs)
            except Exception as exc:
                enabled = False
                warn(exc)

    status = "FINISHED"
    try:
        yield log
    except BaseException:
        status = "FAILED"
        raise
    finally:
        if run is not None:
            try:
                mlflow.end_run(status=status)
            except Exception as exc:
                warn(exc)
