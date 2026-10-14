"""Best-effort MLflow logging using the run identity supplied by the environment."""

from contextlib import contextmanager
import sys


@contextmanager
def tracking_run(params):
    enabled = False
    warned = False
    run_context = None

    def disable(error):
        nonlocal enabled, warned
        enabled = False
        if not warned:
            print(f"Warning: MLflow tracking unavailable: {error}", flush=True)
            warned = True

    try:
        import mlflow

        run_context = mlflow.start_run()
        run_context.__enter__()
        enabled = True
        mlflow.log_params(params)
    except Exception as error:
        disable(error)

    def log_metric(name, value, step=None):
        if enabled:
            try:
                mlflow.log_metric(name, value, step=step)
            except Exception as error:
                disable(error)

    try:
        yield log_metric
    finally:
        if run_context is not None:
            try:
                run_context.__exit__(*sys.exc_info())
            except Exception as error:
                disable(error)
