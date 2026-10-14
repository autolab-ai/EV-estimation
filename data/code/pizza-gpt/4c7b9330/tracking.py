"""Best-effort telemetry for the evaluator-provided MLflow run."""

from contextlib import ExitStack, contextmanager


@contextmanager
def tracking_run(params):
    enabled = True
    stack = ExitStack()

    def disable(exc):
        nonlocal enabled
        if enabled:
            print(f"Warning: MLflow tracking unavailable: {exc}", flush=True)
        enabled = False

    try:
        import mlflow

        stack.enter_context(mlflow.start_run())
        mlflow.log_params(params)
    except Exception as exc:
        disable(exc)

    def log_metric(name, value, step=0):
        if enabled:
            try:
                mlflow.log_metric(name, value, step=step)
            except Exception as exc:
                disable(exc)

    try:
        yield log_metric
    finally:
        try:
            stack.close()
        except Exception as exc:
            disable(exc)
