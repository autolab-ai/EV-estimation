"""Best-effort telemetry using the run identity supplied by the environment."""

from contextlib import ExitStack, contextmanager


@contextmanager
def tracking_run(params):
    stack = ExitStack()
    enabled = True
    warned = False

    def disable(exc):
        nonlocal enabled, warned
        enabled = False
        if not warned:
            print(f"Warning: tracking unavailable ({exc}); continuing without telemetry.", flush=True)
            warned = True

    try:
        import mlflow
        stack.enter_context(mlflow.start_run())
        mlflow.log_params(params)
    except Exception as exc:
        disable(exc)

    def log_metric(name, value, step=0):
        if enabled:
            try:
                mlflow.log_metric(name, float(value), step=step)
            except Exception as exc:
                disable(exc)

    try:
        yield log_metric
    finally:
        try:
            stack.close()
        except Exception as exc:
            disable(exc)
