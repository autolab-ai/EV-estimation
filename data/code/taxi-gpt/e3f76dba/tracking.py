"""Best-effort telemetry using the evaluator-provided MLflow run identity."""

from contextlib import ExitStack, contextmanager


@contextmanager
def tracked_run():
    stack = ExitStack()
    client = None
    warned = False

    def disable(exc):
        nonlocal client, warned
        client = None
        if not warned:
            print(f"Warning: MLflow tracking unavailable; continuing untracked: {exc}", flush=True)
            warned = True

    try:
        import mlflow

        stack.enter_context(mlflow.start_run())
        client = mlflow
    except Exception as exc:
        disable(exc)

    def log(method, *args, **kwargs):
        if client is not None:
            try:
                getattr(client, method)(*args, **kwargs)
            except Exception as exc:
                disable(exc)

    try:
        yield log
    finally:
        try:
            stack.close()
        except Exception as exc:
            disable(exc)
