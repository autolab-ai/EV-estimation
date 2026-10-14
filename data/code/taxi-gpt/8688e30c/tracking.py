"""Best-effort telemetry using the run identity supplied by the environment."""

from contextlib import ExitStack, contextmanager


@contextmanager
def tracked_run():
    disabled = False
    stack = ExitStack()

    def disable(exc):
        nonlocal disabled
        if not disabled:
            print(f"Warning: MLflow tracking unavailable; continuing untracked ({exc})", flush=True)
        disabled = True

    try:
        import mlflow

        stack.enter_context(mlflow.start_run())
    except Exception as exc:
        disable(exc)

    def log(method, *args, **kwargs):
        if disabled:
            return
        try:
            getattr(mlflow, method)(*args, **kwargs)
        except Exception as exc:
            disable(exc)

    try:
        yield log
    finally:
        try:
            stack.close()
        except Exception as exc:
            disable(exc)
