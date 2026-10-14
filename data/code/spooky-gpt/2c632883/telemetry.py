"""Best-effort tracking using the run identity supplied by the environment."""

from contextlib import contextmanager


@contextmanager
def tracking():
    backend = None
    run = None
    enabled = False
    warned = False

    def disable(error):
        nonlocal enabled, warned
        enabled = False
        if not warned:
            print(f"Warning: MLflow tracking unavailable: {error}", flush=True)
            warned = True

    try:
        import mlflow
        backend = mlflow
        run = backend.start_run()
        enabled = True
    except Exception as error:
        disable(error)

    def log(method, *args, **kwargs):
        if enabled:
            try:
                getattr(backend, method)(*args, **kwargs)
            except Exception as error:
                disable(error)

    try:
        yield log
    finally:
        if run is not None:
            try:
                backend.end_run()
            except Exception as error:
                disable(error)
