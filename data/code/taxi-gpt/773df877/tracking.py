"""Best-effort telemetry using the run identity supplied by the environment."""

from contextlib import contextmanager


class Tracking:
    def __init__(self):
        self.client = None
        self.warned = False

    def disable(self, exc):
        self.client = None
        if not self.warned:
            print(f"Warning: MLflow tracking disabled: {exc}", flush=True)
            self.warned = True

    def log(self, method, *args, **kwargs):
        if self.client is not None:
            try:
                getattr(self.client, method)(*args, **kwargs)
            except Exception as exc:
                self.disable(exc)


@contextmanager
def tracking_run():
    tracking = Tracking()
    active_run = None
    try:
        import mlflow

        active_run = mlflow.start_run()
        active_run.__enter__()
        tracking.client = mlflow
    except Exception as exc:
        tracking.disable(exc)

    try:
        yield tracking
    except BaseException as exc:
        if active_run is not None:
            try:
                active_run.__exit__(type(exc), exc, exc.__traceback__)
            except Exception as tracking_exc:
                tracking.disable(tracking_exc)
        raise
    else:
        if active_run is not None:
            try:
                active_run.__exit__(None, None, None)
            except Exception as exc:
                tracking.disable(exc)
