"""Best-effort telemetry using the run identity supplied by the environment."""


class Tracking:
    def __init__(self):
        self.client = None
        self.run = None
        self.warned = False

    def _disable(self, exc):
        self.client = None
        if not self.warned:
            print(f"Warning: MLflow tracking unavailable: {exc}", flush=True)
            self.warned = True

    def __enter__(self):
        try:
            import mlflow

            self.run = mlflow.start_run()
            self.client = mlflow
        except Exception as exc:
            self._disable(exc)
        return self

    def log_params(self, params):
        if self.client is not None:
            try:
                self.client.log_params(params)
            except Exception as exc:
                self._disable(exc)

    def log_metric(self, name, value, step=0):
        if self.client is not None:
            try:
                self.client.log_metric(name, value, step=step)
            except Exception as exc:
                self._disable(exc)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.run is not None:
            try:
                self.run.__exit__(exc_type, exc_value, traceback)
            except Exception as exc:
                self._disable(exc)
        return False
