"""Best-effort MLflow telemetry using the run identity supplied by AutoLab."""


class Tracking:
    def __init__(self):
        self.client = None
        self.context = None
        self.warned = False

    def _disable(self, error):
        if not self.warned:
            print(f"Warning: MLflow tracking unavailable: {error}", flush=True)
            self.warned = True
        self.client = None

    def __enter__(self):
        try:
            import mlflow

            context = mlflow.start_run()
            context.__enter__()
            self.context = context
            self.client = mlflow
        except Exception as error:
            self._disable(error)
        return self

    def _log(self, method, *args, **kwargs):
        if self.client is not None:
            try:
                getattr(self.client, method)(*args, **kwargs)
            except Exception as error:
                self._disable(error)

    def log_params(self, params):
        self._log("log_params", params)

    def log_metric(self, name, value, step=None):
        self._log("log_metric", name, value, step=step)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.context is not None:
            try:
                self.context.__exit__(exc_type, exc_value, traceback)
            except Exception as error:
                self._disable(error)
        return False
