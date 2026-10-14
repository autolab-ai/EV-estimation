"""Best-effort MLflow logging using only the runtime-provided run identity."""


class Tracking:
    def __init__(self):
        self._mlflow = None
        self._context = None
        self._enabled = False
        self._warned = False

    def _disable(self, exc):
        self._enabled = False
        if not self._warned:
            print(f"Warning: MLflow tracking unavailable ({type(exc).__name__}); continuing.", flush=True)
            self._warned = True

    def __enter__(self):
        try:
            import mlflow
            self._mlflow = mlflow
            context = mlflow.start_run()
            context.__enter__()
            self._context = context
            self._enabled = True
        except Exception as exc:
            self._disable(exc)
        return self

    def _log(self, method, *args, **kwargs):
        if self._enabled:
            try:
                getattr(self._mlflow, method)(*args, **kwargs)
            except Exception as exc:
                self._disable(exc)

    def log_params(self, params):
        self._log("log_params", params)

    def log_metric(self, name, value, step=0):
        self._log("log_metric", name, value, step=step)

    def __exit__(self, exc_type, exc_value, traceback):
        if self._context is not None:
            try:
                self._context.__exit__(exc_type, exc_value, traceback)
            except Exception as exc:
                self._disable(exc)
        return False
