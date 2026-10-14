"""Best-effort telemetry using the run identity supplied by the environment."""

from contextlib import ExitStack, contextmanager

_client = None
_warned = False


def _disable(exc):
    global _client, _warned
    _client = None
    if not _warned:
        print(f"Warning: tracking disabled: {exc}", flush=True)
        _warned = True


@contextmanager
def tracking_run():
    global _client
    stack = ExitStack()
    try:
        import mlflow

        stack.enter_context(mlflow.start_run())
        _client = mlflow
    except Exception as exc:
        _disable(exc)
    try:
        yield
    finally:
        try:
            stack.close()
        except Exception as exc:
            _disable(exc)
        _client = None


def log_params(params):
    if _client is not None:
        try:
            _client.log_params(params)
        except Exception as exc:
            _disable(exc)


def log_metric(name, value, step):
    if _client is not None:
        try:
            _client.log_metric(name, value, step=step)
        except Exception as exc:
            _disable(exc)
