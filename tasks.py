from celery import Celery
import numpy as np

# Configure the app with broker URL (RabbitMQ here)
app = Celery(
    "tasks",
    broker="pyamqp://guest@localhost//",
    backend="rpc://",
)

app.conf.update(
    task_serializer="pickle", result_serializer="pickle", accept_content=["pickle"]
)


class Levy:
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 1 + (x - 1.0) / 4.0
        val = (
            np.sin(np.pi * w[0]) ** 2
            + np.sum(
                (w[1 : self.dim - 1] - 1) ** 2
                * (1 + 10 * np.sin(np.pi * w[1 : self.dim - 1] + 1) ** 2)
            )
            + (w[self.dim - 1] - 1) ** 2
            * (1 + np.sin(2 * np.pi * w[self.dim - 1]) ** 2)
        )
        return val


f = Levy(10)


@app.task
def celery_f(x, dim=10):
    return f(np.array(x))


celery_f.dim = f.dim
celery_f.lb = f.lb
celery_f.ub = f.ub
