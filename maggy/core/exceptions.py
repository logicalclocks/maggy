
class EarlyStopException(Exception):

    def __init__(self, metric):
        super().__init__()

        self.metric = metric
