from maggy.optimizer import AbstractOptimizer
from maggy.trial import Trial


class SingleRun(AbstractOptimizer):

    def __init__(self):
        super().__init__()
        self.trial_buffer = []

    def initialize(self):
        for _ in range(self.num_trials):
            self.trial_buffer.append(Trial({}))

    def get_suggestion(self, trial=None):
        if self.trial_buffer:
            return self.trial_buffer.pop()
        else:
            return None

    def finalize_experiment(self, trials):
        return
