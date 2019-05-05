from maggy.optimizer import AbstractOptimizer
from maggy.searchspace import Searchspace
from maggy.trial import Trial


class RandomSearch(AbstractOptimizer):

    def initialize(self):

        if (Searchspace.DOUBLE not in self.searchspace.names().values() and
                Searchspace.INTEGER not in self.searchspace.names().values()):
            raise NotImplementedError(
                "Searchspace needs at least one continuous parameter for random search.")

        list_of_random_trials = self.searchspace.get_random_parameter_values(self.num_trials)
        for parameters_dict in list_of_random_trials:
            self.trial_buffer.append(Trial(parameters_dict))

    def get_suggestion(self, trial=None):
        if self.trial_buffer:
            return self.trial_buffer.pop()
        else:
            return None

    def finalize_experiment(self, trials):
        return
