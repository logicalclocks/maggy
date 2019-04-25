import random

from maggy.optimizer import AbstractOptimizer
from maggy.searchspace import Searchspace
from maggy.trial import Trial


class RandomSearch(AbstractOptimizer):

    def initialize(self):

        if (Searchspace.DOUBLE not in self.searchspace.names().values() and
                Searchspace.INTEGER not in self.searchspace.names().values()):
            raise NotImplementedError(
                "Searchspace needs at least one continuous parameter for random search.")

        for _ in range(self.num_trials):

            params = {}

            for name, value in self.searchspace.names().items():

                feasible_region = self.searchspace.get(name)

                if value == Searchspace.DOUBLE:
                    params[name] = random.uniform(feasible_region[0],
                                                  feasible_region[1])
                elif value == Searchspace.INTEGER:
                    params[name] = random.randint(feasible_region[0],
                                                  feasible_region[1])
                elif value == Searchspace.DISCRETE:
                    params[name] = random.choice(feasible_region)
                elif value == Searchspace.CATEGORICAL:
                    params[name] = random.choice(feasible_region)

            self.trial_buffer.append(Trial(params))

    def get_suggestion(self, trial=None):
        if self.trial_buffer:
            return self.trial_buffer.pop()
        else:
            return None

    def finalize_experiment(self, trials):

        first = True
        results = []
        num_trials = 0

        for trial in trials:

            if trial.status == Trial.FINALIZED:

                num_trials += 1

                metric = trial.final_metric
                param_string = trial.params

                if first:
                    max_id = trial.trial_id
                    max_hp = param_string
                    max_val = metric
                    min_id = trial.trial_id
                    min_hp = param_string
                    min_val = metric
                    first = False

                if metric > max_val:
                    max_id = trial.trial_id
                    max_val = metric
                    max_hp = param_string
                if metric < min_val:
                    min_id = trial.trial_id
                    min_val = metric
                    min_hp = param_string

                results.append(metric)

        avg = sum(results)/float(len(results))

        return {'max_id': max_id, 'max_val': max_val,
                'max_hp': max_hp, 'min_id': min_id,
                'min_val': min_val, 'min_hp': min_hp,
                'avg': avg, 'num_trials': num_trials}