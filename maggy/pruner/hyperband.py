#
#   Copyright 2020 Logical Clocks AB
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

"""
The implementation is heavliy inspired by the BOHB (Falkner et al. 2018) paper and the HpBandSter Framework

BOHB: http://proceedings.mlr.press/v80/falkner18a.html
HpBandSter: https://github.com/automl/HpBandSter
"""

import numpy as np

from maggy.pruner.abstractpruner import AbstractPruner


class Hyperband(AbstractPruner):
    """
    **Hyperband**

    Quote from BOHB_ Paper:

    Hyperband_ (HB) (Li et al., 2017) is a multi-armed bandit strategy for hyperparameter optimization that takes ad-
    vantage of these different budgets b by repeatedly calling SuccessiveHalving (SH) (Jamieson & Talwalkar, 2016) to
    identify the best out of n sampled configurations. It balances very agressive evaluations with many configu-
    rations on the smallest budget, and very conservative runs that are directly evaluated on max_budget

    In the original HB paper random sampling is used to sample new configurations. With this implementation any sampling
    algorithms (i.e. model based bayesian optimization algorithms) can be used.

    .. _Hyperband: http://jmlr.org/papers/v18/16-558.html
    .. _BOHB: http://proceedings.mlr.press/v80/falkner18a.html

    **Integration into Maggy**

    Hyperband is initialized as a subroutine in an instance of `BaseAsyncBO` (optimizer) and its method
    `pruning_routine()` is called at the beginning of the `get_suggestion()` method of the optimizer to return the budget
    and hparam config for the next Trial

    **Parallelization**

    - Start with first SH run that sequential HB would perform ( most aggressive one starting from lowest budget )
    - Sample configurations with TPE until either (a) all workers are busy, or (b) enough configs have been sampled for this SH run
        - in case (a), we simply wait for a worker to free up and then sample a new configuration
        - in case (b), start nex SH run in parallel
    - Observations D are shared across all SH runs

    *System's view*

    - All workers are joined into single pool, whenever a worker becomes available preferentially execute waiting runs with smaller budgets.
    - New SH runs are only started, when the SH runs currently executed are not waiting for a worker to free up
    """

    def __init__(self, min_budget, max_budget, eta, n_iterations, **kwargs):
        """
        Note: `trial_metric_getter` is param of parent class and has to be passed as kwarg

        Attributes
        ----------

        - max_sh_rungs (int): max amount of rungs in the SH iterations
        - budgets (np.array[int]): budgets used for calculating budgets of SH iterations
        - iterations (list(SHIteration)): list of initialized SH iterations
        - updating_iteration (None|int): id of currently updating SH iteration

        :param min_budget: The smallest budget to consider. Needs to be positive!
        :type min_budget: int
        :param max_budget: The largest budget to consider. Needs to be larger than min_budget!
            The budgets will be geometrically distributed
        :type max_budget: int
        :param eta: In each iteration, a complete run of sequential halving is executed. In it,
            after evaluating each configuration on the same subset size, only a fraction of
            1/eta of them 'advances' to the next round.
            Must be greater or equal to 2.
        :type eta: int
        :param n_iterations: number of SH Iterations
        :type n_iterations: int
        :param trial_metric_getter: a that returns a dict with `trial_id` as key and `metric` as value
            with the lowest metric being the "best"
            It's only argument is `trial_ids`, it can be either str of single trial or list of trial ids
        :type trial_metric_getter: function
        """
        super().__init__(**kwargs)

        # checks
        if not min_budget > 0:
            raise ValueError("Expected `min_budget` > 0, got {}".format(min_budget))
        if min_budget >= max_budget:
            raise ValueError(
                "max_budget needs to be larger than min_budget, got {}, {}".format(
                    max_budget, min_budget
                )
            )
        if eta < 2:
            raise ValueError("Expected eta greater or equal to 2, got {}".format(eta))

        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        self.n_iterations = n_iterations

        # calculate HB params
        self.max_sh_rungs = (
            -int(np.log(self.min_budget / self.max_budget) / np.log(self.eta)) + 1
        )
        self.budgets = np.array(
            self.max_budget
            * np.power(
                self.eta, -np.linspace(self.max_sh_rungs - 1, 0, self.max_sh_rungs)
            ),
            dtype=int,
        ).tolist()
        # convert tolist to convert values from np.int64 to int, necessary to be json serializable when creating trialid

        # configure SH iterations
        self.iterations = []
        self.init_iterations()

        # start first SH iteration
        self.start_next_iteration()

        # keep track of `iteration_id` that is currently updating, needed for `self.report_trial()`
        self.updating_iteration = None

    def pruning_routine(self):
        """Returns dict with keys 'trial_id' and 'budget'

        This method is the interface to `Optimizer` and is called in the `get_suggestion()` method.
        It decides over the budget and hparams for the next trial in the optimization loop

        **There are 4 possible outcomes:**

        1. There are still slots to fill in the first rung of an active SH Iteration and hence the optimizer should
           sample a new hparam config from its model.
            - return {"trial_id": None, "budget": `budget`}
        2. A hparam config was promoted to the next rung of a SH Iteration and hence a new trial should be started with
           these hparams on the budget of the next rung.
            - return {"trial_id": `promoted_trial_id`, "budget": `budget`}
        3. It is not possible to immediatley start a new run, because all Iterations have been started and are currently
           busy ( i.e. all trials in current rung are evaluating )
            - return "IDLE"
        4. All SH Iterations have been finished
            - return None

        :return: {"trial_id": `trial_id`, "budget": 9}
        :rtype: dict|None|str
        """
        # loop through active iterations and return config and budget of next run
        next_run = None
        for iteration in self.active_iterations():
            next_run = iteration.get_next_run()
            if next_run is not None:
                # set updateing iteration
                self.updating_iteration = iteration.iteration_id
                break

        if next_run is not None:
            # schedule new run for `iteration`
            self._log(
                "{}. Iteration, {}. Rung. Run next {}".format(
                    iteration.iteration_id, iteration.current_rung, next_run
                )
            )
            return next_run

        else:
            # all active iterations are busy or finished, no immediate run can be scheduled
            if self.n_iterations > 0:
                # start next iteration in the queue
                self.start_next_iteration()
                return self.pruning_routine()
            elif self.finished():
                # All SH iterations in HB are finished
                self._log("All Iterations have finished")
                self._close_log()
                return None
            else:
                # no immediate run can be scheduled, because all iterations are busy.
                self._log(
                    "All Iterations have been started and all trials in their current rung have been started. "
                    "Wait until new run can be scheduled"
                )
                return "IDLE"

    def init_iterations(self):
        """calculates budgets and amount of trials for each iteration"""

        for iteration in range(self.n_iterations):
            # number of rungs
            n_rungs = self.max_sh_rungs - 1 - (iteration % self.max_sh_rungs)
            # number of configurations per rung
            n0 = int(
                np.floor(self.max_sh_rungs / (n_rungs + 1)) * self.eta**n_rungs
            )  # configs in first rung
            ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(n_rungs + 1)]
            # budgets per rung
            budgets = self.budgets[-n_rungs - 1 :]
            self.iterations.append(
                SHIteration(
                    n_configs=ns,
                    budgets=budgets,
                    iteration_id=iteration,
                    trial_metric_getter=self.trial_metric_getter,
                    logger=self._log,
                )
            )

    def active_iterations(self):
        """returns currently active (i.e. state == "RUNNING") iterations

        sorted in ascending order w.r.t. iteration idx

        :rtype: list[SHIteration]
        """
        active = [
            iteration
            for iteration in self.iterations
            if iteration.state == SHIteration.RUNNING
        ]
        return active

    def start_next_iteration(self):
        """Sets state of next SH iteration in queue to RUNNING"""
        for iteration in self.iterations:
            if iteration.state == SHIteration.INIT:
                iteration.state = SHIteration.RUNNING
                self._log(
                    "{}. Iteration started. n_configs: {}, budgets: {}".format(
                        iteration.iteration_id, iteration.n_configs, iteration.budgets
                    )
                )
                self.n_iterations -= 1
                break

    def finished(self):
        """returns True if all iterations have finished

        :return: True, if all iterations have state == 'FINISHED'. Else, False
        :rtype: bool
        """
        for iteration in self.iterations:
            if iteration.state != SHIteration.FINISHED:
                return False

        return True

    def num_trials(self):
        n_trials = 0
        for iteration in self.iterations:
            n_trials += sum(iteration.n_configs)

        return n_trials

    def report_trial(self, original_trial_id, new_trial_id):
        """reports new trial_id to HB, i.e. add trial_id to the currently updateing SH iteration

        This method is an interface to the `optimizer` and is called at the end of `get_suggestion()`

        :param original_trial_id: if it was a promoted trial, the original id of the promoted trial
        :type original_trial_id: None|str
        :param new_trial_id: the id of the newly started trial
        :type new_trial_id: str
        """
        self.iterations[self.updating_iteration].report_trial(
            original_trial_id, new_trial_id
        )
        self.updating_iteration = None


class SHIteration:
    """SuccessiveHalving Iteration

    **Algorithm**

    Quote from BOHB_ Paper:

    SuccessiveHalving is a simple heuristic to allocate more resources to promising candidates. It is initialized
    with a set of configurations, a minimum and maximum budget, and a scaling parameter η. In the first stage all
    configurations are evaluated on the smallest budget. The losses are then sorted and only the best 1/η
    configurations are kept in the set C. For the following stage, the budget is increased by a factor of η. This is
    repeated until the maximum budget for a single configura- tion is reached. Within Hyperband, the budgets are
    chosen such that all SuccessiveHalving executions require a similar total budget.

    .. _BOHB: http://proceedings.mlr.press/v80/falkner18a.html
    """

    # Iteration states
    INIT = "INIT"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"

    def __init__(self, n_configs, budgets, iteration_id, trial_metric_getter, logger):
        """
        Attributes
        ----------

        actual_configs (list[int]): number of configs that have been started in each rung. To complete a SH iteration
                                    it has do be equal to `n_configs`.
                                    actual_n_configs[current_rung] and len(configs[current_rung]) are eventually
                                    consistent. The former gets increased before info about next trial is returned to
                                    the optimizer. The latter is appended to when a trial is created in the optimizer.
        configs (dict): keeps track of trial ids of configs in each rung. For each trial the "original_trial_id" and
                        "actual_trial_id" are stored.
                        In the first rung these to are the same and refer to the `trial_id`
                        of the finished trials of rung 0.
                        In rungs > 0, The former is the `trial_id` of the promoted trial of the first rung
                        and the latter the `trial_id` of the trial with the same hparams - performed with the budget
                        of the current rung.
                        Having a `actual_trial_id` means that a trial has been started, but not neccessarily finished
        n_rungs (int): number of rungs in the SH iteration
        state (str): current state of the iteration, can be "INIT", "RUNNING" or "FINISHED

        Note: this strategy gives also the opportunity to later continue
              trials instead of starting new ones for promoted trials

        Example for `configs`:

        {0:[
            {"original_trial_id":trial_id00
             "actual_trial_id": trial_id00},
            ...,
             {"original_trial_id": trial_id08
             "actual_trial_id": trial_id08}
           ],
        1: [
            {"original_trial_id":trial_id01
             "actual_trial_id": trial_id10},
            ...,
             {"original_trial_id": trial_id07
             "actual_trial_id": trial_id12}
            ],
        2:  [
            {"original_trial_id":trial_id01
             "actual_trial_id": trial_id20}
            ],
        }

        :param n_configs: number of trials per rung
        :type n_configs: list[int]
        :param budgets: budget per rung
        :type budgets: list[int]
        :param iteration_id: the id of the iteration is the index of the iteration in the `iterations` list of the pruner
        :type iteration_id: int
        :param trial_metric_getter: a function that returns a finished Trial object or a list of finished Trial objects from
                             the `final_store` of the `optimizer`.
                             It's only argument is the `trial_id` or a list of `trial_id`
        :type trial_metric_getter: function
        :param logger: logger
        """
        self.iteration_id = iteration_id
        self.state = SHIteration.INIT
        self.n_configs = n_configs  # [9, 3, 1]  n_configs per rung
        self.budgets = budgets  # [1, 3, 9]

        self.n_rungs = len(n_configs)
        self.current_rung = 0
        self.actual_n_configs = [0] * len(self.n_configs)
        self.configs = {rung: [] for rung in range(0, self.n_rungs)}

        self.trial_metric_getter = trial_metric_getter

        # configure logger
        self._log = logger

    def get_next_run(self):
        """returns dict with `trial_id` and `budget` for next trial.

        If iteration is currently busy, i.e has to wait for running trials to finish before it can promote trials
        to next rung or iteration has finished, return None

        **There are 3 possible outcomes:**

        1. There are still slots to fill and `current_rung` == 0
            - return {"trial_id": None, "budget": `budget`}
            - the optimizer will sample no hparam config from its model
        2. There are still slots to fill and `current_rung` == 0
            - return {"trial_id": `promoted_trial_id`, "budget": `budget`}
        3. If iteration is currently busy, i.e has to wait for running trials to finish before it can promote trials
           to next rung or iteration has finished
            - return None

        In case all trials of current rung have finished, promote best performing trials to next rung and rerun this
        method

        :return: dict with info about trial id and budget for the next run in the iteration, or None if iteration is
                 busy or finished.
                 example: {"trial_id": `trial_id`, "budget": 9}
        :rtype: None|dict
        """
        if self.n_configs[self.current_rung] > self.actual_n_configs[self.current_rung]:
            # there are still slots to fill in current rung
            if self.current_rung == 0:
                self.actual_n_configs[self.current_rung] += 1
                return {"trial_id": None, "budget": self.budgets[self.current_rung]}
            else:
                for trial in self.configs[self.current_rung]:
                    # If an `actual_trial_id` has been added, the optimizer has already started that trial
                    if trial["actual_trial_id"]:
                        continue

                    # Return trial id of promoted trial to optimizer, the optimizer will start a trial with the same
                    # params and add its trial_id as `actual_trial_id` to the `configs` dict
                    self.actual_n_configs[self.current_rung] += 1
                    return {
                        "trial_id": trial["original_trial_id"],
                        "budget": self.budgets[self.current_rung],
                    }

        elif (
            self.n_configs[self.current_rung]
            == self.actual_n_configs[self.current_rung]
        ):
            # all slots in current rung have been filled
            if self.promotable():
                # promote best performing trials to next rung
                self.promote()
                return self.get_next_run()
            else:
                # all trials of current rung are started but not finished yet or last rung has finished
                # check if SH iteration is finished
                if self.finished():
                    # set state so it is no longer returned in `active_iterations()`
                    self.state = SHIteration.FINISHED
                    self._log("{}. Iteration finished".format(self.iteration_id))
                return None
        else:
            raise ValueError(
                "Too many configs have been sampled in iteration {}".format(
                    self.iteration_id
                )
            )

    def report_trial(self, original_trial_id, new_trial_id):
        """adds trial_id to iteration

        is called from `pruner.report_trial()` which is called at the end of `optimizer.get_suggestion()`

        in the first rung there are no promoted trials, hence there are no `original_trial_id`, to keep consistency
        of the trial dicts in `configs`, set `original_trial_id` to same value as `actual_trial_id`

        :param original_trial_id: if it was a promoted trial, the original id of the promoted trial
        :type original_trial_id: None|str
        :param new_trial_id: the id of the newly started trial
        :type new_trial_id: str
        """
        if self.current_rung == 0:
            original_trial_id = new_trial_id
            self.configs[self.current_rung].append(
                {
                    "original_trial_id": original_trial_id,
                    "actual_trial_id": new_trial_id,
                }
            )
        else:
            # find index of trial and insert actual trial id
            trial_idx = next(
                (
                    index
                    for (index, d) in enumerate(self.configs[self.current_rung])
                    if d["original_trial_id"] == original_trial_id
                ),
                None,
            )
            self.configs[self.current_rung][trial_idx]["actual_trial_id"] = new_trial_id

        self._log(
            "{}. Iteration, {}. Rung. Started Trial {}/{}".format(
                self.iteration_id,
                self.current_rung,
                self.actual_n_configs[self.current_rung],
                self.n_configs[self.current_rung],
            )
        )

    def promote(self):
        """promotes n_configs to the next rung based on final metric

        only call this method if `promotable()` returns True.

        :return: list of trial ids that are advancing to the next rung
        :rtype: list[str]
        """
        # get trial ids of current rung
        trial_ids = [
            trial["actual_trial_id"] for trial in self.configs[self.current_rung]
        ]

        # get trials from `final_store` of optimizer
        trial_metrics = self.trial_metric_getter(
            trial_ids
        )  # {`trial_id`: `metric`, ... }

        # sort trials
        sorted_trials = list(
            {
                k: v for k, v in sorted(trial_metrics.items(), key=lambda item: item[1])
            }.keys()
        )

        # promoted trials
        n_promote = self.n_configs[self.current_rung + 1]
        promoted_trials = sorted_trials[:n_promote]

        # promote trials to next rung
        self.current_rung += 1
        for trial in promoted_trials:
            self.configs[self.current_rung].append(
                {"original_trial_id": trial, "actual_trial_id": None}
            )

        self._log(
            "{}. Iteration finished rung: {} \n with trials: {} \n promoted trials: {}".format(
                self.iteration_id, self.current_rung - 1, sorted_trials, promoted_trials
            )
        )

    def promotable(self):
        """checks if current rung is promotable, i.e. if all trials are finished and current rung is not the last rung

        :return: True if all trials of rung are finished, False else
        :rtype: bool
        """
        self._log(
            "{}. Iteration, check if rung {} is promotable".format(
                self.iteration_id, self.current_rung
            )
        )
        if len(self.configs[self.current_rung]) < self.n_configs[self.current_rung]:
            # not all trials have been created and started by the optimzer
            self._log(
                "{}. Iteration, rung {} is not promotable. Not all slots in rung filled yet".format(
                    self.iteration_id, self.current_rung
                )
            )
            return False

        if self.current_rung == self.n_rungs - 1:
            # current rung is the last rung in the iteration
            self._log(
                "{}. Iteration, rung {} is the last rung and hence not promotable.".format(
                    self.iteration_id, self.current_rung
                )
            )
            return False

        for trial in self.configs[self.current_rung]:
            if not self.trial_metric_getter(trial["actual_trial_id"]):
                # trial has not finished
                self._log(
                    "{}. Iteration, rung {} is not promotable. At least one trial ({}) is not finished yet".format(
                        self.iteration_id, self.current_rung, trial["actual_trial_id"]
                    )
                )
                return False

        self._log(
            "{}. Iteration, rung {} is not promotable. Not all slots in rung filled yet".format(
                self.iteration_id, self.current_rung
            )
        )
        return True

    def finished(self):
        """checks if SH Iteration has finished, i.e. if all trials in the last rung are finished

        :return: True if SH Iteration is finished, False else
        :rtype: bool
        """
        if len(self.configs[self.current_rung]) < self.n_configs[self.current_rung]:
            # not all trials in current rung have started
            return False

        if self.current_rung != self.n_rungs - 1:
            # current rung is not the last rung in the iteration
            return False

        for trial in self.configs[self.current_rung]:
            if not self.trial_metric_getter(trial["actual_trial_id"]):
                # trial has not finished
                return False

        return True
