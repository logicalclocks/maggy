"""
The experiment driver implements the functionality for scheduling trials on maggy.
"""
import queue
import threading
import datetime
import json
import maggy.util as util
from maggy.optimizer import AbstractOptimizer, RandomSearch
from maggy.core import rpc
from maggy.trial import Trial
from maggy.earlystop import AbstractEarlyStop, MedianStoppingRule
from maggy.searchspace import Searchspace


class ExperimentDriver(object):

    def __init__(self, searchspace, optimizer, direction, num_trials, name, num_executors, hb_interval, es_policy, es_interval, es_min):
        # perform type checks
        if isinstance(searchspace, Searchspace):
            self.searchspace = searchspace
        else:
            raise Exception(
                "No valid searchspace. Please use maggy Searchspace class.")

        if isinstance(optimizer, str):
            if optimizer.lower() == 'randomsearch':
                self.optimizer = RandomSearch(num_trials, self.searchspace)
            else:
                raise Exception(
                    "Unknown Optimizer. Can't initialize experiment driver.")
        elif isinstance(optimizer, AbstractOptimizer):
            print("Custom Optimizer initialized.")
            self.optimizer = optimizer
        else:
            raise Exception(
                "Unknown Optimizer. Can't initialize experiment driver.")

        if isinstance(direction, str):
            if direction.lower() not in ['min', 'max']:
                raise Exception(
                    "Unknown direction. Can't initialize experiment driver.")
        else:
            raise Exception(
                "Unknown direction. Can't initialize experiment driver.")

        if isinstance(es_policy, str):
            if es_policy.lower() == 'median':
                self.earlystop_check = MedianStoppingRule.earlystop_check
            else:
                raise Exception(
                    "Unknown Early Stopping Policy. Can't initialize experiment driver.")
        elif isinstance(es_policy, AbstractEarlyStop):
            print("Custom Early Esrly Stopping policy initialized.")
            self.earlystop_check = es_policy.earlystop_check

        self.direction = direction.lower()
        self._trial_store = {}
        self._final_store = []
        self.num_executors = num_executors
        self._message_q = queue.Queue()
        self.name = name
        self.num_trials = num_trials
        self.experiment_done = False
        self.worker_done = False
        self.hb_interval = hb_interval
        self.es_interval = es_interval
        self.es_min = es_min
        self.direction = direction.lower()
        self.server = rpc.Server(num_executors)

    def init(self):

        self.server_addr = self.server.start(self)

        self.optimizer.initialize()

        self._start_worker()

    def finalize(self, job_start, job_end):

        result = self.optimizer.finalize_experiment(self._final_store)

        self.duration = util._time_diff(job_start, job_end)

        if self.direction == 'max':
            results = '\n------ ' + str(self.optimizer.__class__.__name__) + ' results ------ direction(' + self.direction + ') \n' \
                'BEST combination ' + json.dumps(result['max_hp']) + ' -- metric ' + str(result['max_val']) + '\n' \
                'WORST combination ' + json.dumps(result['min_hp']) + ' -- metric ' + str(result['min_val']) + '\n' \
                'AVERAGE metric -- ' + str(result['avg']) + '\n' \
                'Total job time ' + self.duration + '\n'
            # TODO: write to hdfs
            print(results)
        elif self.direction == 'min':
            results = '\n------ ' + str(self.optimizer.__class__.__name__) + ' results ------ direction(' + self.direction + ') \n' \
                'BEST combination ' + json.dumps(result['min_hp']) + ' -- metric ' + str(result['min_val']) + '\n' \
                'WORST combination ' + json.dumps(result['max_hp']) + ' -- metric ' + str(result['max_val']) + '\n' \
                'AVERAGE metric -- ' + str(result['avg']) + '\n' \
                'Total job time ' + self.duration + '\n'
            # TODO: write to hdfs
            print(results)

        return result

    def get_trial(self, trial_id):
        return self._trial_store[trial_id]

    def add_trial(self, trial):
        self._trial_store[trial.trial_id] = trial

    def add_message(self, msg):
        self._message_q.put(msg)

    def _start_worker(self):

        def _target_function(self):

            time_earlystop_check = datetime.datetime.now()

            while not self.worker_done:
                trial = None
                # get a message
                try:
                    msg = self._message_q.get_nowait()
                except:
                    msg = {'type': None}

                if (datetime.datetime.now() - time_earlystop_check).total_seconds() >= self.es_interval:
                    time_earlystop_check = datetime.datetime.now()

                    # pass currently running trials to early stop component
                    if len(self._final_store) > self.es_min:
                        print("Check for early stopping.")
                        to_stop = self.earlystop_check(
                            self._trial_store, self._final_store, self.direction)
                        if len(to_stop) > 0:
                            print("Trials to stop: {}".format(to_stop))
                        for trial_id in to_stop:
                            self.get_trial(trial_id).set_early_stop()

                # depending on message do the work
                # 1. METRIC
                if msg['type'] == 'METRIC':
                    self.get_trial(msg['trial_id']).append_metric(msg['data'])

                # 2. BLACK
                elif msg['type'] == 'BLACK':
                    trial = self.get_trial(msg['trial_id'])
                    with trial.lock:
                        trial.status = Trial.SCHEDULED
                        self.server.reservations.assign_trial(
                            msg['partition_id'], msg['trial_id'])
                        # print("Scheduled Trial: " + trial.to_json())

                # 3. FINAL
                elif msg['type'] == 'FINAL':
                    # set status
                    # get trial only once
                    trial = self.get_trial(msg['trial_id'])

                    # finalize the trial object
                    with trial.lock:
                        trial.status = Trial.FINALIZED
                        trial.final_metric = msg['data']
                        trial.duration = util._time_diff(
                            trial.start, datetime.datetime.now())

                    # move trial to the finalized ones
                    self._final_store.append(trial)
                    self._trial_store.pop(trial.trial_id)

                    # TODO: make json and write to HDFS

                    # assign new trial
                    trial = self.optimizer.get_suggestion(trial)
                    if trial is None:
                        self.server.reservations.assign_trial(
                            msg['partition_id'], None)
                        self.experiment_done = True
                    else:
                        with trial.lock:
                            trial.start = datetime.datetime.now()
                            trial.status = Trial.SCHEDULED
                            self.server.reservations.assign_trial(
                                msg['partition_id'], trial.trial_id)
                            self.add_trial(trial)

                # 4. REG
                elif msg['type'] == 'REG':
                    trial = self.optimizer.get_suggestion()
                    if trial is None:
                        self.experiment_done = True
                    else:
                        with trial.lock:
                            trial.start = datetime.datetime.now()
                            trial.status = Trial.SCHEDULED
                            self.server.reservations.assign_trial(
                                msg['partition_id'], trial.trial_id)
                            self.add_trial(trial)

        t = threading.Thread(target=_target_function, args=(self,))
        t.daemon = True
        t.start()

    def stop(self):
        """Stop the Driver's worker thread and server."""
        self.worker_done = True
        self.server.stop()

    def experiment_json(self):
        pass
