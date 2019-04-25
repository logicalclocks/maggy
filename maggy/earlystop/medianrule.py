import statistics
from maggy.earlystop import AbstractEarlyStop


class MedianStoppingRule(AbstractEarlyStop):
    """The Median Stopping Rule implements the simple strategy of stopping a
    trial if its performance falls below the median of other trials at similar
    points in time.
    """
    @staticmethod
    def earlystop_check(to_check, finalized_trials, direction):

        stop = []

        for trial_id, trial in to_check.items():

            results = []
            median = None

            # count step from zero so it can be used as index for array
            step = len(trial.metric_history)

            if step > 0:

                for fin_trial in finalized_trials:

                    if len(fin_trial.metric_history) >= step:
                        avg = sum(fin_trial.metric_history[:step])/float(step)
                        results.append(avg)

                try:
                    median = statistics.median(results)
                except Exception as e:
                    print(e)
                    raise

                if median is not None:
                    if direction == 'max':
                        if max(trial.metric_history) < median:
                            stop.append(trial_id)
                    elif direction == 'min':
                        if min(trial.metric_history) > median:
                            stop.append(trial_id)

        return stop
