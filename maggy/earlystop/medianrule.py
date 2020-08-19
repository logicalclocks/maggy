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

import statistics
from maggy.earlystop.abstractearlystop import AbstractEarlyStop


class MedianStoppingRule(AbstractEarlyStop):
    """The Median Stopping Rule implements the simple strategy of stopping a
    trial if its performance falls below the median of other trials at similar
    points in time.
    """

    @staticmethod
    def earlystop_check(to_check, finalized_trials, direction):

        results = []
        median = None

        # count step from zero so it can be used as index for array
        step = len(to_check.metric_history)

        if step > 0:

            for fin_trial in finalized_trials:

                if len(fin_trial.metric_history) >= step:
                    avg = sum(fin_trial.metric_history[:step]) / float(step)
                    results.append(avg)

            try:
                median = statistics.median(results)
            except statistics.StatisticsError as e:
                raise Exception(
                    "Warning: StatisticsError when calling early stop method\n{}".format(
                        e
                    )
                )

            if median is not None:
                if direction == "max":
                    if max(to_check.metric_history) < median:
                        return to_check.trial_id
                elif direction == "min":
                    if min(to_check.metric_history) > median:
                        return to_check.trial_id
            return None
