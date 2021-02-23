#
#   Copyright 2021 Logical Clocks AB
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


class LagomConfig:
    def __init__(self, name, description, hb_interval):
        self.name = name
        self.description = description
        self.hb_interval = hb_interval


class OptimizationConfig(LagomConfig):
    def __init__(
        self,
        num_trials,
        optimizer,
        searchspace,
        optimization_key="metric",
        direction="max",
        es_interval=1,
        es_min=10,
        es_policy="median",
        name="HPOptimization",
        description="",
        hb_interval=1,
    ):
        super().__init__(name, description, hb_interval)
        assert num_trials > 0, "Number of trials should be greater than zero!"
        self.num_trials = num_trials
        self.optimizer = optimizer
        self.optimization_key = optimization_key
        self.searchspace = searchspace
        self.direction = direction
        self.es_policy = es_policy
        self.es_interval = es_interval
        self.es_min = es_min


class AblationConfig(LagomConfig):
    def __init__(
        self,
        ablation_study,
        ablator="loco",
        direction="max",
        name="ablationStudy",
        description="",
        hb_interval=1,
    ):
        super().__init__(name, description, hb_interval)
        self.ablator = ablator
        self.ablation_study = ablation_study
        self.direction = direction


class DistributedConfig(LagomConfig):
    def __init__(
        self,
        model,
        train_set,
        test_set,
        name="torchDist",
        hb_interval=1,
        description="",
    ):
        super().__init__(name, description, hb_interval)
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
