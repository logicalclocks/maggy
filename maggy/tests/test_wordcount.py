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

import pytest
from operator import add

# this allows using the fixture in all tests in this module
pytestmark = pytest.mark.usefixtures("sc")

# Can also use a decorator such as this to use specific fixtures in specific functions
# @pytest.mark.usefixtures("spark_context", "hive_context")


def do_word_counts(lines):
    """ count of words in an rdd of lines """

    counts = lines.flatMap(lambda x: x.split()).map(lambda x: (x, 1)).reduceByKey(add)
    results = {word: count for word, count in counts.collect()}
    return results


# start function with test_ so pytest can discover them
def test_do_word_counts(sc):
    """test that a single event is parsed correctly
    Args:
        spark_context: test fixture SparkContext
        hive_context: test fixture HiveContext
    """

    test_input = [" hello spark ", " hello again spark spark"]

    input_rdd = sc.parallelize(test_input, 1)
    results = do_word_counts(input_rdd)

    expected_results = {"hello": 2, "spark": 3, "again": 1}
    assert results == expected_results
