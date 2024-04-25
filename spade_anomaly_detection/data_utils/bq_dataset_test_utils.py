# Copyright 2024 The spade_anomaly_detection Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
# Copyright 2024 The SPADE Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for helping to run tests with bq_dataset."""
from typing import Sequence

import pandas as pd


def mock_get_bigquery_dataset_return_value(
    mock_client: ...,
    return_value: Sequence[pd.DataFrame],
    mock_list_rows: bool = True,
    mock_query: bool = True,
) -> None:
  """A utility to help mock out bq_dataset return values.

  Args:
    mock_client: A mock of a bigquery.Client instance. This can be created with:
      mock.create_autospec(bigquery.Client, instance=True, spec_set=True).
    return_value: The return value
    mock_list_rows:
    mock_query:
  """
  if mock_query:
    (
        mock_client.query.return_value.result.return_value.to_dataframe_iterable.return_value
    ) = return_value
  if mock_list_rows:
    (mock_client.list_rows.return_value.to_dataframe_iterable.return_value) = (
        return_value
    )
