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

"""Unit tests for parameters.py."""


from spade_anomaly_detection import parameters

import tensorflow as tf


class ParametersTest(tf.test.TestCase):

  def test_none_required_parameter_raises(self):
    with self.subTest(name='two_input_sources_specified'):
      with self.assertRaises(ValueError):
        _ = parameters.RunnerParameters(
            train_setting=parameters.TrainSetting.PNU,
            input_bigquery_table_path='some_project.some_dataset.some_table',
            data_input_gcs_uri='gs://some_bucket/some_data_input_path',
            output_gcs_uri='gs://some_bucket/some_path',
            label_col_name='y',
            positive_data_value='1',
            negative_data_value='0',
            unlabeled_data_value='-1',
            voting_strategy=parameters.VotingStrategy.UNANIMOUS,
        )
    with self.subTest(name='no_input_sources_specified'):
      with self.assertRaises(ValueError):
        _ = parameters.RunnerParameters(
            train_setting=parameters.TrainSetting.PNU,
            input_bigquery_table_path=None,
            data_input_gcs_uri=None,
            output_gcs_uri='gs://some_bucket/some_path',
            label_col_name='y',
            positive_data_value='1',
            negative_data_value='0',
            unlabeled_data_value='-1',
            voting_strategy=parameters.VotingStrategy.UNANIMOUS,
        )

  def test_equal_data_value_parameter_raises(self):
    with self.assertRaises(ValueError):
      _ = parameters.RunnerParameters(
          train_setting=parameters.TrainSetting.PNU,
          input_bigquery_table_path='some_project.some_dataset.some_table',
          data_input_gcs_uri=None,
          output_gcs_uri='gs://some_bucket/some_path',
          label_col_name='y',
          positive_data_value='1',
          negative_data_value='0',
          unlabeled_data_value='0',
      )

  def test_labels_are_strings_discrepancy_raises(self):
    with self.assertRaises(TypeError):
      _ = parameters.RunnerParameters(
          train_setting=parameters.TrainSetting.PNU,
          input_bigquery_table_path='some_project.some_dataset.some_table',
          data_input_gcs_uri=None,
          output_gcs_uri='gs://some_bucket/some_path',
          label_col_name='y',
          labels_are_strings=True,
          positive_data_value=1,
          negative_data_value=0,
          unlabeled_data_value=-1,
          voting_strategy=parameters.VotingStrategy.UNANIMOUS,
      )

  def test_alpha_negative_pseudolabels_not_set_reset_to_alpha(self):
    parameters_object = parameters.RunnerParameters(
        train_setting=parameters.TrainSetting.PNU,
        input_bigquery_table_path='some_project.some_dataset.some_table',
        data_input_gcs_uri=None,
        output_gcs_uri='gs://some_bucket/some_path',
        label_col_name='y',
        labels_are_strings=False,
        positive_data_value=1,
        negative_data_value=0,
        unlabeled_data_value=-1,
        alpha=0.8,
        alpha_negative_pseudolabels=None,
        voting_strategy=parameters.VotingStrategy.UNANIMOUS,
    )
    self.assertEqual(parameters_object.alpha, 0.8)
    self.assertEqual(parameters_object.alpha_negative_pseudolabels, 0.8)


if __name__ == '__main__':
  tf.test.main()
