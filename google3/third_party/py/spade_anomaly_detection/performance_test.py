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

"""Benchmarks the quality of the SPADE algorithm.

This is useful for discovering a CL that could negatively affect performance
of the entire SPADE algorithm that was not caught during the unit testing of
individual modules and functions.
"""

from unittest import mock

from spade_anomaly_detection import data_loader
from spade_anomaly_detection import parameters
from spade_anomaly_detection import runner
from spade_anomaly_detection import supervised_model

import tensorflow as tf


class PerformanceTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    # Using the 10k row covertype dataset here to speed up testing time. The
    # 100k version was timing out on blaze.
    self.total_record_count = 10000
    csv_path = f'covertype_pnu_{self.total_record_count}'

    self.runner_parameters = parameters.RunnerParameters(
        train_setting='PNU',
        input_bigquery_table_path='project_id.dataset.table_name',
        output_gcs_uri='gs://test_bucket/test_folder',
        label_col_name='label',
        positive_data_value=1,
        negative_data_value=0,
        unlabeled_data_value=-1,
        positive_threshold=10,
        negative_threshold=90,
        test_dataset_holdout_fraction=0.3,
        alpha=1.0,
        batches_per_model=1,
        max_occ_batch_size=50000,
        labeling_and_model_training_batch_size=self.total_record_count,
        ensemble_count=5,
        verbose=True,
    )

    self.unlabeled_features, self.unlabeled_labels = data_loader.load_dataframe(
        dataset_name=csv_path,
        filter_label_value=self.runner_parameters.unlabeled_data_value,
    )
    self.unlabeled_record_count = len(self.unlabeled_labels)

    self.occ_fit_batch_size = (
        self.unlabeled_record_count // self.runner_parameters.ensemble_count
    ) // self.runner_parameters.batches_per_model

    self.unlabeled_tensorflow_dataset = data_loader.load_tf_dataset_from_csv(
        dataset_name=csv_path,
        batch_size=self.occ_fit_batch_size,
        filter_label_value=self.runner_parameters.unlabeled_data_value,
    )
    self.complete_tensorflow_dataset = data_loader.load_tf_dataset_from_csv(
        dataset_name=csv_path,
        batch_size=self.runner_parameters.labeling_and_model_training_batch_size,
    )

    self.mock_get_total_records = self.enter_context(
        mock.patch.object(
            data_loader.DataLoader,
            'get_query_record_result_length',
            autospec=True,
        )
    )
    self.mock_supervised_model_save = self.enter_context(
        mock.patch.object(
            supervised_model.RandomForestModel,
            'save',
            autospec=True,
            instance=True,
        )
    )
    self.mock_data_loader = self.enter_context(
        mock.patch.object(
            data_loader.DataLoader,
            'load_tf_dataset_from_bigquery',
            autospec=True,
        )
    )

  def test_spade_auc_performance_pnu_single_batch(self):
    self.runner_parameters.train_setting = parameters.TrainSetting.PNU
    self.runner_parameters.positive_threshold = 10
    self.runner_parameters.negative_threshold = 90

    self.mock_data_loader.side_effect = [
        self.unlabeled_tensorflow_dataset,
        self.complete_tensorflow_dataset,
    ]
    self.mock_get_total_records.side_effect = [
        self.total_record_count,
        self.unlabeled_record_count,
    ]

    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    auc = runner_object.supervised_model_metrics['Supervised_Model_AUC']
    # See SPADE performance for Covertype PNU Setting in the design document for
    # setting and adjusting the AUC here. 0.9251 roughly equates to the
    # performance seen on the ~580k row Coertype dataset in the PNU setting.
    self.assertAlmostEqual(auc, 0.9251, delta=0.02)

  def test_spade_auc_performance_pu_single_batch(self):
    self.runner_parameters.train_setting = parameters.TrainSetting.PU
    self.runner_parameters.positive_threshold = 10
    self.runner_parameters.negative_threshold = 50

    self.mock_data_loader.side_effect = [
        self.unlabeled_tensorflow_dataset,
        self.complete_tensorflow_dataset,
    ]
    self.mock_get_total_records.side_effect = [
        self.total_record_count,
        self.unlabeled_record_count,
    ]

    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    auc = runner_object.supervised_model_metrics['Supervised_Model_AUC']
    # See SPADE performance for Covertype PU Setting in the design document for
    # setting and adjusting the AUC here. 0.8870 represents the performance seen
    # on the ~580k row Coertype dataset in the PU setting.
    self.assertAlmostEqual(auc, 0.8870, delta=0.02)


if __name__ == '__main__':
  tf.test.main()
