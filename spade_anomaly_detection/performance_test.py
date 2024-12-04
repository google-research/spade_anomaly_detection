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

from absl.testing import parameterized
from spade_anomaly_detection import csv_data_loader
from spade_anomaly_detection import data_loader
from spade_anomaly_detection import parameters
from spade_anomaly_detection import runner
from spade_anomaly_detection import supervised_model
import tensorflow as tf


class PerformanceTestOnBQData(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    # Using the 10k row covertype dataset here to speed up testing time. The
    # 100k version was timing out on blaze.
    self.orig_record_count = 10000
    self.csv_pnu_path = 'covertype_pnu_train'
    self.csv_pu_path = 'covertype_pu_train'
    self.csv_test_path = 'covertype_test'

    self.runner_parameters = parameters.RunnerParameters(
        train_setting='PNU',
        input_bigquery_table_path='project_id.dataset.table_name',
        data_input_gcs_uri=None,
        output_gcs_uri='gs://test_bucket/test_folder',
        label_col_name='y',
        positive_data_value=1,
        negative_data_value=0,
        unlabeled_data_value=-1,
        labels_are_strings=False,
        positive_threshold=10,
        negative_threshold=90,
        test_label_col_name='y',
        test_dataset_holdout_fraction=None,
        test_bigquery_table_path='project_id.dataset.test_table_name',
        alpha=1.0,
        batches_per_model=1,
        max_occ_batch_size=50000,
        labeling_and_model_training_batch_size=self.orig_record_count,
        ensemble_count=5,
        upload_only=False,
        verbose=False,
    )

    self.unlabeled_features, self.unlabeled_labels = data_loader.load_dataframe(
        dataset_name=self.csv_pnu_path,
        filter_label_value=self.runner_parameters.unlabeled_data_value,
        index_col=None,
        skiprows=0,
    )
    self.unlabeled_record_count = len(self.unlabeled_labels)
    _, positive_labels = data_loader.load_dataframe(
        dataset_name=self.csv_pnu_path,
        filter_label_value=self.runner_parameters.positive_data_value,
        index_col=None,
        skiprows=0,
    )
    self.positive_record_count = len(positive_labels)
    _, negative_labels = data_loader.load_dataframe(
        dataset_name=self.csv_pnu_path,
        filter_label_value=self.runner_parameters.negative_data_value,
        index_col=None,
        skiprows=0,
    )
    self.negative_record_count = len(negative_labels)
    _, test_labels = data_loader.load_dataframe(
        dataset_name=self.csv_test_path,
        filter_label_value=None,
        index_col=None,
        skiprows=0,
    )
    self.test_record_count = len(test_labels)

    self.pnu_label_and_train_batch_size = (
        self.unlabeled_record_count
        + self.negative_record_count
        + self.positive_record_count
    )
    self.pu_label_and_train_batch_size = (
        self.unlabeled_record_count + self.positive_record_count
    )

    self.occ_fit_batch_size = (
        (self.unlabeled_record_count + self.negative_record_count)
        // self.runner_parameters.ensemble_count
    ) // self.runner_parameters.batches_per_model

    self.nu_complete_tensorflow_dataset = data_loader.load_tf_dataset_from_csv(
        dataset_name=self.csv_pnu_path,
        batch_size=self.occ_fit_batch_size,
        filter_label_value=[
            self.runner_parameters.unlabeled_data_value,
            self.runner_parameters.negative_data_value,
        ],
        index_col=None,
        skiprows=0,
    )
    self.pnu_complete_tensorflow_dataset = data_loader.load_tf_dataset_from_csv(
        dataset_name=self.csv_pnu_path,
        batch_size=self.pnu_label_and_train_batch_size,
        index_col=None,
        skiprows=0,
    )
    self.pu_complete_tensorflow_dataset = data_loader.load_tf_dataset_from_csv(
        dataset_name=self.csv_pu_path,
        batch_size=self.pu_label_and_train_batch_size,
        index_col=None,
        skiprows=0,
    )
    self.test_tensorflow_dataset = data_loader.load_tf_dataset_from_csv(
        dataset_name=self.csv_test_path,
        batch_size=self.test_record_count,
        index_col=None,
        skiprows=0,
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
    self.runner_parameters.labeling_and_model_training_batch_size = (
        self.pnu_label_and_train_batch_size
    )

    self.mock_data_loader.side_effect = [
        self.nu_complete_tensorflow_dataset,  # For OCC fit
        self.pnu_complete_tensorflow_dataset,  # Pseudo-labeling
        self.test_tensorflow_dataset,  # Test
    ]
    # There are 4 calls to get_query_record_result_length.
    self.mock_get_total_records.side_effect = [
        self.orig_record_count,
        self.unlabeled_record_count,
        self.negative_record_count,
        self.test_record_count,
    ]

    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    with self.subTest('check_call_counts'):
      self.assertEqual(self.mock_data_loader.call_count, 3)
      self.assertEqual(self.mock_get_total_records.call_count, 4)

    auc = runner_object.supervised_model_metrics['Supervised_Model_AUC']
    # See SPADE performance for Covertype PNU Setting in the design document for
    # setting and adjusting the AUC here. 0.9251 roughly equates to the
    # performance seen on the ~580k row Covertype dataset in the PNU setting.
    self.assertAlmostEqual(auc, 0.9251, delta=0.02)

  def test_spade_auc_performance_pu_single_batch(self):
    self.runner_parameters.train_setting = parameters.TrainSetting.PU
    self.runner_parameters.positive_threshold = 10
    self.runner_parameters.negative_threshold = 50
    self.runner_parameters.labeling_and_model_training_batch_size = (
        self.pu_label_and_train_batch_size
    )

    self.mock_data_loader.side_effect = [
        self.nu_complete_tensorflow_dataset,  # For OCC fit
        self.pu_complete_tensorflow_dataset,  # Pseudo-labeling
        self.test_tensorflow_dataset,  # Test
    ]
    # There are 4 calls to get_query_record_result_length.
    self.mock_get_total_records.side_effect = [
        self.orig_record_count,
        self.unlabeled_record_count,
        self.negative_record_count,
        self.test_record_count,
    ]

    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    auc = runner_object.supervised_model_metrics['Supervised_Model_AUC']
    # See SPADE performance for Covertype PU Setting in the design document for
    # setting and adjusting the AUC here. 0.8870 represents the performance seen
    # on the ~580k row Covertype dataset in the PU setting.
    self.assertAlmostEqual(auc, 0.9178, delta=0.02)


class PerformanceTestOnCSVData(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # Using the 10k row covertype dataset here to speed up testing time. The
    # 100k version was timing out on blaze.
    orig_record_count = 10000
    self.label_and_train_batch_size = orig_record_count // 2
    self.csv_pnu_path = 'covertype_pnu_train'
    self.csv_pu_path = 'covertype_pu_train'
    self.csv_test_path = 'covertype_test'

    self.runner_parameters = parameters.RunnerParameters(
        train_setting='PNU',
        input_bigquery_table_path='',
        data_input_gcs_uri='gs://some_bucket/input_folder',
        output_gcs_uri='gs://test_bucket/test_folder',
        label_col_name='y',
        positive_data_value=1,
        negative_data_value=0,
        unlabeled_data_value=-1,
        labels_are_strings=False,
        positive_threshold=10,
        negative_threshold=90,
        test_label_col_name='y',
        test_dataset_holdout_fraction=None,
        data_test_gcs_uri='gs://some_bucket/test_folder',
        upload_only=False,
        output_bigquery_table_path='',
        data_output_gcs_uri=None,
        alpha=1.0,
        batches_per_model=1,
        max_occ_batch_size=50000,
        labeling_and_model_training_batch_size=self.label_and_train_batch_size,
        ensemble_count=5,
        verbose=True,
    )

    _, self.unlabeled_labels = data_loader.load_dataframe(
        dataset_name=self.csv_pnu_path,
        filter_label_value=self.runner_parameters.unlabeled_data_value,
        index_col=None,
        skiprows=0,
    )
    self.unlabeled_record_count = len(self.unlabeled_labels)
    _, positive_labels = data_loader.load_dataframe(
        dataset_name=self.csv_pnu_path,
        filter_label_value=self.runner_parameters.positive_data_value,
        index_col=None,
        skiprows=0,
    )
    self.positive_record_count = len(positive_labels)
    _, negative_labels = data_loader.load_dataframe(
        dataset_name=self.csv_pnu_path,
        filter_label_value=self.runner_parameters.negative_data_value,
        index_col=None,
        skiprows=0,
    )
    self.negative_record_count = len(negative_labels)
    _, test_labels = data_loader.load_dataframe(
        dataset_name=self.csv_test_path,
        filter_label_value=None,
        index_col=None,
        skiprows=0,
    )
    self.test_positive_record_count = len(
        test_labels[test_labels == self.runner_parameters.positive_data_value]
    )
    self.test_negative_record_count = len(
        test_labels[test_labels == self.runner_parameters.negative_data_value]
    )
    self.test_record_count = len(test_labels)

    self.pnu_label_and_train_batch_size = (
        self.unlabeled_record_count
        + self.negative_record_count
        + self.positive_record_count
    )
    self.pu_label_and_train_batch_size = (
        self.unlabeled_record_count + self.positive_record_count
    )

    self.train_label_counts = {
        self.runner_parameters.positive_data_value: self.positive_record_count,
        self.runner_parameters.negative_data_value: self.negative_record_count,
        self.runner_parameters.unlabeled_data_value: (
            self.unlabeled_record_count
        ),
    }
    self.test_label_counts = {
        self.runner_parameters.positive_data_value: (
            self.test_positive_record_count
        ),
        self.runner_parameters.negative_data_value: (
            self.test_negative_record_count
        ),
        self.runner_parameters.unlabeled_data_value: 0,
    }

    self.occ_fit_batch_size = (
        (self.unlabeled_record_count + self.negative_record_count)
        // self.runner_parameters.ensemble_count
    ) // self.runner_parameters.batches_per_model

    self.nu_complete_tensorflow_dataset = data_loader.load_tf_dataset_from_csv(
        dataset_name=self.csv_pu_path,
        batch_size=self.occ_fit_batch_size,
        filter_label_value=[
            self.runner_parameters.unlabeled_data_value,
            self.runner_parameters.negative_data_value,
        ],
        index_col=None,
        skiprows=0,
    )

    self.pnu_complete_tensorflow_dataset = data_loader.load_tf_dataset_from_csv(
        dataset_name=self.csv_pnu_path,
        batch_size=self.pnu_label_and_train_batch_size,
        index_col=None,
        skiprows=0,
    )
    self.pu_complete_tensorflow_dataset = data_loader.load_tf_dataset_from_csv(
        dataset_name=self.csv_pu_path,
        batch_size=self.pu_label_and_train_batch_size,
        index_col=None,
        skiprows=0,
    )
    self.test_tensorflow_dataset = data_loader.load_tf_dataset_from_csv(
        dataset_name=self.csv_test_path,
        batch_size=None,  # CSV test dataset is batched in the runner.
        index_col=None,
        skiprows=0,
    )

    self.mock_label_counts = self.enter_context(
        mock.patch.object(
            csv_data_loader.CsvDataLoader,
            'label_counts',
            new_callable=mock.PropertyMock,
        )
    )

    self.mock_load_tf_dataset_from_csv = self.enter_context(
        mock.patch.object(
            csv_data_loader.CsvDataLoader,
            'load_tf_dataset_from_csv',
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

  @parameterized.named_parameters([
      ('labels_are_ints', False, 1, 0, -1),
      ('labels_are_strings', True, '1', '0', '-1'),
  ])
  def test_spade_auc_performance_pnu_single_batch(
      self,
      labels_are_strings: bool,
      positive_data_value: str | int,
      negative_data_value: str | int,
      unlabeled_data_value: str | int,
  ):
    self.runner_parameters.train_setting = parameters.TrainSetting.PNU
    self.runner_parameters.labels_are_strings = labels_are_strings
    self.runner_parameters.positive_data_value = positive_data_value
    self.runner_parameters.negative_data_value = negative_data_value
    self.runner_parameters.unlabeled_data_value = unlabeled_data_value
    self.runner_parameters.positive_threshold = 0.1
    self.runner_parameters.negative_threshold = 95
    self.runner_parameters.alpha = 0.1
    self.runner_parameters.ensemble_count = 5
    self.runner_parameters.labeling_and_model_training_batch_size = (
        self.pnu_label_and_train_batch_size
    )

    self.mock_load_tf_dataset_from_csv.side_effect = [
        self.pnu_complete_tensorflow_dataset,  # For label counts
        self.nu_complete_tensorflow_dataset,  # For OCC fit
        self.pnu_complete_tensorflow_dataset,  # Pseudo-labeling
        self.test_tensorflow_dataset,  # Test
    ]
    self.mock_label_counts.side_effect = [  # There are 6 calls to label_counts.
        self.train_label_counts,
        self.train_label_counts,
        self.train_label_counts,
        self.train_label_counts,
        self.test_label_counts,
        self.test_label_counts,
    ]

    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    auc = runner_object.supervised_model_metrics['Supervised_Model_AUC']
    # See SPADE performance for Covertype PNU Setting in the design document for
    # setting and adjusting the AUC here. 0.9251 roughly equates to the
    # performance seen on the ~580k row Coertype dataset in the PNU setting.
    self.assertAlmostEqual(auc, 0.9755, delta=0.02)

  @parameterized.named_parameters([
      ('labels_are_ints', False, 1, 0, -1),
      ('labels_are_strings', True, '1', '0', '-1'),
  ])
  def test_spade_auc_performance_pu_single_batch(
      self,
      labels_are_strings: bool,
      positive_data_value: str | int,
      negative_data_value: str | int,
      unlabeled_data_value: str | int,
  ):
    self.runner_parameters.train_setting = parameters.TrainSetting.PU
    self.runner_parameters.labels_are_strings = labels_are_strings
    self.runner_parameters.positive_data_value = positive_data_value
    self.runner_parameters.negative_data_value = negative_data_value
    self.runner_parameters.unlabeled_data_value = unlabeled_data_value
    self.runner_parameters.positive_threshold = 10
    self.runner_parameters.negative_threshold = 50
    self.runner_parameters.labeling_and_model_training_batch_size = (
        self.pu_label_and_train_batch_size
    )

    self.mock_load_tf_dataset_from_csv.side_effect = [
        self.pu_complete_tensorflow_dataset,  # For label counts
        self.nu_complete_tensorflow_dataset,  # For OCC fit
        self.pu_complete_tensorflow_dataset,  # Pseudo-labeling
        self.test_tensorflow_dataset,  # Test
    ]
    self.mock_label_counts.side_effect = [
        self.train_label_counts,
        self.train_label_counts,
        self.train_label_counts,
        self.train_label_counts,
        self.test_label_counts,
        self.test_label_counts,
    ]

    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    auc = runner_object.supervised_model_metrics['Supervised_Model_AUC']
    # See SPADE performance for Covertype PU Setting in the design document for
    # setting and adjusting the AUC here. 0.8870 represents the performance seen
    # on the ~580k row Covertype dataset in the PU setting.
    self.assertAlmostEqual(auc, 0.8837, delta=0.02)


if __name__ == '__main__':
  tf.test.main()
