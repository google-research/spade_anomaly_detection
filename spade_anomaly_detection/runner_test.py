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

"""Tests for runner."""

import re
from typing import Optional, Pattern, Sequence, Union
from unittest import mock

import numpy as np
from parameterized import parameterized

from spade_anomaly_detection import data_loader
from spade_anomaly_detection import occ_ensemble
from spade_anomaly_detection import parameters
from spade_anomaly_detection import runner
from spade_anomaly_detection import supervised_model
import tensorflow as tf
import tensorflow_decision_forests as tfdf


class RunnerTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    self.runner_parameters = parameters.RunnerParameters(
        train_setting='PNU',
        input_bigquery_table_path='project.dataset.table',
        output_gcs_uri='gs://test_bucket/test_folder',
        label_col_name='label',
        positive_data_value=5,
        negative_data_value=3,
        unlabeled_data_value=-100,
        positive_threshold=5,
        negative_threshold=95,
        ignore_columns=None,
        where_statements=None,
        test_bigquery_table_path='',
        test_label_col_name='',
        test_dataset_holdout_fraction=0.3,
        upload_only=False,
        output_bigquery_table_path='',
        alpha=1.0,
        batches_per_model=1,
        max_occ_batch_size=50000,
        labeling_and_model_training_batch_size=None,
        ensemble_count=5,
        verbose=False,
    )

    self.mock_get_query_record_result_length = self.enter_context(
        mock.patch.object(
            data_loader.DataLoader,
            'get_query_record_result_length',
            autospec=True,
        )
    )

    self.mock_load_tf_dataset_from_bigquery = self.enter_context(
        mock.patch.object(
            data_loader.DataLoader,
            'load_tf_dataset_from_bigquery',
            autospec=True,
        )
    )
    self.mock_supervised_model_train = self.enter_context(
        mock.patch.object(
            supervised_model.RandomForestModel,
            'train',
            autospec=True,
        )
    )
    self.mock_supervised_model_save = self.enter_context(
        mock.patch.object(
            supervised_model.RandomForestModel,
            'save',
            autospec=True,
        )
    )
    self.mock_supervised_model_evaluate = self.enter_context(
        mock.patch.object(
            tfdf.keras.RandomForestModel,
            'evaluate',
            autospec=True,
        )
    )
    self.mock_bigquery_upload = self.enter_context(
        mock.patch.object(
            data_loader.DataLoader,
            'upload_dataframe_as_bigquery_table',
            autospec=True,
        )
    )

    self._create_mock_datasets()

  def _create_mock_datasets(self) -> None:
    num_features = 2
    self.per_class_labeled_example_count = 10
    self.unlabeled_examples = 200
    self.all_examples = (
        self.per_class_labeled_example_count * 2
    ) + self.unlabeled_examples
    self.total_test_records = self.per_class_labeled_example_count * 2
    self.negative_examples = self.per_class_labeled_example_count * 1

    unlabeled_features = np.random.rand(
        self.unlabeled_examples, num_features
    ).astype(np.float32)
    unlabeled_labels = np.repeat(
        self.runner_parameters.unlabeled_data_value, self.unlabeled_examples
    )

    all_features = np.random.rand(self.all_examples, num_features).astype(
        np.float32
    )

    all_labels = np.concatenate(
        [
            np.repeat(
                self.runner_parameters.positive_data_value,
                self.per_class_labeled_example_count,
            ),
            np.repeat(
                self.runner_parameters.negative_data_value,
                self.per_class_labeled_example_count,
            ),
            unlabeled_labels,
        ],
        axis=0,
    ).astype(np.int8)

    if self.runner_parameters.labeling_and_model_training_batch_size is None:
      batch_size_all_data = self.all_examples
    else:
      batch_size_all_data = (
          self.runner_parameters.labeling_and_model_training_batch_size
      )

    self.all_data_tf_dataset = tf.data.Dataset.from_tensor_slices(
        (all_features, all_labels)
    ).batch(batch_size_all_data, drop_remainder=True)

    records_per_occ = (
        self.unlabeled_examples // self.runner_parameters.ensemble_count
    )
    unlabeled_batch_size = (
        records_per_occ // self.runner_parameters.batches_per_model
    )
    self.unlabeled_data_tf_dataset = tf.data.Dataset.from_tensor_slices(
        (unlabeled_features, unlabeled_labels)
    ).batch(unlabeled_batch_size, drop_remainder=True)

    if self.runner_parameters.test_bigquery_table_path:
      self.test_labels = np.repeat(
          [
              self.runner_parameters.positive_data_value,
              self.runner_parameters.negative_data_value,
          ],
          [
              self.per_class_labeled_example_count,
              self.per_class_labeled_example_count,
          ],
      )
      self.test_features = np.random.rand(
          self.total_test_records, num_features
      ).astype(np.float32)
      self.test_tf_dataset = tf.data.Dataset.from_tensor_slices(
          (self.test_features, self.test_labels)
      ).batch(self.total_test_records, drop_remainder=True)

      self.mock_load_tf_dataset_from_bigquery.side_effect = [
          self.unlabeled_data_tf_dataset,
          self.all_data_tf_dataset,
          self.test_tf_dataset,
      ]
      self.mock_get_query_record_result_length.side_effect = [
          self.all_examples,
          self.unlabeled_examples,
          self.negative_examples,
          self.total_test_records,
      ]
    else:
      self.mock_load_tf_dataset_from_bigquery.side_effect = [
          self.unlabeled_data_tf_dataset,
          self.all_data_tf_dataset,
      ]
      self.mock_get_query_record_result_length.side_effect = [
          self.all_examples,
          self.unlabeled_examples,
          self.negative_examples,
      ]

  def test_runner_data_loader_no_error(self):
    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    # Check the batch size here to ensure we are dividing the dataset into 5
    # shards.
    with self.subTest(name='OccDataset'):
      self.mock_load_tf_dataset_from_bigquery.assert_any_call(
          mock.ANY,
          input_path=self.runner_parameters.input_bigquery_table_path,
          label_col_name=self.runner_parameters.label_col_name,
          where_statements=self.runner_parameters.where_statements,
          ignore_columns=self.runner_parameters.ignore_columns,
          # Verify that both negative and unlabeled samples are used.
          label_column_filter_value=[
              self.runner_parameters.unlabeled_data_value,
              self.runner_parameters.negative_data_value,
          ],
          # Verify that batch size is computed with both negative and unlabeled
          # sample counts.
          batch_size=(self.unlabeled_examples + self.negative_examples)
          // self.runner_parameters.ensemble_count,
      )
    # Assert that the data loader is also called to fetch all records.
    with self.subTest(name='InferenceAndSupervisedDataset'):
      self.mock_load_tf_dataset_from_bigquery.assert_any_call(
          mock.ANY,
          input_path=self.runner_parameters.input_bigquery_table_path,
          label_col_name=self.runner_parameters.label_col_name,
          where_statements=self.runner_parameters.where_statements,
          ignore_columns=self.runner_parameters.ignore_columns,
          batch_size=self.all_examples,
      )

  def test_runner_supervised_model_fit(self):
    self.runner_parameters.alpha = 0.8
    self.runner_parameters.negative_threshold = 0

    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    supervised_model_actual_kwargs = (
        self.mock_supervised_model_train.call_args.kwargs
    )

    with self.subTest('NoUnlabeledData'):
      self.assertNotIn(
          self.runner_parameters.unlabeled_data_value,
          supervised_model_actual_kwargs['labels'],
          msg='Unlabeled data was used to fit the supervised model.',
      )
    with self.subTest('LabelWeights'):
      self.assertIn(
          self.runner_parameters.alpha,
          supervised_model_actual_kwargs['weights'],
      )

  def test_runner_get_record_count_with_where_statement_no_error(self):
    self.runner_parameters.where_statements = ['a = 1', 'b = 2']
    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    with self.subTest('AllRecordsQueryWithWhereStatements'):
      self.mock_get_query_record_result_length.assert_any_call(
          mock.ANY,
          input_path=self.runner_parameters.input_bigquery_table_path,
          where_statements=self.runner_parameters.where_statements,
      )
    with self.subTest('UnlabeledRecordsQueryWithWhereStatements'):
      self.mock_get_query_record_result_length.assert_any_call(
          mock.ANY,
          input_path=self.runner_parameters.input_bigquery_table_path,
          where_statements=[self.runner_parameters.where_statements]
          + [
              f'{self.runner_parameters.label_col_name} ='
              f' {self.runner_parameters.unlabeled_data_value}'
          ],
      )

  def test_runner_get_record_count_without_where_statement_no_error(self):
    self.runner_parameters.where_statements = None
    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    with self.subTest('AllRecordsQueryWithoutWhereStatements'):
      self.mock_get_query_record_result_length.assert_any_call(
          mock.ANY,
          input_path=self.runner_parameters.input_bigquery_table_path,
          where_statements=self.runner_parameters.where_statements,
      )
    with self.subTest('UnlabeledRecordsQueryWithoutWhereStatements'):
      self.mock_get_query_record_result_length.assert_any_call(
          mock.ANY,
          input_path=self.runner_parameters.input_bigquery_table_path,
          where_statements=[
              f'{self.runner_parameters.label_col_name} ='
              f' {self.runner_parameters.unlabeled_data_value}'
          ],
      )

  def test_runner_record_count_raise_error(self):
    self.runner_parameters.ensemble_count = 10
    self.mock_get_query_record_result_length.side_effect = [5, 0, 1]
    runner_object = runner.Runner(self.runner_parameters)

    with self.assertRaisesRegex(
        ValueError, r'There are not enough records in the table to fit one'
    ):
      runner_object.run()

  def test_runner_no_records_raise_error(self):
    self.mock_get_query_record_result_length.side_effect = [0, 0, 0]
    runner_object = runner.Runner(self.runner_parameters)

    with self.assertRaisesRegex(
        ValueError, r'There are no records in the table:'
    ):
      runner_object.run()

  def _assert_regex_in(
      self,
      seq: Sequence[str],
      regex: Union[str, Pattern[str]],
      msg: Optional[str] = None,
  ):
    if not msg:
      msg = f'{regex} not found in {seq}'
    self.assertTrue(any([re.search(regex, s) for s in seq]), msg=msg)

  def test_record_count_warning_raise(self):
    # Will raise a warning when there are < 1k samples in the entire dataset.
    self.mock_get_query_record_result_length.side_effect = [500, 100, 10]
    runner_object = runner.Runner(self.runner_parameters)

    with self.assertLogs() as training_logs:
      runner_object.run()

    self._assert_regex_in(
        training_logs.output,
        r'Using a small number of examples to train the model,',
    )

  def test_verbose_logging_no_error(self):
    example_features = np.random.rand(100, 10).astype(np.float32)
    example_labels_pos = np.repeat(1, 10)
    example_labels_neg = np.repeat(0, 90)
    example_labels = np.concatenate(
        [example_labels_pos, example_labels_neg], axis=0
    )
    example_weights = np.repeat(
        self.runner_parameters.alpha, len(example_labels)
    )

    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    with self.assertLogs() as training_logs:
      runner_object.write_verbose_logs(
          features=example_features,
          labels=example_labels,
          weights=example_weights,
      )
    with self.subTest(name='LabelCountLogs'):
      self._assert_regex_in(
          training_logs.output, r'Updated label counts 0    90\n1    10\n'
      )
    with self.subTest(name='TrainFeatureShapeLogs'):
      self._assert_regex_in(
          training_logs.output, r'Updated features shape: \(100, 10\)'
      )
      self._assert_regex_in(
          training_logs.output, r'Updated labels shape: \(100,\)'
      )
    with self.subTest(name='WeightShapeLogs'):
      self._assert_regex_in(training_logs.output, r'Weights shape: \(100,\)')

  def test_supervised_model_evaluation_no_error(self):
    runner_obj = runner.Runner(self.runner_parameters)
    runner_obj.run()

    evaluate_arguments = self.mock_supervised_model_evaluate.call_args.kwargs

    with self.subTest(name='TestLabels'):
      self.assertNotIn(
          self.runner_parameters.unlabeled_data_value, evaluate_arguments['y']
      )
      self.assertIn(1, evaluate_arguments['y'])
      self.assertIn(0, evaluate_arguments['y'])
    with self.subTest(name='FeaturesNotNull'):
      self.assertIsNotNone(evaluate_arguments['x'])

  def test_proprocessing_inputs_supervised_model_train(self):
    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    mmock_train_feature_value = (
        self.mock_supervised_model_train.call_args.kwargs['labels']
    )

    self.assertNotIn(
        self.runner_parameters.unlabeled_data_value, mmock_train_feature_value
    )

  @mock.patch.object(occ_ensemble.GmmEnsemble, 'pseudo_label', autospec=True)
  @mock.patch.object(
      runner.Runner, 'preprocess_train_test_split', autospec=True
  )
  def test_batch_sizing_no_error(self, mock_split, mock_pseudo_label):
    self.runner_parameters.labeling_and_model_training_batch_size = 1
    # Update mock datasets with the new batch parameter
    self._create_mock_datasets()
    mock_split.return_value = ([], [])
    mock_pseudo_label.return_value = (
        np.empty((1, 1)),
        np.empty((1, 1)),
        np.empty((1, 1)),
    )
    runner_object = runner.Runner(self.runner_parameters)

    # Manually create train/test splits since we mock out
    # preprocess_train_test_split
    runner_object.test_x = np.random.rand(100, 10).astype(np.float32)
    runner_object.test_y = np.random.rand(10, 1).astype(np.int8)
    runner_object.run()

    label_batch_shape = mock_split.call_args.kwargs['labels'].shape[0]
    feature_batch_shape = mock_split.call_args.kwargs['features'].shape[0]

    with self.subTest(name='FeatureBatchSize'):
      self.assertEqual(
          feature_batch_shape,
          self.runner_parameters.labeling_and_model_training_batch_size,
      )
    with self.subTest(name='LabelBatchSize'):
      self.assertEqual(
          label_batch_shape,
          self.runner_parameters.labeling_and_model_training_batch_size,
      )
    with self.subTest(name='CallsToSplit'):
      self.assertEqual(mock_split.call_count, self.all_examples)

  def test_batch_size_too_large_throw_error(self):
    self.runner_parameters.labeling_and_model_training_batch_size = 1000
    self.mock_get_query_record_result_length.side_effect = [100, 5, 10]
    runner_object = runner.Runner(self.runner_parameters)

    with self.assertRaisesRegex(
        ValueError,
        r'labeling_and_model_training_batch_size can not be '
        r'greater than the total number of examples. There are',
    ):
      runner_object.run()

  @mock.patch.object(occ_ensemble.GmmEnsemble, 'pseudo_label', autospec=True)
  def test_preprocessing_pu_no_error(self, mock_pseudo_label):
    mock_pseudo_label.return_value = (
        np.empty((1, 1)),
        np.empty((1, 1)),
        np.empty((1, 1)),
    )
    self.runner_parameters.verbose = True
    self.runner_parameters.test_dataset_holdout_fraction = 0.2
    self.runner_parameters.train_setting = parameters.TrainSetting.PU

    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    y_train_value = mock_pseudo_label.call_args.kwargs['labels']

    # Train sets
    with self.subTest(name='NoNegativeInTrainingSet'):
      self.assertNotIn(
          self.runner_parameters.negative_data_value, y_train_value
      )
    with self.subTest(name='PositiveInTrainingSet'):
      self.assertIn(self.runner_parameters.positive_data_value, y_train_value)
    with self.subTest(name='UnlabeledInTrainingSet'):
      self.assertIn(self.runner_parameters.unlabeled_data_value, y_train_value)

    # Test sets, we check against 1 and 0 here since labels are normalized
    # for the testing sets.
    with self.subTest(name='PositiveInTestSet'):
      self.assertIn(1, runner_object.test_y)
    with self.subTest(name='NegativeInTestSet'):
      self.assertIn(0, runner_object.test_y)
    with self.subTest(name='UnlabeledNotInTestSet'):
      self.assertNotIn(
          self.runner_parameters.unlabeled_data_value, runner_object.test_y
      )

  @mock.patch.object(occ_ensemble.GmmEnsemble, 'pseudo_label', autospec=True)
  def test_preprocessing_pnu_no_error(self, mock_pseudo_label):
    mock_pseudo_label.return_value = (
        np.empty((1, 1)),
        np.empty((1, 1)),
        np.empty((1, 1)),
    )
    self.runner_parameters.verbose = True
    self.runner_parameters.test_dataset_holdout_fraction = 0.2
    self.runner_parameters.train_setting = parameters.TrainSetting.PNU

    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    pseudo_label_y_input = mock_pseudo_label.call_args.kwargs['labels']

    # Test sets
    with self.subTest(name='NegativeInTrainingSet'):
      self.assertIn(
          self.runner_parameters.negative_data_value, pseudo_label_y_input
      )
    with self.subTest(name='PositiveInTrainingSet'):
      self.assertIn(
          self.runner_parameters.positive_data_value, pseudo_label_y_input
      )
    with self.subTest(name='UnlabeledInTrainingSet'):
      self.assertIn(
          self.runner_parameters.unlabeled_data_value, pseudo_label_y_input
      )

    # Test sets, we check against 1 and 0 here since labels are normalized
    # for the testing sets.
    with self.subTest(name='PositiveInTestSet'):
      self.assertIn(1, runner_object.test_y)
    with self.subTest(name='NegativeInTestSet'):
      self.assertIn(0, runner_object.test_y)
    with self.subTest(name='UnlabeledNotInTestSet'):
      self.assertNotIn(
          self.runner_parameters.unlabeled_data_value, runner_object.test_y
      )

  @parameterized.expand(
      [parameters.TrainSetting.PNU, parameters.TrainSetting.PU]
  )
  @mock.patch.object(occ_ensemble.GmmEnsemble, 'pseudo_label', autospec=True)
  def test_preprocessing_array_sizes_no_error(
      self,
      train_setting,
      mock_pseudo_label,
  ):
    mock_pseudo_label.return_value = (
        np.empty((1, 1)),
        np.empty((1, 1)),
        np.empty((1, 1)),
    )
    self.runner_parameters.verbose = True
    self.runner_parameters.train_setting = train_setting

    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    x_train_length = len(mock_pseudo_label.call_args.kwargs['features'])
    y_train_length = len(mock_pseudo_label.call_args.kwargs['labels'])

    x_test_length = len(runner_object.test_x)
    y_test_length = len(runner_object.test_y)

    with self.subTest(name='TrainAndTestSetLengths'):
      self.assertEqual(x_train_length, y_train_length)
      self.assertEqual(x_test_length, y_test_length)
    with self.subTest(name='ReturnSizeGivenSize'):
      all_data_preprocessed_length = x_train_length + x_test_length
      self.assertEqual(all_data_preprocessed_length, self.all_examples)

  def test_preprocessing_array_size_imbalance_throws_error(self):
    features = np.random.rand(100, 10).astype(np.float32)
    labels = np.random.rand(10, 1).astype(np.int8)
    runner_object = runner.Runner(self.runner_parameters)

    with self.assertRaisesRegex(
        ValueError, r'Feature and label arrays have different lengths.'
    ):
      runner_object.preprocess_train_test_split(features, labels)

  def test_preprocessing_unknown_train_setting_throw_error(self):
    self.runner_parameters.train_setting = 'not a setting'
    features = np.random.rand(100, 10).astype(np.float32)
    labels = np.random.rand(100, 1).astype(np.int8)
    runner_object = runner.Runner(self.runner_parameters)

    with self.assertRaisesRegex(
        ValueError, r'Unknown train setting for preparing train/test'
    ):
      runner_object.preprocess_train_test_split(features, labels)

  def test_bigquery_test_set_contents_no_error(self):
    self.runner_parameters.test_bigquery_table_path = 'test_proj.data.test_set'
    self.runner_parameters.test_label_col_name = 'label'
    self.runner_parameters.verbose = True
    self._create_mock_datasets()

    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    test_labels_preprocessed = self.test_labels
    test_labels_preprocessed[
        test_labels_preprocessed == self.runner_parameters.positive_data_value
    ] = 1
    test_labels_preprocessed[
        test_labels_preprocessed == self.runner_parameters.negative_data_value
    ] = 0

    with self.subTest('TestSetEqualToBigQuery'):
      self.assertEqual(runner_object.test_x.shape, self.test_features.shape)
      self.assertAllEqual(
          runner_object.test_y.tolist(), test_labels_preprocessed.tolist()
      )

    with self.subTest('NoUnlabeledDataInTestSet'):
      self.assertNotIn(
          self.runner_parameters.unlabeled_data_value,
          runner_object.test_y,
          msg='Unlabeled data was used to fit the supervised model.',
      )

  def test_bigquery_testing_set_throws_warning(self):
    self.runner_parameters.test_bigquery_table_path = 'test_proj.data.test_set'
    self.runner_parameters.test_label_col_name = 'label'
    self.runner_parameters.test_dataset_holdout_fraction = 0.5
    self.runner_parameters.verbose = True

    self._create_mock_datasets()

    runner_object = runner.Runner(self.runner_parameters)
    with self.assertLogs() as training_logs:
      runner_object.run()

    self._assert_regex_in(
        training_logs.output,
        r'Only a test holdout fraction and a single input table',
    )

  def test_dataset_creation_function_calls_no_error(self):
    self.runner_parameters.test_bigquery_table_path = 'test_proj.data.test_set'
    self.runner_parameters.test_label_col_name = 'label_test_col'
    self._create_mock_datasets()

    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    with self.subTest('TestRecordCount'):
      self.mock_get_query_record_result_length.assert_any_call(
          mock.ANY,
          input_path=self.runner_parameters.test_bigquery_table_path,
          where_statements=[
              f'{self.runner_parameters.test_label_col_name} !='
              f' {self.runner_parameters.unlabeled_data_value}'
          ],
      )
    with self.subTest('TestTableDataLoader'):
      self.mock_load_tf_dataset_from_bigquery.assert_any_call(
          mock.ANY,
          input_path=self.runner_parameters.test_bigquery_table_path,
          label_col_name=self.runner_parameters.test_label_col_name,
          where_statements=[
              f'{self.runner_parameters.test_label_col_name} !='
              f' {self.runner_parameters.unlabeled_data_value}'
          ],
          ignore_columns=self.runner_parameters.ignore_columns,
          batch_size=self.total_test_records,
      )

  def test_dataset_label_values_positive_and_negative_throws_error(self):
    self.runner_parameters.test_bigquery_table_path = 'test_proj.data.test_set'
    self.runner_parameters.test_label_col_name = 'label_test_col'
    self.runner_parameters.verbose = True
    total_test_records = 5

    # Create a new test set with just one label type.
    test_features = np.random.rand(total_test_records, 5).astype(np.float32)
    test_labels = np.repeat(
        self.runner_parameters.positive_data_value, total_test_records
    ).astype(np.int8)

    test_tf_dataset = tf.data.Dataset.from_tensor_slices(
        (test_features, test_labels)
    ).batch(total_test_records, drop_remainder=True)

    self.mock_load_tf_dataset_from_bigquery.side_effect = [
        self.unlabeled_data_tf_dataset,
        self.all_data_tf_dataset,
        test_tf_dataset,
    ]
    self.mock_get_query_record_result_length.side_effect = [
        self.all_examples,
        self.unlabeled_examples,
        self.negative_examples,
        total_test_records,
    ]

    runner_object = runner.Runner(self.runner_parameters)

    with self.assertRaisesRegex(
        ValueError, r'Positive and negative labels must be in the testing set.'
    ):
      runner_object.run()

  def test_upload_only_setting_true_no_error(self):
    self.runner_parameters.upload_only = True
    self.runner_parameters.output_bigquery_table_path = (
        'project.dataset.pseudo_labeled_data'
    )

    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    with self.subTest('SupervisedModelNotCalled'):
      self.mock_supervised_model_train.assert_not_called()
      self.mock_supervised_model_evaluate.assert_not_called()
      self.mock_supervised_model_save.assert_not_called()

    with self.subTest('BigQueryUploadCalled'):
      self.mock_bigquery_upload.assert_called_once()

  def test_upload_only_setting_true_throw_error_no_table(self):
    self.runner_parameters.upload_only = True
    self.runner_parameters.output_bigquery_table_path = ''
    runner_object = runner.Runner(self.runner_parameters)

    with self.assertRaisesRegex(
        ValueError, r'output_bigquery_table_path needs to be specified in'
    ):
      runner_object.run()

  def test_upload_only_false_no_error(self):
    self.runner_parameters.upload_only = False

    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    with self.subTest('SupervisedModelCalled'):
      self.mock_supervised_model_train.assert_called_once()
      self.mock_supervised_model_evaluate.assert_called_once()
      self.mock_supervised_model_save.assert_called_once()

    with self.subTest('BigQueryUploadNotCalled'):
      self.mock_bigquery_upload.assert_not_called()

  def test_evaluate_set_throw_error_not_initialized(self):
    runner_object = runner.Runner(self.runner_parameters)
    with self.assertRaisesRegex(
        ValueError, r'There is no test set to evaluate on'
    ):
      runner_object.evaluate_model()

  def test_supervised_model_not_instantiated_throw_error(self):
    self.runner_parameters.upload_only = True
    self.runner_parameters.output_bigquery_table_path = (
        'project.dataset.output_table'
    )
    runner_object = runner.Runner(self.runner_parameters)
    runner_object.run()

    with self.assertRaisesRegex(
        ValueError, r'Evaluate called without a trained supervised model'
    ):
      runner_object.evaluate_model()

  def test_evaluatiom_dataset_batch_training(self):
    self.runner_parameters.test_dataset_holdout_fraction = 0.5
    # Set batch size to ensure that we are building a test set over multiple
    # batches from the entire dataset.
    self.runner_parameters.labeling_and_model_training_batch_size = 50

    runner_object = runner.Runner(self.runner_parameters)
    self._create_mock_datasets()

    runner_object.run()

    self.assertAllClose(
        len(runner_object.test_y),
        (self.per_class_labeled_example_count * 2)
        * self.runner_parameters.test_dataset_holdout_fraction,
    )
    self.assertAllClose(
        len(runner_object.test_x),
        (self.per_class_labeled_example_count * 2)
        * self.runner_parameters.test_dataset_holdout_fraction,
    )

  @parameterized.expand([
      (None, 99, 0.01, 99),
      (5, None, 5, 98),
      (None, None, 0.01, 98),
      (5, 99, 5, 99),
  ])
  @mock.patch.object(
      data_loader.DataLoader, 'get_label_thresholds', autospec=True
  )
  def test_threshold_parameter_initialization_positive_threshold_set_no_error(
      self,
      positive_threshold,
      negative_threshold,
      expected_positive,
      expected_negative,
      mock_get_label_thresholds,
  ):
    # This test evaluates each option for manually setting parameters: positive
    # only, negative only, setting neither, and setting both.

    mock_get_label_thresholds.return_value = {
        'positive_threshold': 0.01,
        'negative_threshold': 98,
    }
    self.runner_parameters.positive_threshold = positive_threshold
    self.runner_parameters.negative_threshold = negative_threshold

    # Parameters are set on init, so we do not have to call any methods on the
    # runner object.
    runner_object = runner.Runner(self.runner_parameters)

    self.assertEqual(
        runner_object.runner_parameters.positive_threshold,
        expected_positive,
    )
    self.assertEqual(
        runner_object.runner_parameters.negative_threshold,
        expected_negative,
    )


if __name__ == '__main__':
  tf.test.main()
