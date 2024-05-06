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

"""Tests for the data_loader module."""

import os
import re
from unittest import mock

from absl.testing import parameterized
from google.cloud import bigquery
import numpy as np
import pandas as pd

import pytest

from spade_anomaly_detection import data_loader
from spade_anomaly_detection import parameters
from spade_anomaly_detection.data_utils import bq_dataset
from spade_anomaly_detection.data_utils import bq_dataset_test_utils
from spade_anomaly_detection.data_utils import feature_metadata

import tensorflow as tf


class DataLoaderTest(tf.test.TestCase, parameterized.TestCase):

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
        test_bigquery_table_path='',
        test_label_col_name='',
        test_dataset_holdout_fraction=0.3,
        output_bigquery_table_path='',
        alpha=1.0,
        batches_per_model=1,
        labeling_and_model_training_batch_size=None,
        ensemble_count=5,
        verbose=False,
    )

    self.batch_size = 16
    self.dataset_size = (self.batch_size * 5, 10)

    self.mock_bq_dataset = self.enter_context(
        mock.patch.object(
            bq_dataset,
            'get_dataset_and_metadata_for_table',
            autospec=True,
        )
    )
    self.mock_bq_dataset.return_value = (
        self._get_example_tf_dataset(),
        None,
    )

    self.mock_bq_client = self.enter_context(
        mock.patch.object(bigquery, 'Client', autospec=True, spec_set=True)
    )

  def _get_example_tf_dataset(self) -> tf.data.Dataset:
    mock_client = mock.create_autospec(
        bigquery.Client, instance=True, spec_set=True
    )

    table_metadata = feature_metadata.BigQueryTableMetadata(
        [
            feature_metadata.FeatureMetadata('col0', 0, 'FLOAT64'),
            feature_metadata.FeatureMetadata('label', 1, 'INT64'),
        ],
        'project',
        'dataset',
        'table_name',
    )
    columns = [m.name for m in table_metadata]

    def _create_rand_df():
      """Creates a DataFrame with random data of the correct types."""
      all_data = np.random.randn(self.batch_size, len(table_metadata))
      all_data[:, 0] = all_data[:, 0].astype(np.float32)
      all_data[:, 1] = all_data[:, 1].astype(np.int8)
      return pd.DataFrame(all_data, columns=columns)

    # Create a list of dataframes to replicate different page sizes being
    # returned by the client.
    df_list = [_create_rand_df(), _create_rand_df()]
    # Have to_dataframe_iterable return the above list.
    bq_dataset_test_utils.mock_get_bigquery_dataset_return_value(
        mock_client, df_list
    )

    return bq_dataset.get_bigquery_dataset(
        table_metadata,
        mock_client,
        batch_size=self.batch_size,
        with_mask=False,
        nested=False,
        limit=None,
    )

  def _get_feature_metadata(self) -> feature_metadata.BigQueryTableMetadata:
    """Creates a metadata object to use for tests."""
    metadata_1 = feature_metadata.FeatureMetadata('x1', 0, 'FLOAT64')
    metadata_2 = feature_metadata.FeatureMetadata('x2', 1, 'FLOAT64')
    metadata_3 = feature_metadata.FeatureMetadata('x3', 2, 'FLOAT64')
    label_metadata = feature_metadata.FeatureMetadata(
        self.runner_parameters.label_col_name, 3, 'INT64'
    )
    metadata_container = [metadata_1, metadata_2, metadata_3, label_metadata]
    bq_metadata = feature_metadata.BigQueryTableMetadata(
        all_feature_metadata=metadata_container,
        project_id='project-id',
        bq_dataset_name='dataset-id',
        bq_table_name='table-id',
    )
    return bq_metadata

  def test_load_csv_default_no_error(self):
    features, labels = data_loader.load_dataframe('thyroid_labeled')

    self.assertIsInstance(features, pd.core.frame.DataFrame)
    self.assertIsInstance(labels, pd.core.series.Series)
    self.assertEqual(features.shape[0], labels.shape[0])
    self.assertEqual(labels.ndim, 1)
    self.assertEqual(features.ndim, 2)

  def test_load_csv_label_column_no_error(self):
    # Test using a different column for labels.
    features, labels = data_loader.load_dataframe(
        'thyroid_labeled',
        label_col_index=5,
    )

    self.assertEqual(features.shape[0], labels.shape[0])

  def test_load_csv_label_column_throws_error(self):
    with self.assertRaisesRegex(ValueError, r'is outside of the column range.'):
      data_loader.load_dataframe('thyroid_labeled', label_col_index=50)

  def test_load_csv_unknown_dataset_throws_error(self):
    with self.assertRaisesRegex(ValueError, r'Unknown dataset_name:'):
      data_loader.load_dataframe('not a dataset')

  def test_load_tensorflow_dataset_csv_type_no_error(self):
    batch_size = 100
    output_dataset = data_loader.load_tf_dataset_from_csv(
        'drug_train_pu_labeled', batch_size=batch_size
    )

    self.assertIsInstance(output_dataset, tf.data.Dataset)

  def test_load_tensorflow_dataset_csv_no_error_batched(self):
    batch_size = 100
    tf_dataset = data_loader.load_tf_dataset_from_csv(
        'drug_train_pu_labeled', batch_size=batch_size
    )

    feature_batch, label_batch = next(iter(tf_dataset.take(1)))

    self.assertEqual(feature_batch.dtype, tf.float32)
    self.assertEqual(label_batch.dtype, tf.int8)

    self.assertIsNotNone(feature_batch)
    self.assertIsNotNone(label_batch)

    self.assertLen(feature_batch, batch_size)
    self.assertLen(feature_batch, label_batch.numpy().shape[0])

  def test_load_tensorflow_dataset_csv_no_error_no_batch(self):
    tf_dataset = data_loader.load_tf_dataset_from_csv(
        'drug_train_pu_labeled', batch_size=None
    )
    self.assertIsInstance(tf_dataset, tf.data.Dataset)

    feature_batch, label_batch = next(iter(tf_dataset.take(1)))

    self.assertEqual(feature_batch.dtype, tf.float32)
    self.assertEqual(label_batch.dtype, tf.int8)

    # The feature set will be a single array of length 30, and the label will be
    # a single value.
    self.assertLen(feature_batch.numpy(), 30)
    self.assertEqual(label_batch.numpy(), -1)

  def test_load_tensorflow_dataset_name_throws_error(self):
    with self.assertRaisesRegex(ValueError, r'Unknown dataset_name:'):
      data_loader.load_tf_dataset_from_csv('not a dataset')

  def test_load_bigquery_dataset_no_error(self):
    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    batch_size = 100
    tf_dataset_instance_mock = mock.create_autospec(
        tf.data.Dataset, instance=True
    )
    self.mock_bq_dataset.return_value = (tf_dataset_instance_mock, None)

    data_loader_object.load_tf_dataset_from_bigquery(
        input_path=self.runner_parameters.input_bigquery_table_path,
        label_col_name=self.runner_parameters.label_col_name,
        batch_size=batch_size,
    )

    bq_dataset_table_path = self.mock_bq_dataset.call_args.kwargs['table_path']
    bq_dataset_batch_size = self.mock_bq_dataset.call_args.kwargs['batch_size']

    self.assertEqual(
        bq_dataset_table_path, self.runner_parameters.input_bigquery_table_path
    )
    self.assertEqual(bq_dataset_batch_size, batch_size)

  def test_load_bigquery_dataset_type_no_error(self):
    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    batch_size = 16

    data_loader_result = data_loader_object.load_tf_dataset_from_bigquery(
        input_path=self.runner_parameters.input_bigquery_table_path,
        label_col_name=self.runner_parameters.label_col_name,
        batch_size=batch_size,
    )

    self.assertIsInstance(data_loader_result, tf.data.Dataset)

  def test_load_bigquery_dataset_mapping_no_error(self):
    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    data_loader_result = data_loader_object.load_tf_dataset_from_bigquery(
        input_path=self.runner_parameters.input_bigquery_table_path,
        label_col_name=self.runner_parameters.label_col_name,
        batch_size=self.batch_size,
    )

    numpy_iterator = data_loader_result.as_numpy_iterator()
    features, labels = next(numpy_iterator)

    self.assertEqual(features.dtype, np.float32)
    self.assertEqual(labels.dtype, np.int32)

    # Assert against batch_size=1 because we create the dataset with one column.
    self.assertEqual(features.shape, (self.batch_size, 1))
    self.assertEqual(labels.shape, (self.batch_size,))

    self.assertLen(list(numpy_iterator), 1)

  def test_load_bigquery_dataset_float64_features_mapping_no_error(self):
    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    data_loader_result = data_loader_object.load_tf_dataset_from_bigquery(
        input_path=self.runner_parameters.input_bigquery_table_path,
        label_col_name=self.runner_parameters.label_col_name,
        batch_size=self.batch_size,
        convert_features_to_float64=True,
    )

    numpy_iterator = data_loader_result.as_numpy_iterator()
    features, labels = next(numpy_iterator)

    self.assertEqual(features.dtype, np.float64)
    self.assertEqual(labels.dtype, np.int32)

    # Assert against batch_size=1 because we create the dataset with one column.
    self.assertEqual(features.shape, (self.batch_size, 1))
    self.assertEqual(labels.shape, (self.batch_size,))

    self.assertLen(list(numpy_iterator), 1)

  def test_load_bigquery_dataset_mapping_label_error(self):
    self.runner_parameters.label_col_name = 'not_a_name'
    data_loader_object = data_loader.DataLoader(self.runner_parameters)

    # The dataset we created has a label column name of label, this should fail.
    with self.assertRaisesRegex(
        ValueError, r'Label column not found in the input table.'
    ):
      data_loader_object.load_tf_dataset_from_bigquery(
          input_path=self.runner_parameters.input_bigquery_table_path,
          label_col_name=self.runner_parameters.label_col_name,
          batch_size=self.batch_size,
      )

  def test_load_bigquery_dataset_unlabeled_value_none(self):
    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    data_loader_object.load_tf_dataset_from_bigquery(
        input_path=self.runner_parameters.input_bigquery_table_path,
        label_col_name=self.runner_parameters.label_col_name,
        batch_size=self.batch_size,
    )

    mock_data_loader_metadata_value = self.mock_bq_dataset.call_args.kwargs[
        'metadata_options'
    ]

    # Ensure that a where statement was not created when we don't pass in label
    # values.
    self.assertIsNone(mock_data_loader_metadata_value)

  @mock.patch.object(
      feature_metadata, 'MetadataRetrievalOptions', autospec=True
  )
  def test_load_bigquery_dataset_unlabeled_value(self, metadata_mock):
    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    label_column_filter_value = -1

    data_loader_object.load_tf_dataset_from_bigquery(
        input_path=self.runner_parameters.input_bigquery_table_path,
        label_col_name=self.runner_parameters.label_col_name,
        batch_size=self.batch_size,
        label_column_filter_value=label_column_filter_value,
    )

    mock_metadata_call_actual = metadata_mock.call_args.kwargs['where_clauses']

    # Ensure that a where statement was not created when we don't pass in label
    # values.
    expected_where_clause = ['label = -1']
    self.assertEqual(expected_where_clause, mock_metadata_call_actual)

  def test_record_count_where_statement_no_error(self):
    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    where_statements = ['label = 1', 'col is not null']
    query = (
        'SELECT count(*) FROM `'
        + self.runner_parameters.input_bigquery_table_path
        + '` '
        + 'WHERE '
        + ' AND '.join(where_statements)
    )

    data_loader_object.get_query_record_result_length(
        input_path=self.runner_parameters.input_bigquery_table_path,
        where_statements=where_statements,
    )

    query_mock = self.mock_bq_client.return_value.__enter__.return_value.query

    query_mock.assert_called_with(query)

  def test_record_count_no_error(self):
    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    query = (
        'SELECT count(*) FROM `'
        + self.runner_parameters.input_bigquery_table_path
        + '` '
    )

    data_loader_object.get_query_record_result_length(
        input_path=self.runner_parameters.input_bigquery_table_path,
        where_statements=None,
    )

    query_mock = self.mock_bq_client.return_value.__enter__.return_value.query

    query_mock.assert_called_with(query)

  @mock.patch.object(
      feature_metadata, 'MetadataRetrievalOptions', autospec=True
  )
  def test_where_statement_construction_no_error(self, mock_metadata):
    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    expected_where_statements = ['time_col > 1', 'col2 IS NOT NULL']
    data_loader_object.load_tf_dataset_from_bigquery(
        input_path=self.runner_parameters.input_bigquery_table_path,
        label_col_name=self.runner_parameters.label_col_name,
        where_statements=expected_where_statements,
    )

    self.assertListEqual(
        expected_where_statements,
        mock_metadata.call_args.kwargs['where_clauses'],
    )

  @mock.patch.object(
      feature_metadata, 'MetadataRetrievalOptions', autospec=True
  )
  def test_where_statements_with_label_filter_no_error(self, mock_metadata):
    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    where_statements = ['time_col > 1', 'col2 IS NOT NULL']
    data_loader_object.load_tf_dataset_from_bigquery(
        input_path=self.runner_parameters.input_bigquery_table_path,
        label_col_name=self.runner_parameters.label_col_name,
        where_statements=where_statements,
        label_column_filter_value=1,
    )

    where_statements.append(f'{self.runner_parameters.label_col_name} = 1')

    self.assertListEqual(
        where_statements,
        mock_metadata.call_args.kwargs['where_clauses'],
    )

  @mock.patch.object(feature_metadata, 'BigQueryMetadataBuilder', autospec=True)
  def test_ignore_columns_load_tf_dataset_from_bigquery(
      self, mock_metadata_builder
  ):
    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    ignore_columns = ['col1', 'time']
    data_loader_object.load_tf_dataset_from_bigquery(
        input_path=self.runner_parameters.input_bigquery_table_path,
        label_col_name=self.runner_parameters.label_col_name,
        ignore_columns=ignore_columns,
    )

    mock_ignore_columns_call_list = mock_metadata_builder.call_args.kwargs[
        'ignore_columns'
    ]

    self.assertListEqual(ignore_columns, mock_ignore_columns_call_list)

  @mock.patch.object(bigquery, 'LoadJobConfig', autospec=True)
  def test_upload_dataframe_as_bigquery_table_no_error(
      self, mock_bqclient_loadjobconfig
  ):
    self.runner_parameters.output_bigquery_table_path = (
        'project.dataset.pseudo_labeled_data'
    )
    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    feature_column_names = ['x1', 'x2', self.runner_parameters.label_col_name]

    features = np.random.rand(10, 2).astype(np.float32)
    labels = np.repeat(0, 10).reshape(10, 1).astype(np.int8)

    tf_dataset_instance_mock = mock.create_autospec(
        tf.data.Dataset, instance=True
    )

    feature1_metadata = feature_metadata.FeatureMetadata('x1', 0, 'FLOAT64')
    feature2_metadata = feature_metadata.FeatureMetadata('x2', 0, 'FLOAT64')
    label_metadata = feature_metadata.FeatureMetadata(
        self.runner_parameters.label_col_name, 1, 'INT64'
    )
    metadata_container = feature_metadata.FeatureMetadataContainer(
        [feature1_metadata, feature2_metadata, label_metadata]
    )

    self.mock_bq_dataset.return_value = (
        tf_dataset_instance_mock,
        metadata_container,
    )

    # Perform this call so that FeatureMetadata is set.
    data_loader_object.load_tf_dataset_from_bigquery(
        input_path=self.runner_parameters.input_bigquery_table_path,
        label_col_name=self.runner_parameters.label_col_name,
        batch_size=self.batch_size,
    )

    data_loader_object.upload_dataframe_as_bigquery_table(
        features=features,
        labels=labels,
    )
    job_config_object = mock_bqclient_loadjobconfig.return_value

    load_table_mock_kwargs = (
        self.mock_bq_client.return_value.__enter__.return_value.load_table_from_dataframe.call_args.kwargs
    )

    with self.subTest(name='LabelColumnCorrect'):
      self.assertListEqual(
          list(
              load_table_mock_kwargs['dataframe'][
                  self.runner_parameters.label_col_name
              ]
          ),
          list(labels),
      )

    with self.subTest(name='LabelColumnDataTypeBool'):
      self.assertEqual(
          load_table_mock_kwargs['dataframe'][
              self.runner_parameters.label_col_name
          ].dtype,
          bool,
      )

    with self.subTest(name='EqualColumnNames'):
      self.assertListEqual(
          feature_column_names,
          list(load_table_mock_kwargs['dataframe'].columns),
      )
    with self.subTest(name='EqualDestinationPath'):
      self.assertEqual(
          self.runner_parameters.output_bigquery_table_path,
          load_table_mock_kwargs['destination'],
      )
    with self.subTest(name='EqualJobConfig'):
      self.assertEqual(job_config_object, load_table_mock_kwargs['job_config'])

  def test_bigquery_table_upload_throw_error_metadata(self):
    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    features = np.random.rand(10, 2).astype(np.float32)
    labels = np.repeat(0, 10).reshape(10, 1).astype(np.int8)

    with self.assertRaisesRegex(
        ValueError, r'Input feature metadata has not ' 'been instantiated yet'
    ):
      data_loader_object.upload_dataframe_as_bigquery_table(
          features=features, labels=labels
      )

  def test_get_label_thresholds_no_error(self):
    mock_query_return_dictionary = {
        self.runner_parameters.label_col_name: [
            self.runner_parameters.positive_data_value,
            self.runner_parameters.negative_data_value,
        ],
        'record_count': [200, 800],
    }
    query_dataframe_mock_result = pd.DataFrame(
        data=mock_query_return_dictionary
    )

    # Mock out the dataframe returned by the SQL query.
    self.mock_bq_client.return_value.__enter__.return_value.query.return_value.result.return_value.to_dataframe.return_value = (
        query_dataframe_mock_result
    )

    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    label_thresholds = data_loader_object.get_label_thresholds(
        self.runner_parameters.input_bigquery_table_path
    )

    self.assertEqual(label_thresholds['positive_threshold'], 20)
    self.assertEqual(label_thresholds['negative_threshold'], 80)

  def test_get_label_thresholds_with_where_statements_no_error(self):
    where_statements = ['label = 1', 'col is not null']
    sql_composed_where_statements = ' AND '.join(where_statements)

    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    data_loader_object.get_label_thresholds(
        bigquery_table_path=self.runner_parameters.input_bigquery_table_path,
        where_statements=where_statements,
    )

    query_mock_query_string = self.mock_bq_client.return_value.__enter__.return_value.query.call_args.kwargs[
        'query'
    ]

    where_statements_in_query = bool(
        re.search(sql_composed_where_statements, query_mock_query_string)
    )
    self.assertTrue(where_statements_in_query)

  def test_parse_gcs_folder_uri(self):
    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    gcs_uri = 'gs://bucket-name/folder/path'
    bucket_name, folder_name = data_loader_object.parse_gcs_folder_uri(gcs_uri)
    self.assertEqual(bucket_name, 'bucket-name')
    self.assertEqual(folder_name, 'folder/path')

  def test_construct_schema_from_dataframe(self):
    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    df = pd.DataFrame(
        {'x1': [0.1, 0.2, 0.3], 'x2': [4, 5, 6], 'label': [0.0, -1.0, 1.0]}
    )
    expected_schema = [
        bigquery.SchemaField(name='x1', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='x2', field_type='INTEGER', mode='NULLABLE'),
        bigquery.SchemaField(name='label', field_type='FLOAT', mode='NULLABLE'),
    ]
    actual_schema = data_loader_object.construct_schema_from_dataframe(df)
    self.assertListEqual(expected_schema, actual_schema)

  # Params to test: convert_label_to_bool.
  @parameterized.named_parameters(
      ('labels_to_bool', True, np.array([True, False], dtype=bool)),
      ('labels_unchanged', False, np.array([1.0, 0.0], dtype=np.float64)),
  )
  def test_construct_dataframe_from_features(
      self, convert_label_to_bool: bool, expected_labels: np.ndarray
  ):
    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    data_loader_object.input_feature_metadata = self._get_feature_metadata()
    features = np.array([[0.1, 0.2, 3], [0.3, 0.4, 4]], dtype=np.float64)
    labels = np.array([1, 0], dtype=np.int64)
    actual_df = data_loader_object.construct_dataframe_from_features(
        features,
        labels,
        convert_label_to_bool=convert_label_to_bool,
    )
    expected_df = pd.DataFrame({
        'x1': [0.1, 0.3],
        'x2': [0.2, 0.4],
        'x3': [3.0, 4.0],
        'label': expected_labels,
    })
    pd.testing.assert_frame_equal(expected_df, actual_df)

  def test_get_bigquery_job_config(self):
    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    expected_schema = [
        bigquery.SchemaField(name='x1', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='x2', field_type='INTEGER', mode='NULLABLE'),
        bigquery.SchemaField(name='y', field_type='FLOAT', mode='NULLABLE'),
    ]
    job_config = data_loader_object._get_bigquery_job_config(
        schema=expected_schema,
        write_disposition='WRITE_TRUNCATE',
        use_parquet=True,
    )
    self.assertEqual(job_config.schema, expected_schema)
    self.assertEqual(job_config.write_disposition, 'WRITE_TRUNCATE')
    self.assertEqual(job_config.source_format, bigquery.SourceFormat.PARQUET)

  @pytest.mark.skip(reason="create_tempdir is broken in pytest")
  def test_construct_file_paths_for_gcs(self):
    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    local_folder = 'test_folder'
    source_folder = self.create_tempdir(name=local_folder)
    name1 = 'labels_file_1.parquet'
    name2 = 'labels_file_2.parquet'
    _ = self.create_tempfile(
        file_path=f'{source_folder.full_path}/{name1}',
        content=b'0.1,0.2,0.0',
        mode='wb',
    )
    _ = self.create_tempfile(
        file_path=f'{source_folder.full_path}/{name2}',
        content=b'0.6,0.9,1.0',
        mode='wb',
    )
    file_paths = data_loader_object._construct_file_paths_for_gcs(source_folder)
    self.assertSetEqual(set(file_paths), {name1, name2})

  # Params to test: use_parquet.
  @parameterized.named_parameters(
      ('use_parquet', True),
      ('use_csv', False),
  )
  @pytest.mark.skip(reason="create_tempdir is broken in pytest")
  def test_upload_data_to_local_disk(self, use_parquet: bool):
    data_loader_object = data_loader.DataLoader(self.runner_parameters)
    data_loader_object.input_feature_metadata = self._get_feature_metadata()
    local_folder = 'test_folder'
    source_folder = self.create_tempdir(name=local_folder)
    name_ext = 'parquet' if use_parquet else 'csv'
    name = f'labels_file_1.{name_ext}'
    file_path = f'{source_folder.full_path}/{name}'
    features = np.array([[0.1, 0.2, 3], [0.3, 0.4, 4]], dtype=np.float64)
    labels = np.array([1, 0], dtype=np.int64)
    data_loader_object.upload_data_to_local_disk(
        file_path=file_path,
        features=features,
        labels=labels,
        convert_label_to_bool=False,
        use_parquet=use_parquet,
    )
    self.assertTrue(os.path.isfile(file_path))


if __name__ == '__main__':
  tf.test.main()
