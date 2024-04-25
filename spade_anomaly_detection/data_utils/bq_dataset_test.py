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

import itertools
from typing import cast
from unittest import mock

from google.cloud import bigquery
from google.cloud import bigquery_storage
import numpy as np
import pandas as pd
from parameterized import parameterized

from spade_anomaly_detection.data_utils import bq_dataset
from spade_anomaly_detection.data_utils import bq_dataset_test_utils
from spade_anomaly_detection.data_utils import bq_utils
from spade_anomaly_detection.data_utils import feature_metadata
import tensorflow as tf


def _parameterized_name_func(testcase_func, param_num, param):
  return parameterized.to_safe_name(f'{testcase_func.__name__}_{param_num}_'
                                    f'{"_".join(str(x) for x in param.args)}')


class BQFeatureConversionInfoTest(tf.test.TestCase):

  @parameterized.expand([
      ('float', 0.0, np.float32, tf.float32),
      ('integer', 0, np.int32, tf.int32),
      ('string', '__NULL_PLACEHOLDER__', np.str_, tf.string),
  ])
  def test_convert_series_to_tensor_no_mask(self, name, null_value, np_dtype,
                                            tf_dtype):
    del name  # Unused
    # Ensure that the output is expected for each of the null value and data
    # type pairs that we expect to use.
    input_series = pd.Series(np.arange(8))
    input_series = input_series.astype(np_dtype)

    # Null out some values.
    input_series.iloc[0] = pd.NA
    input_series.iloc[-2] = pd.NA

    expected_output = input_series.copy(deep=True)
    expected_output.iloc[0] = null_value
    expected_output.iloc[-2] = null_value
    expected_output = tf.constant(expected_output.astype(np_dtype), tf_dtype)

    converter = bq_dataset.BQFeatureConverter(null_value, np_dtype, tf_dtype)
    output_tensor = converter.series_to_tensor(input_series, with_mask=False)

    self.assertIsInstance(output_tensor, tf.Tensor)
    self.assertDTypeEqual(output_tensor, tf_dtype)
    self.assertEqual(output_tensor[0], null_value)
    self.assertEqual(output_tensor[-2], null_value)
    self.assertAllEqual(expected_output, output_tensor)

  @parameterized.expand([
      ('float', 0.0, np.float32, tf.float32),
      ('integer', 0, np.int32, tf.int32),
      ('string', '__NULL_PLACEHOLDER__', np.str_, tf.string),
  ])
  def test_convert_series_to_tensor_with_mask(self, name, null_value, np_dtype,
                                              tf_dtype):
    del name  # Unused
    input_series = pd.Series(np.arange(8))
    input_series = input_series.astype(np_dtype)

    # Null out some values.
    input_series.iloc[0] = pd.NA
    input_series.iloc[-2] = pd.NA

    expected_output = input_series.copy(deep=True)
    expected_output.iloc[0] = expected_output.iloc[-2] = null_value
    expected_output = tf.constant(expected_output.astype(np_dtype), tf_dtype)

    expected_null = np.zeros(input_series.shape[0], dtype=bool)
    expected_null[0] = expected_null[-2] = True

    converter = bq_dataset.BQFeatureConverter(null_value, np_dtype, tf_dtype)
    output_dict = converter.series_to_tensor(input_series, with_mask=True)

    self.assertIn('values', output_dict)
    self.assertIn('was_null', output_dict)

    values_tensor = output_dict['values']
    self.assertDTypeEqual(values_tensor, tf_dtype)
    self.assertAllEqual(expected_output, values_tensor)

    mask_tensor = output_dict['was_null']
    self.assertDTypeEqual(mask_tensor, tf.bool)
    self.assertAllEqual(expected_null, mask_tensor)


class FeatureSeriesEqualTest(tf.test.TestCase):

  def assert_feature_dict_equals_series(
      self,
      feature_data,
      expected_values,
      expected_mask,
      with_mask,
  ):
    """Asserts that the feature data matches the data in the series.

    Args:
      feature_data: The feature data are: a Tensor of feature values if
        with_mask is false, or a dictionary with keys 'values' and 'was_null' if
        with_mask is True.
      expected_values: The series to compare the values Tensor with.
      expected_mask: The series to compare the was_nulls Tensor with. Not used
        if with_mask=False.
      with_mask: If True the feature_data should include a Tensor showing which
        of the input elements were null before being filled.
    """
    # pyformat: enable
    # If with_mask is True check that the sub-dictionary has all
    # fields and that none of the was_null values were True.
    if with_mask:
      # Check for expected fields
      self.assertIn('values', feature_data)
      self.assertIn('was_null', feature_data)
      self.assertShapeEqual(feature_data['values'], feature_data['was_null'])
      # Make sure mask was all False
      self.assertAllEqual(feature_data['was_null'], expected_mask)
      # Move values to c_data to abstract over differences in output
      # between with_mask=True and with_mask=False.
      feature_data = feature_data['values']

    values_assertion = (
        self.assertAllClose
        if feature_data.dtype.is_floating else self.assertAllEqual)

    values_assertion(
        feature_data,
        expected_values,
        msg='The batched values should match the sliced data,',
    )


class BigQueryBatchGeneratorTest(FeatureSeriesEqualTest):

  def setUp(self):
    super().setUp()
    self.mock_client = mock.create_autospec(
        bigquery.Client, instance=True, spec_set=True)
    self.query_mock = self.mock_client.query

  @parameterized.expand([('nested', True), ('not_nested', False)])
  def test_query_batch_generator_crates_a_dictionary_of_tensors_from_a_query(
      self, name: str, nested: bool
  ):
    del name  # Unused
    input_df = pd.DataFrame.from_dict({
        'bignumeric_feature': [1.0, 2.0, np.nan],
        'float_feature': [11.0, 12.0, np.nan],
        'int_feature': [2, 1, np.nan],
        'numeric_feature': [21.0, 22.0, np.nan],
        'string_feature': ['one', 'two', None],
    })
    bq_dataset_test_utils.mock_get_bigquery_dataset_return_value(
        self.mock_client, [input_df], mock_list_rows=False
    )

    metadata = feature_metadata.FeatureMetadataContainer([
        feature_metadata.FeatureMetadata('bignumeric_feature', 0, 'BIGNUMERIC'),
        feature_metadata.FeatureMetadata('float_feature', 1, 'FLOAT64'),
        feature_metadata.FeatureMetadata('int_feature', 2, 'INT64'),
        feature_metadata.FeatureMetadata('numeric_feature', 3, 'NUMERIC'),
        feature_metadata.FeatureMetadata('string_feature', 4, 'STRING'),
    ])

    output = bq_dataset.bigquery_query_batch_generator(
        'SELECT * from project.dataset.table',
        metadata,
        self.mock_client,
        batch_size=3,
        with_mask=True,
        nested=nested,
    )

    # The query should not be called until we start iterating through the
    # results.
    self.query_mock.assert_not_called()

    all_batches = []
    for batch in output:
      all_batches.append(batch)

    self.query_mock.assert_called()

    self.assertLen(all_batches, 1)
    output_data = all_batches[0]

    def _assert_feature(name, expected_value, expected_was_null):
      if nested:
        self.assertIn(name, output_data)
        feature_data = output_data[name]
      else:
        self.assertIn(f'{name}_values', output_data)
        self.assertIn(f'{name}_was_null', output_data)
        feature_data = {
            'values': output_data[f'{name}_values'],
            'was_null': output_data[f'{name}_was_null'],
        }

      self.assertAllEqual(feature_data['values'], expected_value)
      self.assertAllEqual(feature_data['was_null'], expected_was_null)

    _assert_feature(
        'bignumeric_feature',
        tf.constant([1.0, 2.0, bq_dataset.NULL_FLOAT_PLACEHOLDER], tf.float32),
        tf.constant([False, False, True]),
    )

    _assert_feature(
        'float_feature',
        tf.constant(
            [11.0, 12.0, bq_dataset.NULL_FLOAT_PLACEHOLDER], tf.float32
        ),
        tf.constant([False, False, True]),
    )

    _assert_feature(
        'int_feature',
        tf.constant([2, 1, bq_dataset.NULL_INT_PLACEHOLDER], tf.int32),
        tf.constant([False, False, True]),
    )

    _assert_feature(
        'numeric_feature',
        tf.constant(
            [21.0, 22.0, bq_dataset.NULL_FLOAT_PLACEHOLDER], tf.float32
        ),
        tf.constant([False, False, True]),
    )

    _assert_feature(
        'string_feature',
        tf.constant(['one', 'two', bq_dataset.NULL_STRING_PLACEHOLDER]),
        tf.constant([False, False, True]),
    )

  @parameterized.expand([('nested', True), ('not_nested', False)])
  def test_table_batch_generator_crates_a_dictionary_of_tensors_for_a_table(
      self, name: str, nested: bool
  ):
    del name  # Unused
    input_df = pd.DataFrame.from_dict({
        'bignumeric_feature': [1.0, 2.0, np.nan],
        'float_feature': [11.0, 12.0, np.nan],
        'int_feature': [2, 1, np.nan],
        'numeric_feature': [21.0, 22.0, np.nan],
        'string_feature': ['one', 'two', None],
    })
    bq_dataset_test_utils.mock_get_bigquery_dataset_return_value(
        self.mock_client, [input_df], mock_query=False
    )

    metadata = feature_metadata.BigQueryTableMetadata(
        [
            feature_metadata.FeatureMetadata(
                'bignumeric_feature', 0, 'BIGNUMERIC'
            ),
            feature_metadata.FeatureMetadata('float_feature', 1, 'FLOAT64'),
            feature_metadata.FeatureMetadata('int_feature', 2, 'INT64'),
            feature_metadata.FeatureMetadata('numeric_feature', 3, 'NUMERIC'),
            feature_metadata.FeatureMetadata('string_feature', 4, 'STRING'),
        ],
        project_id='project',
        bq_dataset_name='dataset',
        bq_table_name='table',
    )
    batch_size = 3
    output = bq_dataset.bigquery_table_batch_generator(
        metadata,
        self.mock_client,
        batch_size=batch_size,
        with_mask=True,
        nested=nested,
    )

    self.mock_client.list_rows.assert_not_called()

    all_batches = []
    for batch in output:
      all_batches.append(batch)

    self.assertLen(all_batches, 1)
    output_data = all_batches[0]

    self.mock_client.list_rows.assert_called_once_with(
        metadata.bigquery_table,
        metadata.to_bigquery_schema(),
        page_size=batch_size,
    )

    def _assert_feature(name, expected_value, expected_was_null):
      if nested:
        self.assertIn(name, output_data)
        feature_data = output_data[name]
      else:
        self.assertIn(f'{name}_values', output_data)
        self.assertIn(f'{name}_was_null', output_data)
        feature_data = {
            'values': output_data[f'{name}_values'],
            'was_null': output_data[f'{name}_was_null'],
        }

      self.assertAllEqual(feature_data['values'], expected_value)
      self.assertAllEqual(feature_data['was_null'], expected_was_null)

    _assert_feature(
        'bignumeric_feature',
        tf.constant([1.0, 2.0, bq_dataset.NULL_FLOAT_PLACEHOLDER], tf.float32),
        tf.constant([False, False, True]),
    )

    _assert_feature(
        'float_feature',
        tf.constant([11.0, 12.0, bq_dataset.NULL_FLOAT_PLACEHOLDER],
                    tf.float32),
        tf.constant([False, False, True]),
    )

    _assert_feature(
        'int_feature',
        tf.constant([2, 1, bq_dataset.NULL_INT_PLACEHOLDER], tf.int32),
        tf.constant([False, False, True]),
    )

    _assert_feature(
        'numeric_feature',
        tf.constant([21.0, 22.0, bq_dataset.NULL_FLOAT_PLACEHOLDER],
                    tf.float32),
        tf.constant([False, False, True]),
    )

    _assert_feature(
        'string_feature',
        tf.constant(['one', 'two', bq_dataset.NULL_STRING_PLACEHOLDER]),
        tf.constant([False, False, True]),
    )

  @parameterized.expand(itertools.product([True, False], [True, False], [2, 3]))
  def test_can_drop_remainder(self, nested: bool, drop_remainder: bool,
                              batch_size: int):
    input_df = pd.DataFrame.from_dict({
        'bignumeric_feature': [1.0, 2.0, np.nan],
        'float_feature': [11.0, 12.0, np.nan],
        'int_feature': [2, 1, np.nan],
        'numeric_feature': [21.0, 22.0, np.nan],
        'string_feature': ['one', 'two', None],
    })
    bq_dataset_test_utils.mock_get_bigquery_dataset_return_value(
        self.mock_client, [input_df], mock_list_rows=False
    )

    metadata = feature_metadata.FeatureMetadataContainer([
        feature_metadata.FeatureMetadata('bignumeric_feature', 0, 'BIGNUMERIC'),
        feature_metadata.FeatureMetadata('float_feature', 1, 'FLOAT64'),
        feature_metadata.FeatureMetadata('int_feature', 2, 'INT64'),
        feature_metadata.FeatureMetadata('numeric_feature', 3, 'NUMERIC'),
        feature_metadata.FeatureMetadata('string_feature', 4, 'STRING'),
    ])

    output = bq_dataset.bigquery_query_batch_generator(
        'SELECT * from project.dataset.table',
        metadata,
        self.mock_client,
        batch_size=batch_size,
        with_mask=True,
        nested=nested,
        drop_remainder=drop_remainder,
    )

    all_batches = []
    for batch in output:
      all_batches.append(batch)

    if batch_size == 3:
      self.assertLen(all_batches, 1)
    else:
      if drop_remainder:
        self.assertLen(all_batches, 1)
      else:
        self.assertLen(all_batches, 2)

  @parameterized.expand(
      itertools.product((True, False), (True, False)),
      name_func=_parameterized_name_func,
  )
  def test_handles_varying_output_sizes_to_make_batches(self, with_mask,
                                                        nested):
    # Because the BigQuery client does not appear to always honor the page_size
    # request we have adjusted our client to do aggregation internally if needed
    # to obtain the requested batch_size.
    # https://github.com/googleapis/python-bigquery/issues/915#issuecomment-910730406
    # https://github.com/googleapis/python-bigquery/issues/50
    # https://github.com/som-shahlab/subpopulation_robustness/blob/af6d177024b4e0fdce7cb1615dd0c08a2001c36e/group_robustness_fairness/prediction_utils/extraction_utils/database.py#L85

    n_features = 3
    batch_size = 5
    metadata = feature_metadata.FeatureMetadataContainer([
        feature_metadata.FeatureMetadata(f'col{i}', i, 'FLOAT64')
        for i in range(n_features)
    ])
    columns = [m.name for m in metadata]
    # Create a list of dataframes to replicate different page sizes being
    # returned by the client.
    dfs = [
        pd.DataFrame(np.random.rand(4, n_features), columns=columns),
        pd.DataFrame(np.random.rand(13, n_features), columns=columns),
        pd.DataFrame(np.random.rand(2, n_features), columns=columns),
        pd.DataFrame(np.random.rand(1, n_features), columns=columns),
        pd.DataFrame(np.random.rand(4, n_features), columns=columns),
    ]
    # Have to_dataframe_iterable return the above list.
    bq_dataset_test_utils.mock_get_bigquery_dataset_return_value(
        self.mock_client, dfs, mock_list_rows=False
    )

    # Slices of this combined DataFrame should be returned in each batch.
    combined_df = pd.concat(dfs, axis=0, ignore_index=True)

    # This should create 5 batches based on the batch_size of 5 and the total
    # number of rows being 24.
    expected_batch_sizes = (5, 5, 5, 5, 4)

    output = bq_dataset.bigquery_query_batch_generator(
        'SELECT * from project.dataset.table',
        metadata,
        self.mock_client,
        batch_size=batch_size,
        with_mask=with_mask,
        nested=nested,
    )

    # Get all of the outputs from generator and put then in a list.
    output_batches = list(output)
    self.assertLen(
        output_batches, 5, msg='Did not get the expected number of batches')

    batch_start_index = 0
    # Because there are multiple batches that are output we will iterate through
    # them and make sure that they all match the corresponding slice of the
    # combined DataFrame.
    for batch_index, cur_batch in enumerate(output_batches):
      with self.subTest(batch_index=batch_index):
        # All of the columns should have the expected size.
        expected_size = expected_batch_sizes[batch_index]
        # There are no nulls in any of the data.
        no_nulls = np.zeros(expected_size, dtype=np.bool_)
        # Loop through all of the columns to make sure they each match.
        for c_name in columns:
          with self.subTest(column_name=c_name):
            if with_mask and not nested:
              self.assertIn(
                  f'{c_name}_values',
                  cur_batch,
                  msg="Column's values should be in every batch.",
              )
              self.assertIn(
                  f'{c_name}_was_null',
                  cur_batch,
                  msg="Column's was null should be in every batch.",
              )
              c_data = {
                  'values': cur_batch[f'{c_name}_values'],
                  'was_null': cur_batch[f'{c_name}_was_null'],
              }
            else:
              self.assertIn(
                  c_name,
                  cur_batch,
                  msg='Each column should be in every batch.')
              c_data = cur_batch[c_name]

            # Pandas loc is inclusive so subtract 1.
            batch_end_inclusive = batch_start_index + expected_size - 1
            expected_series = combined_df.loc[
                batch_start_index:batch_end_inclusive, c_name]
            self.assert_feature_dict_equals_series(c_data, expected_series,
                                                   no_nulls, with_mask)

      batch_start_index += expected_size


class TensorOutputSignatureFromMetadataTest(tf.test.TestCase):

  @parameterized.expand(
      itertools.product((True, False), (True, False)),
      name_func=_parameterized_name_func,
  )
  def test_can_convert_metadata_into_a_tensor_spec(self, with_mask, nested):
    bq_dtypes = [
        'INT64',
        'NUMERIC',
        'BIGNUMERIC',
        'FLOAT64',
        'STRING',
        'BYTES',
        'BOOL',
    ]
    expected_tf_dtypes = [
        tf.int32,
        tf.float32,
        tf.float32,
        tf.float32,
        tf.string,
        tf.string,
        tf.bool,
    ]
    all_metadata = []
    for idx, bq_dtype in enumerate(bq_dtypes):
      all_metadata.append(
          feature_metadata.FeatureMetadata(
              name=f'col{idx}',
              index=idx,
              input_data_type=bq_dtype,
          ))
    metadata_container = feature_metadata.FeatureMetadataContainer(all_metadata)
    bq_dataset.update_tf_data_types_from_bq_data_types(metadata_container)

    output_spec = bq_dataset.tensor_output_signature_from_metadata(
        metadata_container, with_mask=with_mask, nested=nested)

    self.assertIsInstance(output_spec, dict)
    # Make sure all of the columns are in the output spec.
    for index, feature in enumerate(metadata_container):
      with self.subTest(batch_index=feature.name):
        if with_mask:
          if nested:
            self.assertIn(feature.name, output_spec)
            sub_dict = output_spec[feature.name]
            self.assertIn('values', sub_dict)
            self.assertIn('was_null', sub_dict)
            values_spec = sub_dict['values']
            null_spec = sub_dict['was_null']
          else:
            self.assertIn(f'{feature.name}_values', output_spec)
            self.assertIn(f'{feature.name}_was_null', output_spec)
            values_spec = output_spec[f'{feature.name}_values']
            null_spec = output_spec[f'{feature.name}_was_null']

          # We now will always have a TensorSpec.
          null_spec = cast(tf.TensorSpec, null_spec)
          values_spec = cast(tf.TensorSpec, values_spec)

          self.assertEqual(null_spec.dtype, tf.bool)
          self.assertIsNone(
              null_spec.shape[0], msg='The batch dimension should be null.')

          self.assertAllEqual(null_spec.shape, values_spec.shape)
          self.assertEqual(values_spec.dtype, expected_tf_dtypes[index])
          self.assertIsNone(
              values_spec.shape[0],
              msg='The batch dimension should be variable.',
          )
        else:
          values_spec = output_spec[feature.name]
          # We now will always have a TensorSpec.
          values_spec = cast(tf.TensorSpec, values_spec)
          self.assertEqual(values_spec.dtype, expected_tf_dtypes[index])
          self.assertIsNone(
              values_spec.shape[0],
              msg='The batch dimension should be variable.',
          )


class GetBigQueryDatasetTest(FeatureSeriesEqualTest):

  def setUp(self):
    super().setUp()
    self.mock_client = mock.create_autospec(
        bigquery.Client, instance=True, spec_set=True)
    self.query_mock = self.mock_client.query

  @parameterized.expand([('nested', True), ('not_nested', False)])
  def test_returns_a_dataset_with_the_expected_data(self, _, nested):
    with_mask = True
    limit = None
    table_metadata = feature_metadata.BigQueryTableMetadata(
        [
            feature_metadata.FeatureMetadata('col0', 0, 'FLOAT64'),
            feature_metadata.FeatureMetadata('col1', 1, 'INT64'),
            feature_metadata.FeatureMetadata('col2', 2, 'BOOL'),
            feature_metadata.FeatureMetadata('col3', 3, 'STRING'),
        ],
        'project',
        'dataset',
        'table_name',
    )
    columns = [m.name for m in table_metadata]

    def _create_rand_df():
      """Creates a DataFrame with random data of the correct types."""
      all_data = np.random.randn(16, len(table_metadata))
      all_data[0, 0] = None
      all_data[:, 1] = all_data[:, 1].astype(int).astype(float)
      all_data[1, 1] = None
      all_data[:, 2] = all_data[:, 2] > 0
      all_data[2, 2] = None
      all_data[:, 3] = all_data[:, 3].astype(int).astype(str)
      all_data[3, 3] = None
      return pd.DataFrame(all_data, columns=columns)

    # Create a list of dataframes to replicate different page sizes being
    # returned by the client.
    df_list = [_create_rand_df(), _create_rand_df(), _create_rand_df()]

    # With no limit or where clause this calls the list_rows method on
    # bq_client.
    (
        self.mock_client.list_rows.return_value.to_dataframe_iterable.return_value
    ) = df_list

    # With the default batch_size of 64 we expect 2 batches the second of which
    # is partial. Make sure to copy any DataFrames so you don't modify the
    # inputs.
    expected_dataframes = [
        pd.concat(df_list[:2], axis=0, ignore_index=True),
        df_list[-1].copy(deep=True),
    ]

    expected_masks = []
    # Convert the dataframe into the expected forms
    for ex_df in expected_dataframes:
      # Keep track of the null locations so we can check those.
      expected_masks.append(pd.isnull(ex_df))
      # Fill null values appropriately in each column.
      ex_df.fillna(
          {
              'col0': bq_dataset.NULL_FLOAT_PLACEHOLDER,
              'col1': bq_dataset.NULL_INT_PLACEHOLDER,
              'col2': bq_dataset.NULL_BOOL_PLACEHOLDER,
              'col3': bq_dataset.NULL_STRING_PLACEHOLDER,
          },
          inplace=True,
      )
      # Do dtype conversions for the columns that must change.
      ex_df.loc[:, 'col1'] = ex_df.loc[:, 'col1'].astype(int)
      # Tensorflow seems to return them as bytes rather than strings.
      ex_df.loc[:, 'col2'] = ex_df.loc[:, 'col2'].astype(bool)
      ex_df.loc[:, 'col3'] = ex_df.loc[:, 'col3'].astype(bytes)

    output_dataset = bq_dataset.get_bigquery_dataset(
        table_metadata,
        self.mock_client,
        batch_size=32,
        with_mask=with_mask,
        nested=nested,
        limit=limit,
    )

    self.assertIsInstance(output_dataset, tf.data.Dataset)

    # Loop through the batches and make sure they all match.
    for epoch in range(1, 3):
      batch_index = 0
      for cur_batch in output_dataset:
        with self.subTest(batch_index=batch_index, epoch=epoch):
          expected_df = expected_dataframes[batch_index]
          # Loop through all the columns to make sure they each match.
          for c_name in columns:
            with self.subTest(column_name=c_name):
              if nested:
                self.assertIn(
                    c_name,
                    cur_batch,
                    msg='Each column should be in every batch.',
                )
                c_data = cur_batch[c_name]
              else:
                self.assertIn(
                    f'{c_name}_values',
                    cur_batch,
                    msg="Column's values should be in every batch.",
                )
                self.assertIn(
                    f'{c_name}_was_null',
                    cur_batch,
                    msg="Column's was null should be in every batch.",
                )
                c_data = {
                    'values': cur_batch[f'{c_name}_values'],
                    'was_null': cur_batch[f'{c_name}_was_null'],
                }
              self.assert_feature_dict_equals_series(
                  c_data,
                  expected_df[c_name],
                  expected_masks[batch_index][c_name],
                  with_mask,
              )
        batch_index += 1
      self.assertLen(
          expected_dataframes,
          batch_index,
          msg='The number of batches was different than expected.',
      )
    self.assertEqual(epoch, 2)


class GetDatasetAndMetadataForTableTest(tf.test.TestCase):

  @parameterized.expand((
      (
          'both',
          'project.dataset.table',
          bq_utils.BQTablePathParts('project', 'dataset', 'table'),
          r'Only one of .* can be specified.',
      ),
      ('neither', None, None, r'Either .* must be specified'),
  ))
  def test_get_dataset_and_metadata_for_table_raises_for_invalid_table_inputs(
      self, _, table_path, table_parts, err_regex):
    with self.assertRaisesRegex(ValueError, err_regex):
      bq_dataset.get_dataset_and_metadata_for_table(
          table_path=table_path, table_parts=table_parts)

  @mock.patch.object(bq_dataset, 'get_bigquery_dataset', autospec=True)
  @mock.patch.object(
      feature_metadata.BigQueryMetadataBuilder, 'get_metadata_for_all_features'
  )
  def test_get_dataset_and_metadata_for_table_path(
      self,
      get_metadata_mock,
      get_bigquery_dataset_mock,
  ):
    table_path = 'project.dataset.table'
    mock_bq_client = mock.create_autospec(
        bigquery.Client, spec_set=True, instance=True)
    mock_bq_storage_client = mock.create_autospec(
        bigquery_storage.BigQueryReadClient, spec_set=True, instance=True)
    metadata_options = feature_metadata.MetadataRetrievalOptions.get_all()
    batch_sie = 128
    with_mask = False

    output_dataset, output_metadata = (
        bq_dataset.get_dataset_and_metadata_for_table(
            table_path=table_path,
            bigquery_client=mock_bq_client,
            bigquery_storage_client=mock_bq_storage_client,
            metadata_options=metadata_options,
            batch_size=batch_sie,
            with_mask=with_mask,
        )
    )

    get_metadata_mock.assert_called_once_with(metadata_options)

    get_bigquery_dataset_mock.assert_called_once_with(
        get_metadata_mock.return_value,
        mock_bq_client,
        bqstorage_client=mock_bq_storage_client,
        batch_size=batch_sie,
        with_mask=with_mask,
        cache_location=None,
        where_clauses=(),
        drop_remainder=False,
    )

    self.assertEqual(output_dataset, get_bigquery_dataset_mock.return_value)
    self.assertEqual(output_metadata, get_metadata_mock.return_value)

  @mock.patch.object(
      bq_dataset, 'update_tf_data_types_from_bq_data_types', autospec=True
  )
  @mock.patch.object(bigquery, 'Client', autospec=True)
  @mock.patch.object(bigquery_storage, 'BigQueryReadClient', autospec=True)
  @mock.patch.object(bq_dataset, 'get_bigquery_dataset', autospec=True)
  @mock.patch.object(
      feature_metadata.BigQueryMetadataBuilder, 'get_metadata_for_all_features'
  )
  def test_get_dataset_and_metadata_for_table_parts_defaults(
      self,
      get_metadata_mock,
      get_bigquery_dataset_mock,
      bq_storage_mock,
      bq_client_mock,
      update_tf_dtypes_mock,
  ):
    table_parts = bq_utils.BQTablePathParts('project', 'dataset', 'table')

    default_options = feature_metadata.MetadataRetrievalOptions.get_none()
    default_batch_size = 64
    default_with_mask = False

    (output_dataset, output_metadata) = (
        bq_dataset.get_dataset_and_metadata_for_table(table_parts=table_parts)
    )

    bq_storage_mock.assert_called_once_with()
    bq_client_mock.assert_called_once_with(project='project')
    get_metadata_mock.assert_called_once_with(default_options)
    update_tf_dtypes_mock.assert_called_once_with(
        get_metadata_mock.return_value)

    get_bigquery_dataset_mock.assert_called_once_with(
        get_metadata_mock.return_value,
        bq_client_mock.return_value,
        bqstorage_client=bq_storage_mock.return_value,
        batch_size=default_batch_size,
        with_mask=default_with_mask,
        cache_location=None,
        where_clauses=(),
        drop_remainder=False,
    )

    self.assertEqual(output_dataset, get_bigquery_dataset_mock.return_value)
    self.assertEqual(output_metadata, get_metadata_mock.return_value)


if __name__ == '__main__':
  tf.test.main()
