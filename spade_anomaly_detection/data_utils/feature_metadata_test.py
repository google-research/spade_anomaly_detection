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

import dataclasses
import json
import re
import unittest
from unittest import mock

import frozendict
from google.cloud import bigquery
import numpy as np
from parameterized import parameterized

from spade_anomaly_detection.data_utils import feature_metadata


def _create_schema_response(all_names, all_dtypes):
  """Helper to mock the information schema response."""
  return [
      bigquery.SchemaField(name=n, field_type=t)
      for n, t in zip(all_names, all_dtypes)
  ]


def _get_numeric_data_response(feature_name, seed, options):
  """Creates a response data dictionary for a feature.

  Args:
    feature_name: The name of the feature.
    seed: A seed to help make the data unique.
    options: The options to simulate the result from.

  Returns:
    A dictionary with all possible numeric metadata_collection.
  """
  result = {}
  if options.get_mean:
    result[f'{feature_name}_mean'] = 1.2 + seed
  if options.get_variance:
    result[f'{feature_name}_variance'] = 2.2 + 2 * seed
  if options.get_min:
    result[f'{feature_name}_min'] = -1.0 - seed
  if options.get_max:
    result[f'{feature_name}_max'] = 100.0 + 3 * seed
  if options.get_median:
    result[f'{feature_name}_median'] = 2.123 + seed
  if options.get_log_mean or options.get_log_variance:
    result[f'{feature_name}_log_shift'] = -1.0001
  if options.get_log_mean:
    result[f'{feature_name}_log_mean'] = (-1.234 / seed,)
  if options.get_log_variance:
    result[f'{feature_name}_log_variance'] = -1.234 * seed
  if options.number_of_quantiles:
    result[f'{feature_name}_quantiles'] = np.arange(
        1, options.number_of_quantiles + 1
    ).tolist()

  return result


def _get_discrete_data_response(feature_name, cardinality):
  """Creates a response data dictionary for a discrete feature.

  The mode is always equal to cX where X is the cardinality.

  Args:
    feature_name: The name of the feature.
    cardinality: The cardinality of the feature. Also used to make the top count
      more unique.

  Returns:
    A dictionary with all possible numeric metadata_collection.
  """
  top_count = []
  for i in range(cardinality, 0, -1):
    top_count.append({
        'value': f'c{i}',
        'count': 123 * (i - 1) + 1,
    })
  return {
      f'{feature_name}_cardinality': cardinality,
      f'{feature_name}_top_count': top_count,
  }


def _get_vocab_from_top_count(top_count):
  return {d['value']: d['count'] for d in top_count}


_ALL_METADATA_RETRIEVAL_FALSE = frozendict.frozendict({
    'get_mean': False,
    'get_variance': False,
    'get_min': False,
    'get_max': False,
    'get_median': False,
    'get_log_mean': False,
    'get_log_variance': False,
    'min_log_value': 0.0,
    'number_of_quantiles': None,
    'get_mode': False,
    'max_vocab_size': 0,
})

_ALL_METADATA_RETRIEVAL_TRUE = frozendict.frozendict({
    'get_mean': True,
    'get_variance': True,
    'get_min': True,
    'get_max': True,
    'get_median': True,
    'get_log_mean': True,
    'get_log_variance': True,
    'min_log_value': 1.0e-4,
    'number_of_quantiles': 5,
    'get_mode': True,
    'max_vocab_size': 1,
})

_ALL_METADATA_IGNORED_KEYS_ANY = frozenset(('min_log_value',))


class MetadataRetrievalOptionsTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.all_options = tuple(_ALL_METADATA_RETRIEVAL_FALSE.keys())
    true_keys = tuple(_ALL_METADATA_RETRIEVAL_TRUE.keys())
    if cls.all_options != true_keys:
      raise ValueError(
          f'_ALL_METADATA_RETRIEVAL_FALSE keys ({cls.all_options}) must equal '
          f'_ALL_METADATA_RETRIEVAL_TRUE ({true_keys}) '
      )

  def test_get_all_makes_all_options_true(self):
    mro = feature_metadata.MetadataRetrievalOptions.get_all(
        number_of_quantiles=5, max_vocab_size=10
    )
    self.assertTrue(all(getattr(mro, o) for o in self.all_options))

  def test_get_none_makes_all_options_false(self):
    mro = feature_metadata.MetadataRetrievalOptions.get_none()
    self.assertFalse(any(getattr(mro, o) for o in self.all_options))

  def test_no_options_returns_true_if_all_objects_are_false(self):
    mro = feature_metadata.MetadataRetrievalOptions(
        **_ALL_METADATA_RETRIEVAL_FALSE
    )
    self.assertTrue(mro.no_options())

  def test_no_options_ignores_min_log_value(self):
    mro_options = dict(**_ALL_METADATA_RETRIEVAL_FALSE)
    mro_options['min_log_value'] = 1.0
    mro = feature_metadata.MetadataRetrievalOptions(**mro_options)
    self.assertTrue(mro.no_options())

  def test_any_ignores_where_clauses(self):
    mro_options = dict(**_ALL_METADATA_RETRIEVAL_FALSE)
    mro_options['where_clauses'] = ('this = test',)
    mro = feature_metadata.MetadataRetrievalOptions(**mro_options)
    self.assertTrue(mro.no_options())

  # Check all the parameters that aren't ignored.
  @parameterized.expand(
      k
      for k in _ALL_METADATA_RETRIEVAL_FALSE.keys()
      if k not in _ALL_METADATA_IGNORED_KEYS_ANY
  )
  def test_no_options_returns_false_if_any_option_is_true(
      self, true_option_name
  ):
    mro_options = dict(**_ALL_METADATA_RETRIEVAL_FALSE)
    if true_option_name.startswith('get_'):
      value = True
    else:
      value = 1
    mro_options[true_option_name] = value
    mro = feature_metadata.MetadataRetrievalOptions(**mro_options)
    self.assertFalse(mro.no_options())

  @parameterized.expand((
      ('minmax', ('get_min', 'get_max')),
      ('standard', ('get_mean', 'get_variance')),
  ))
  def test_for_normalization(self, normalization, expected_true_fields):
    mro = feature_metadata.MetadataRetrievalOptions.for_normalization(
        normalization
    )
    # To check that just the expected values are true check their value and then
    # set them to false and make sure that no others are set at the end.
    for field in expected_true_fields:
      with self.subTest('field'):
        self.assertTrue(getattr(mro, field))
        setattr(mro, field, False)

    self.assertGreater(mro.max_vocab_size, 0)
    mro.max_vocab_size = 0

    self.assertTrue(mro.no_options())

  def test_for_normalization_raises(self):
    with self.assertRaisesRegex(
        ValueError, r'The normalization .* is not valid'
    ):
      feature_metadata.MetadataRetrievalOptions.for_normalization('fake')


class FeatureMetadataTest(unittest.TestCase):

  def test_config_json_roundtrip(self):
    metadata = feature_metadata.FeatureMetadata(
        'name',
        1,
        input_data_type='NUMERIC',
        tf_data_type_str='float32',
        mean=1.234,
        variance=2.345,
        min=0.123,
        max=9.876,
        median=5.55,
        log_shift=1.1,
        log_mean=2.2,
        log_variance=3.3,
        quantiles=[1.1, 2.2, 3.3],
        cardinality=54321,
        mode='mode',
        vocabulary={
            'asdf': 1,
            True: 3,
            False: 4,
            2: 2,
            1.234: 5,
            None: 12345,
        },
    )

    config_json = json.dumps(metadata.get_config())
    recreated_metadata = feature_metadata.FeatureMetadata.from_config(
        json.loads(config_json)
    )
    self.assertEqual(metadata, recreated_metadata)


class BigQueryMetadataBuilderTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_client = mock.create_autospec(
        bigquery.Client, instance=True, spec_set=True
    )
    self.query_mock = self.mock_client.query
    self.table_mock = mock.create_autospec(
        bigquery.Table, instance=True, spec_set=True
    )
    self.mock_client.get_table.return_value = self.table_mock

  def _update_query_mock_with_results(self, results):
    """Updates a mock of the query method with a result."""
    query_job_mock = mock.Mock(autospec=bigquery.QueryJob)
    query_job_mock.result.side_effect = results
    self.query_mock.return_value = query_job_mock

  @mock.patch.object(bigquery, 'Client', autospec=True, spec_set=True)
  def test_full_table_id_includes_all_fields_and_quotes(
      self, client_class_mock
  ):
    expected_id = 'project_id.bq_dataset_name.bq_table_name'
    expected_escaped_id = '`project_id.bq_dataset_name.bq_table_name`'
    builder = feature_metadata.BigQueryMetadataBuilder(
        'project_id',
        'bq_dataset_name',
        'bq_table_name',
        bq_client=self.mock_client,
    )

    # Ensure that we didn't try to create another client.
    client_class_mock.assert_not_called()

    self.assertEqual(builder.full_table_id, expected_id)
    self.assertEqual(builder.escaped_table_id, expected_escaped_id)

  @mock.patch.object(bigquery, 'Client', autospec=True, spec_set=True)
  def test_creates_bq_client_if_not_provided(self, client_class_mock):
    # We mock the class her so we can stay hermetic and not try to get default
    # credentials in the tests.
    feature_metadata.BigQueryMetadataBuilder(
        'project_id', 'bq_dataset_name', 'bq_table_name'
    )
    client_class_mock.assert_called_once_with(project='project_id')

  def test_get_and_caches_total_row_count(self):
    self._update_query_mock_with_results(
        [
            iter(
                [[3]],
            )
        ]
    )

    builder = feature_metadata.BigQueryMetadataBuilder(
        'project_id',
        'bq_dataset_name',
        'bq_table_name',
        bq_client=self.mock_client,
    )
    self.assertEqual(builder.rows, 3)
    # Note that this does not include the instance because of how mock works.
    # TODO(b/333154677): Replace with matchers.
    self.query_mock.assert_called_once_with(
        'SELECT COUNT(*)\nFROM `project_id.bq_dataset_name.bq_table_name`',
        job_config=mock.ANY,
    )
    # Make sure it uses the cached value the second time.
    _ = builder.rows
    self.query_mock.assert_called_once()

  def test_get_column_names_and_types_raises_for_unsupported_dtypes(self):
    self.table_mock.schema = _create_schema_response(
        ['col1', 'col2'], ['INT64', 'FAKETYPE']
    )

    builder = feature_metadata.BigQueryMetadataBuilder(
        'project_id',
        'bq_dataset_name',
        'bq_table_name',
        bq_client=self.mock_client,
    )

    with self.assertRaises(NotImplementedError):
      builder.get_feature_names_and_types()

  def test_get_column_names_and_types_returns_metadata(self):
    column_dtypes = [
        'INT64',
        'NUMERIC',
        'BIGNUMERIC',
        'FLOAT64',
        'STRING',
        'BYTES',
        'BOOL',
    ]
    column_names = [f'col{i}' for i in range(len(column_dtypes))]
    self.table_mock.schema = _create_schema_response(
        column_names, column_dtypes
    )

    builder = feature_metadata.BigQueryMetadataBuilder(
        'project_id',
        'bq_dataset_name',
        'bq_table_name',
        bq_client=self.mock_client,
    )

    output_metadata = builder.get_feature_names_and_types()

    self.mock_client.get_table.assert_called_once_with(
        'project_id.bq_dataset_name.bq_table_name'
    )

    self.assertListEqual(
        column_names, [col_metadata.name for col_metadata in output_metadata]
    )
    self.assertListEqual(
        column_dtypes,
        [col_metadata.input_data_type for col_metadata in output_metadata],
    )

  def test_get_column_names_and_types_can_ignore_columns(self):
    ignored_columns = ['col3', 'col1']
    column_names = ['col1', 'col2', 'col3']
    column_dtypes = ['INT64', 'FLOAT64', 'STRING']
    self.table_mock.schema = _create_schema_response(
        column_names, column_dtypes
    )

    builder = feature_metadata.BigQueryMetadataBuilder(
        'project_id',
        'bq_dataset_name',
        'bq_table_name',
        ignored_columns,
        bq_client=self.mock_client,
    )

    output_metadata = builder.get_feature_names_and_types()

    self.assertEqual(len(output_metadata), 1)
    output_column = output_metadata[0]
    self.assertListEqual([output_column.name], ['col2'])
    self.assertListEqual([output_column.input_data_type], ['FLOAT64'])

  @parameterized.expand([
      ('all_log', True, True),
      ('mean_only', True, False),
      ('variance_only', False, True),
      ('no_log', False, False),
  ])
  def test_update_numeric_feature_metadata(
      self, name, get_log_mean, get_log_variance
  ):
    del name  # Unused
    feature = feature_metadata.FeatureMetadata('age', 1, 'FLOAT64')
    # TODO(b/333154677): Do this in an integration test to check the query
    #  correctness.
    options = feature_metadata.MetadataRetrievalOptions.get_all(
        number_of_quantiles=5
    )
    options.get_log_mean = get_log_mean
    options.get_log_variance = get_log_variance
    bq_results = _get_numeric_data_response('age', 1.234, options)

    self._update_query_mock_with_results([iter((bq_results,))])

    builder = feature_metadata.BigQueryMetadataBuilder(
        'project_id',
        'bq_dataset_name',
        'bq_table_name',
        bq_client=self.mock_client,
    )

    builder.update_numeric_feature_metadata(feature, options)

    self.query_mock.assert_called_once()

    args, _ = self.query_mock.call_args

    # For now just look for every desired output in the query.
    query = args[0]
    for output_attribute in bq_results:
      self.assertIn(
          output_attribute,
          query,
          msg=f'{output_attribute} was not in the query',
      )
    if get_log_mean or get_log_variance:
      self.assertIn(
          'log_transformed AS (SELECT LOG(',
          query,
          msg='The log transform was not cached as expected.',
      )
    else:
      self.assertNotIn(
          'log_transformed AS (SELECT LOG(',
          query,
          msg='The log transform should not be calculated.',
      )

    self.assertEqual(feature.mean, bq_results['age_mean'])
    self.assertEqual(feature.variance, bq_results['age_variance'])
    self.assertEqual(feature.min, bq_results['age_min'])
    self.assertEqual(feature.max, bq_results['age_max'])
    self.assertEqual(feature.median, bq_results['age_median'])
    if get_log_mean or get_log_variance:
      self.assertEqual(feature.log_shift, bq_results['age_log_shift'])
    else:
      self.assertIsNone(feature.log_shift)
    if get_log_mean:
      self.assertEqual(feature.log_mean, bq_results['age_log_mean'])
    else:
      self.assertIsNone(feature.log_mean)
    if get_log_variance:
      self.assertEqual(feature.log_variance, bq_results['age_log_variance'])
    else:
      self.assertIsNone(feature.log_variance)
    self.assertEqual(feature.quantiles, bq_results['age_quantiles'])

  def test_update_discrete_feature_metadata(self):
    # TODO(b/333154677): Make an integration test for this.
    # TODO(b/333154677): Do this in a way that is less coupled to the query.

    feature = feature_metadata.FeatureMetadata('class', 1, 'STRING')
    # TODO(b/333154677): Do this in an integration test to check the query
    #  correctness.
    cardinality = 2
    bq_results = _get_discrete_data_response('class', cardinality)
    self._update_query_mock_with_results([iter((bq_results,))])
    options = feature_metadata.MetadataRetrievalOptions.get_all()
    builder = feature_metadata.BigQueryMetadataBuilder(
        'project_id',
        'bq_dataset_name',
        'bq_table_name',
        bq_client=self.mock_client,
    )

    builder.update_discrete_feature_metadata(feature, options)

    self.query_mock.assert_called_once()

    args, _ = self.query_mock.call_args

    # For now just look for every desired output in the query.
    query = args[0]
    for output_attribute in bq_results:
      self.assertIn(
          output_attribute,
          query,
          msg=f'{output_attribute} was not in the query',
      )

    self.assertEqual(feature.cardinality, bq_results['class_cardinality'])
    self.assertEqual(feature.mode, f'c{bq_results["class_cardinality"]}')
    self.assertEqual(
        feature.vocabulary,
        _get_vocab_from_top_count(bq_results['class_top_count']),
    )

  def test_update_discrete_and_numeric_feature_metadata_defaults(self):
    feature = feature_metadata.FeatureMetadata('int_feature', 2, 'INT64')
    cardinality = 5
    default_options = feature_metadata.MetadataRetrievalOptions()
    discrete_results = _get_discrete_data_response('int_feature', cardinality)
    numeric_results = _get_numeric_data_response(
        'int_feature', 5.4321, default_options
    )
    all_results = {**numeric_results, **discrete_results}
    self._update_query_mock_with_results(
        [iter((numeric_results,)), iter((discrete_results,))]
    )

    builder = feature_metadata.BigQueryMetadataBuilder(
        'project_id',
        'bq_dataset_name',
        'bq_table_name',
        bq_client=self.mock_client,
    )

    builder.update_numeric_feature_metadata(feature, default_options)
    builder.update_discrete_feature_metadata(feature, default_options)

    self.assertEqual(self.query_mock.call_count, 2)

    self.assertEqual(feature.mean, all_results['int_feature_mean'])
    self.assertEqual(feature.variance, all_results['int_feature_variance'])
    self.assertEqual(feature.min, all_results['int_feature_min'])
    self.assertEqual(feature.max, all_results['int_feature_max'])
    self.assertIsNone(feature.median)
    self.assertEqual(feature.log_shift, all_results['int_feature_log_shift'])
    self.assertEqual(feature.log_mean, all_results['int_feature_log_mean'])
    self.assertEqual(
        feature.log_variance,
        all_results['int_feature_log_variance'],
    )
    self.assertIsNone(feature.quantiles)

    self.assertEqual(
        feature.cardinality, all_results['int_feature_cardinality']
    )
    self.assertEqual(feature.mode, f'c{all_results["int_feature_cardinality"]}')
    self.assertEqual(
        feature.vocabulary,
        _get_vocab_from_top_count(all_results['int_feature_top_count']),
    )

  def test_update_discrete_and_numeric_feature_metadata_where_clauses(self):
    feature = feature_metadata.FeatureMetadata('int_feature', 0, 'INT64')
    where_clauses = ['int_feature > 0', 'int_feature < 10']
    default_options = feature_metadata.MetadataRetrievalOptions(
        where_clauses=where_clauses
    )

    cardinality = 5
    discrete_results = _get_discrete_data_response('int_feature', cardinality)
    numeric_results = _get_numeric_data_response(
        'int_feature', 5.4321, default_options
    )
    self._update_query_mock_with_results(
        [iter((numeric_results,)), iter((discrete_results,))]
    )

    builder = feature_metadata.BigQueryMetadataBuilder(
        'project_id',
        'bq_dataset_name',
        'bq_table_name',
        bq_client=self.mock_client,
    )

    builder.update_numeric_feature_metadata(feature, default_options)
    builder.update_discrete_feature_metadata(feature, default_options)

    self.assertEqual(self.query_mock.call_count, 2)

    numeric_call, discrete_call = self.query_mock.call_args_list

    # Look for occurrences of the table name but without the where statement.
    table_without_where_re = re.compile(
        rf'FROM\s+`project_id\.bq_dataset_name\.bq_table_name`'
        rf'(?!\s+WHERE\s+.*'
        rf'\(*{where_clauses[0]}\)*\s+and\s+\(*{where_clauses[1]}\)*.*)',
        flags=re.IGNORECASE,
    )

    numeric_query_str = numeric_call.args[0]
    self.assertNotRegex(numeric_query_str, table_without_where_re)

    discrete_query_str = discrete_call.args[0]
    self.assertNotRegex(discrete_query_str, table_without_where_re)

  def test_get_metadata_for_all_features_get_all(self):
    # TODO(b/333154677): Make an integration test for this.

    column_dtypes = [
        'INT64',
        'NUMERIC',
        'BIGNUMERIC',
        'FLOAT64',
        'STRING',
        'BYTES',
        'BOOL',
    ]
    column_names = [f'col{i}' for i in range(len(column_dtypes))]
    self.table_mock.schema = _create_schema_response(
        column_names, column_dtypes
    )
    options = feature_metadata.MetadataRetrievalOptions.get_all()
    col_data = []
    col0_numeric = _get_numeric_data_response('col0', 4.5, options)
    col0_discrete = _get_discrete_data_response('col0', 4)
    col_data.append({**col0_numeric, **col0_discrete})
    col_data.append(_get_numeric_data_response('col1', 3.1, options))
    col_data.append(_get_numeric_data_response('col2', 2.6, options))
    col_data.append(_get_numeric_data_response('col3', 1.0, options))
    col_data.append(_get_discrete_data_response('col4', 3))
    col_data.append(_get_discrete_data_response('col5', 5))
    col_data.append(_get_discrete_data_response('col6', 2))
    self._update_query_mock_with_results([
        iter([col0_numeric]),
        iter([col0_discrete]),
        iter([col_data[1]]),
        iter([col_data[2]]),
        iter([col_data[3]]),
        iter([col_data[4]]),
        iter([col_data[5]]),
        iter([col_data[6]]),
    ])

    builder = feature_metadata.BigQueryMetadataBuilder(
        'project_id',
        'bq_dataset_name',
        'bq_table_name',
        bq_client=self.mock_client,
    )
    all_metadata = builder.get_metadata_for_all_features(options)

    self.mock_client.get_table.assert_called_once_with(
        'project_id.bq_dataset_name.bq_table_name'
    )
    self.assertEqual(
        self.query_mock.call_count,
        len(column_dtypes) + 1,
        msg='Query once for each column other than int and 2 for int.',
    )

    def _check_fields(metadata, name, index, dtype, expected_attributes):
      for field in dataclasses.fields(metadata):
        actual_value = getattr(metadata, field.name)
        if field.name in {'mode', 'vocabulary'}:
          expected_key = f'{name}_top_count'
        else:
          expected_key = f'{name}_{field.name}'
        if field.name == 'name':
          self.assertEqual(actual_value, name, msg=f'Name mismatch for {name}')
        elif field.name == 'index':
          self.assertEqual(
              actual_value, index, msg=f'Index mismatch for {name}'
          )
        elif field.name == 'input_data_type':
          self.assertEqual(
              actual_value, dtype, msg=f'Index mismatch for {name}'
          )
        elif expected_key in expected_attributes:
          if field.name == 'mode':
            vocab = _get_vocab_from_top_count(
                expected_attributes[f'{name}_top_count']
            )
            self.assertEqual(
                actual_value, next(iter(vocab)), msg=f'Mode mismatch for {name}'
            )
          elif field.name == 'vocabulary':
            vocab = _get_vocab_from_top_count(
                expected_attributes[f'{name}_top_count']
            )
            self.assertEqual(
                actual_value, vocab, msg=f'Vocab mismatch for {name}'
            )
          else:
            self.assertEqual(
                actual_value,
                expected_attributes[expected_key],
                msg=f'{field.name} mismatch for {name}',
            )
        else:
          self.assertIsNone(actual_value)

    for i, dtype in enumerate(column_dtypes):
      _check_fields(all_metadata[i], f'col{i}', i, dtype, col_data[i])

  def test_get_metadata_for_all_features_get_none(self):
    column_dtypes = [
        'INT64',
        'NUMERIC',
        'BIGNUMERIC',
        'FLOAT64',
        'STRING',
        'BYTES',
        'BOOL',
    ]
    column_names = [f'col{i}' for i in range(len(column_dtypes))]
    self.table_mock.schema = _create_schema_response(
        column_names, column_dtypes
    )
    options = feature_metadata.MetadataRetrievalOptions.get_none()

    builder = feature_metadata.BigQueryMetadataBuilder(
        'project_id',
        'bq_dataset_name',
        'bq_table_name',
        bq_client=self.mock_client,
    )
    all_metadata = builder.get_metadata_for_all_features(options)
    self.assertEqual(len(all_metadata), len(column_dtypes))

    self.mock_client.get_table.assert_called_once_with(
        'project_id.bq_dataset_name.bq_table_name'
    )

    for name, dtype, feature in zip(column_names, column_dtypes, all_metadata):
      self.assertEqual(name, feature.name)
      self.assertEqual(dtype, feature.input_data_type)


class FeatureMetadataContainerTest(unittest.TestCase):

  def test_feature_metadata_by_names(self):
    numeric_example = feature_metadata.FeatureMetadata(
        'example_numeric', 0, 'FLOAT64'
    )
    string_example = feature_metadata.FeatureMetadata(
        'example_string', 1, 'STRING'
    )
    input_metadata = [numeric_example, string_example]
    container = feature_metadata.FeatureMetadataContainer(input_metadata)
    self.assertDictEqual(
        container.feature_metadata_by_names,
        {'example_numeric': numeric_example, 'example_string': string_example},
    )

  def test_names(self):
    int_example = feature_metadata.FeatureMetadata('int_feature', 0, 'INT64')
    float_example = feature_metadata.FeatureMetadata(
        'float_feature', 1, 'FLOAT64'
    )
    container = feature_metadata.FeatureMetadataContainer(
        [int_example, float_example]
    )
    self.assertTupleEqual(container.names, ('int_feature', 'float_feature'))

  def test_feature_metadata_by_names_raises_for_repeated_names(self):
    first_example = feature_metadata.FeatureMetadata(
        'feature_name', 0, 'FLOAT64'
    )
    same_name = feature_metadata.FeatureMetadata('feature_name', 1, 'STRING')
    input_metadata = [first_example, same_name]
    container = feature_metadata.FeatureMetadataContainer(input_metadata)
    with self.assertRaises(ValueError):
      _ = container.feature_metadata_by_names

  def test_get_metadata_by_name(self):
    numeric_example = feature_metadata.FeatureMetadata(
        'example_numeric', 0, 'FLOAT64'
    )
    string_example = feature_metadata.FeatureMetadata(
        'example_string', 1, 'STRING'
    )
    input_metadata = [numeric_example, string_example]
    container = feature_metadata.FeatureMetadataContainer(input_metadata)
    self.assertEqual(
        container.get_metadata_by_name('example_string'),
        string_example,
    )
    self.assertEqual(
        container.get_metadata_by_name('example_numeric'),
        numeric_example,
    )

  def test_feature_metadata_by_dtypes(self):
    numeric_example = feature_metadata.FeatureMetadata(
        'example_numeric', 0, 'FLOAT64'
    )
    string_example = feature_metadata.FeatureMetadata(
        'example_string', 1, 'STRING'
    )
    numeric_example2 = feature_metadata.FeatureMetadata(
        'second_float', 2, 'FLOAT64'
    )
    input_metadata = [numeric_example, string_example, numeric_example2]
    container = feature_metadata.FeatureMetadataContainer(input_metadata)
    self.assertEqual(len(container.feature_metadata_by_dtypes), 2)
    self.assertEqual(
        container.feature_metadata_by_dtypes['FAKE_DTYPE'],
        [],
        msg='Returns an empty sequence for a missing dtype',
    )
    self.assertListEqual(
        container.feature_metadata_by_dtypes['FLOAT64'],
        [numeric_example, numeric_example2],
    )
    self.assertListEqual(
        container.feature_metadata_by_dtypes['STRING'],
        [string_example],
    )

  def test_get_metadata_by_dtype(self):
    numeric_example = feature_metadata.FeatureMetadata(
        'example_numeric', 0, 'FLOAT64'
    )
    string_example = feature_metadata.FeatureMetadata(
        'example_string', 1, 'STRING'
    )
    numeric_example2 = feature_metadata.FeatureMetadata(
        'second_float', 2, 'FLOAT64'
    )
    input_metadata = [numeric_example, string_example, numeric_example2]
    container = feature_metadata.FeatureMetadataContainer(input_metadata)
    self.assertEqual(
        container.get_metadata_by_dtype('FLOAT64'),
        [numeric_example, numeric_example2],
    )
    self.assertEqual(
        container.get_metadata_by_dtype('STRING'),
        [string_example],
    )

  def test_to_bigquery_schema(self):
    column_dtypes = [
        'INT64',
        'NUMERIC',
        'BIGNUMERIC',
        'FLOAT64',
        'STRING',
        'BYTES',
        'BOOL',
    ]
    all_metadata = [
        feature_metadata.FeatureMetadata(f'col{i}', i, column_dtypes[i])
        for i in range(len(column_dtypes))
    ]
    container = feature_metadata.FeatureMetadataContainer(all_metadata)
    bq_schema = container.to_bigquery_schema()
    for idx, feature_schema in enumerate(bq_schema):
      self.assertEqual(feature_schema.name, f'col{idx}')
      self.assertEqual(feature_schema.field_type, column_dtypes[idx])

  @parameterized.expand((
      ('None', None),
      ('info', 'info'),
      ('warning', 'warning'),
      ('error', 'error'),
      ('raise', 'raise'),
  ))
  def test_equal_names_and_types_true_if_same_name_and_type(
      self, _, difference_method
  ):
    numeric_example = feature_metadata.FeatureMetadata(
        'example_numeric', 0, 'FLOAT64', mean=10.0
    )
    string_example = feature_metadata.FeatureMetadata(
        'example_string', 1, 'STRING', vocabulary=['a']
    )
    numeric_example2 = feature_metadata.FeatureMetadata(
        'example_numeric', 0, 'FLOAT64', mean=20.0
    )
    string_example2 = feature_metadata.FeatureMetadata(
        'example_string', 1, 'STRING', vocabulary=['b']
    )
    three = feature_metadata.FeatureMetadataContainer(
        [numeric_example, string_example]
    )
    two = feature_metadata.FeatureMetadataContainer(
        [numeric_example2, string_example2]
    )
    self.assertTrue(
        two.equal_names_and_types(three, difference_method=difference_method)
    )
    self.assertTrue(
        three.equal_names_and_types(two, difference_method=difference_method)
    )

  def test_equal_names_and_types_false_for_different_number(self):
    numeric_example = feature_metadata.FeatureMetadata(
        'example_numeric', 0, 'FLOAT64'
    )
    string_example = feature_metadata.FeatureMetadata(
        'example_string', 1, 'STRING'
    )
    numeric_example2 = feature_metadata.FeatureMetadata(
        'second_float', 2, 'FLOAT64'
    )
    three = feature_metadata.FeatureMetadataContainer(
        [numeric_example, string_example, numeric_example2]
    )
    two = feature_metadata.FeatureMetadataContainer(
        [numeric_example, string_example]
    )
    self.assertFalse(two.equal_names_and_types(three))
    self.assertFalse(three.equal_names_and_types(two))

  def test_equal_names_and_types_false_for_different_names(self):
    numeric_example = feature_metadata.FeatureMetadata(
        'example_numeric', 0, 'FLOAT64'
    )
    string_example = feature_metadata.FeatureMetadata(
        'example_string', 1, 'STRING'
    )
    different_numeric = feature_metadata.FeatureMetadata(
        'different_numeric', 0, 'FLOAT64'
    )
    example = feature_metadata.FeatureMetadataContainer(
        [numeric_example, string_example]
    )
    different = feature_metadata.FeatureMetadataContainer(
        [different_numeric, string_example]
    )
    self.assertFalse(example.equal_names_and_types(different))
    self.assertFalse(different.equal_names_and_types(example))

  def test_equal_names_and_types_raises_if_specified(self):
    numeric_example = feature_metadata.FeatureMetadata(
        'example_numeric', 0, 'FLOAT64'
    )
    string_example = feature_metadata.FeatureMetadata(
        'example_string', 1, 'STRING'
    )
    different_numeric = feature_metadata.FeatureMetadata(
        'different_numeric', 0, 'FLOAT64'
    )
    example = feature_metadata.FeatureMetadataContainer(
        [numeric_example, string_example]
    )
    different = feature_metadata.FeatureMetadataContainer(
        [different_numeric, string_example]
    )
    with self.assertRaisesRegex(ValueError, r'The following features differ.*'):
      example.equal_names_and_types(different, difference_method='raise')
    with self.assertRaisesRegex(ValueError, r'The following features differ.*'):
      different.equal_names_and_types(example, difference_method='raise')

  @parameterized.expand((
      ('debug', 'debug'),
      ('info', 'info'),
      ('warning', 'warning'),
      ('error', 'error'),
      ('critical', 'critical'),
  ))
  def test_equal_names_and_types_logs_if_specified(self, _, difference_method):
    numeric_example = feature_metadata.FeatureMetadata(
        'example_numeric', 0, 'FLOAT64'
    )
    string_example = feature_metadata.FeatureMetadata(
        'example_string', 1, 'STRING'
    )
    different_numeric = feature_metadata.FeatureMetadata(
        'different_numeric', 0, 'FLOAT64'
    )
    example = feature_metadata.FeatureMetadataContainer(
        [numeric_example, string_example]
    )
    different = feature_metadata.FeatureMetadataContainer(
        [different_numeric, string_example]
    )

    with self.assertLogs(level=difference_method.upper()):
      self.assertFalse(
          example.equal_names_and_types(
              different, difference_method=difference_method
          )
      )
    with self.assertLogs(level=difference_method.upper()):
      self.assertFalse(
          different.equal_names_and_types(
              example, difference_method=difference_method
          )
      )

  def test_equal_names_and_types_false_for_different_types(self):
    numeric_example = feature_metadata.FeatureMetadata(
        'example_numeric', 0, 'NUMERIC'
    )
    string_example = feature_metadata.FeatureMetadata(
        'example_string', 1, 'STRING'
    )
    float_feature = feature_metadata.FeatureMetadata(
        'example_numeric', 0, 'FLOAT64'
    )
    numeric = feature_metadata.FeatureMetadataContainer(
        [numeric_example, string_example]
    )
    float64 = feature_metadata.FeatureMetadataContainer(
        [float_feature, string_example]
    )
    self.assertFalse(numeric.equal_names_and_types(float64))
    self.assertFalse(float64.equal_names_and_types(numeric))

  def test_equal_names_and_types_raises_for_invalid_difference_method(self):
    container = feature_metadata.FeatureMetadataContainer(
        [feature_metadata.FeatureMetadata('example_numeric', 0, 'FLOAT64')]
    )
    with self.assertRaisesRegex(
        ValueError, r'The difference_method .* is not valid.'
    ):
      container.equal_names_and_types(
          container, difference_method='invalid_method'
      )

  def test_config_json_roundtrip(self):
    column_dtypes = [
        'INT64',
        'NUMERIC',
        'BIGNUMERIC',
        'FLOAT64',
        'STRING',
        'BYTES',
        'BOOL',
    ]
    all_metadata = [
        feature_metadata.FeatureMetadata(f'col{i}', i, column_dtypes[i])
        for i in range(len(column_dtypes))
    ]
    container = feature_metadata.FeatureMetadataContainer(all_metadata)

    config_json = json.dumps(container.get_config())
    recreated_container = feature_metadata.FeatureMetadataContainer.from_config(
        json.loads(config_json)
    )
    for expected, actual in zip(container, recreated_container):
      self.assertEqual(expected, actual)


class BigQueryTableMetadataTest(unittest.TestCase):

  def test_config_json_roundtrip(self):
    column_dtypes = [
        'INT64',
        'NUMERIC',
        'BIGNUMERIC',
        'FLOAT64',
        'STRING',
        'BYTES',
        'BOOL',
    ]
    all_metadata = [
        feature_metadata.FeatureMetadata(f'col{i}', i, column_dtypes[i])
        for i in range(len(column_dtypes))
    ]
    container = feature_metadata.BigQueryTableMetadata(
        all_metadata,
        'project_id',
        'bq_dataset_name',
        'bq_table_name',
    )

    config_json = json.dumps(container.get_config())
    recreated_container = feature_metadata.BigQueryTableMetadata.from_config(
        json.loads(config_json)
    )
    for expected, actual in zip(container, recreated_container):
      self.assertEqual(expected, actual)


if __name__ == '__main__':
  unittest.main()
