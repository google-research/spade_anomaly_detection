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

"""Class and functions for loading data for the SPADE algorithm.

Includes functionality for loading local CSVs for testing, as well
as functions for interacting with BigQuery where end users read and upload
table data.
"""

import functools
import os
import pathlib
# TODO(b/247116870): Change to collections when Vertex supports python 3.9
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
from google.cloud import bigquery
from google.cloud import storage
from google.cloud.storage import transfer_manager
import numpy as np
import pandas as pd

from spade_anomaly_detection import parameters
from spade_anomaly_detection.data_utils import bq_dataset
from spade_anomaly_detection.data_utils import bq_utils
from spade_anomaly_detection.data_utils import feature_metadata

import tensorflow as tf

_DATA_ROOT = 'spade_anomaly_detection/example_data/'


def load_dataframe(
    dataset_name: str,
    label_col_index: int = -1,
    filter_label_value: Optional[Union[str, int]] = None,
) -> Sequence[Union[pd.DataFrame, pd.Series]]:
  """Loads csv data located in the ./example_data directory for unit tests.

  Args:
    dataset_name: 'thyroid_labeled' is supported right now.
    label_col_index: Column to use for labels, default to the last column.
    filter_label_value: Value to filter the label column on. Could be a string
      or an integer. If there is no value specified, no filtering will be
      performed.

  Returns:
    A tuple (features, labels), both are dataframes corresponding to the
    features and labels of the dataset respectively.
  """

  file_path = os.path.join(_DATA_ROOT, f'{dataset_name}.csv')

  if dataset_name not in [
      'thyroid_labeled',
      'covertype_pu_labeled',
      'drug_train_pu_labeled',
      'covertype_pnu_10000',
      'covertype_pnu_100000',
  ]:
    raise ValueError(f'Unknown dataset_name: {dataset_name}')

  dataframe = pd.read_csv(file_path, delimiter=',', skiprows=1, index_col=0)

  if len(dataframe.shape) != 2:
    raise ValueError(
        f'{dataset_name} has {len(dataframe.shape)} dimension(s) and is not '
        'equal to the 2 dimensional requirement.'
    )

  if label_col_index >= dataframe.shape[1]:
    raise ValueError(
        f'Label column {label_col_index} is outside of the column range.'
    )

  if filter_label_value is not None:
    dataframe = dataframe[
        dataframe.iloc[:, label_col_index] == filter_label_value
    ]

  features = dataframe.drop(dataframe.columns[label_col_index], axis=1)
  labels = dataframe.iloc[:, label_col_index].astype(int)

  return features, labels


def load_tf_dataset_from_csv(
    dataset_name: str,
    label_col_index: int = -1,
    batch_size: Optional[int] = None,
    filter_label_value: Optional[Any] = None,
    return_feature_count: bool = False,
) -> Union[tf.data.Dataset, Tuple[tf.data.Dataset, int]]:
  """Loads a TensorFlow dataset from the ./example_data directory.

  Args:
    dataset_name: The name of the CSV file to be loaded.
    label_col_index: Column to use for labels, default to the last column.
    batch_size: The number of records the generator returns in a single call.
      Tune this depending on the size of the machines memory the algorithm is
      being executed on. The default value is None, which means that the dataset
      is not batched. In this case, when iterating through the dataset, it will
      yield one record per call instead of a batch of records.
    filter_label_value: Value to filter the label column on. Could be a string
      or an integer.
    return_feature_count: If True, returns a tuple of the Dataset and the number
      of features.

  Returns:
    A TensorFlow dataset, or a tuple containing a Tensorflow Dataset and the
      number of features,
  """
  features, labels = load_dataframe(
      dataset_name, label_col_index, filter_label_value
  )

  feature_tensors = tf.convert_to_tensor(features, dtype=tf.dtypes.float32)

  label_tensors = tf.convert_to_tensor(labels, dtype=tf.dtypes.int8)

  tf_dataset = tf.data.Dataset.from_tensor_slices(
      (feature_tensors, label_tensors)
  )

  if batch_size is not None:
    tf_dataset = tf_dataset.batch(batch_size)

  if return_feature_count:
    return tf_dataset, len(features.columns)
  return tf_dataset


class DataLoader:
  """Contains methods for interacting with BigQuery using RunnerParameters."""

  def __init__(self, runner_parameters: parameters.RunnerParameters):
    self.input_feature_metadata: feature_metadata.BigQueryTableMetadata = None
    self.runner_parameters = runner_parameters

    self.table_parts = bq_utils.BQTablePathParts.from_full_path(
        self.runner_parameters.input_bigquery_table_path
    )

  @property
  def num_features(self) -> int:
    """Returns the number of features in the dataset."""
    if not self.input_feature_metadata:
      raise ValueError(
          'Input feature metadata has not been instantiated yet, '
          'ensure that you have made a call to '
          'load_tf_dataset_from_bigquery() before this method '
          'is called.'
      )
    # Now that the data is loaded, access the metadata and get the number of
    # features. We get this from the number of elements in the metadata sequence
    # and subtracting 1 for the label column.
    return len(self.input_feature_metadata._metadata_sequence) - 1  # pylint:disable=protected-access

  def get_label_thresholds(
      self,
      bigquery_table_path: str,
      where_statements: Optional[List[str]] = None,
  ) -> Mapping[str, float]:
    """Computes positive and negative thresholds based on label ratios.

    This method is useful for setting percentile thresholds when performing
    inference on the OCC ensemble. Feature vectors can be labeled as normal
    or anomalous depending on these values. This method requires that the label
    column, and positive and negative values are set in RunnerParameters.

    Args:
      bigquery_table_path: Full BigQuery table path to compute label ratios on.
      where_statements: A list of valid SQL where statements.

    Returns:
      A dictionary containing 'positive_threshold' and 'negative_threshold'.
    """
    if where_statements is not None:
      additional_where_statements = ' AND ' + ' AND '.join(where_statements)
    else:
      additional_where_statements = ''

    with bigquery.Client(
        project=self.table_parts.project_id
    ) as bigquery_client:
      label_count_query = (
          f'SELECT {self.runner_parameters.label_col_name},'
          f' count({self.runner_parameters.label_col_name}) AS record_count'
          f' FROM {bigquery_table_path}'
          f' WHERE {self.runner_parameters.label_col_name} <>'
          f' {self.runner_parameters.unlabeled_data_value}'
          f' {additional_where_statements}'
          f' GROUP BY {self.runner_parameters.label_col_name}'
      )
      # Note: Parameterization was explored, however there were severe
      # limitations including the inability to parameterize the table name and
      # the unlabeled data value. The latter is due to the fact that the label
      # column name and the label vale can be strings or integers, and in order
      # to make SQL comparisons, these must be the same type.
      label_count_query_job = bigquery_client.query(query=label_count_query)

    label_counts = label_count_query_job.result().to_dataframe()
    logging.info('Label counts: %s', label_counts)

    positive_count = label_counts[
        label_counts[self.runner_parameters.label_col_name]
        == self.runner_parameters.positive_data_value
    ].iat[0, -1]

    labeled_data_record_count = label_counts['record_count'].sum()

    positive_threshold = 100 * (positive_count / labeled_data_record_count)
    label_thresholds = {
        'positive_threshold': positive_threshold,
        'negative_threshold': 100 - positive_threshold,
    }

    if self.runner_parameters.verbose:
      logging.info('Computed label thresholds: %s', label_thresholds)

    return label_thresholds

  def get_query_record_result_length(
      self,
      input_path: str,
      where_statements: Optional[List[str]] = None,
  ) -> int:
    """Returns an integer representing the total number of rows in the query.

    Args:
      input_path: BigQuery string in the format 'project.dataset.table'.
      where_statements: A list of valid SQL where statements.

    Returns:
      An integer representing the number of rows in the query.
    """

    with bigquery.Client(
        project=self.table_parts.project_id
    ) as big_query_client:
      query = f'SELECT count(*) FROM `{input_path}` '
      if where_statements:
        query += 'WHERE ' + ' AND '.join(where_statements)

      query_job = big_query_client.query(query)
      result = query_job.result().to_dataframe().values

    return int(result)

  def _preprocess(
      self,
      batch: Mapping[str, tf.Tensor],
      label_col_name: str,
      convert_label_to_int: bool = False,
      convert_features_to_float64: bool = False,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Reshapes the results returned by the dataloader utility.

    Method that is used as a mapping function on a TensorFlow dataset. Instead
    of returning a single matrix, we adjust the results to be two matrices of
    (M, Nx), (M, 1). Where M is the numnber of training examples and Nx is the
    dimension of the feature vector.

    Args:
      batch: A batch of tensors in the format of (key, value) where the key is a
        string defingin the column name in BigQuery.
      label_col_name: The name of the label column in the input BigQuery table.
      convert_label_to_int: Set to True to cast the contents of the label column
        to integer.
      convert_features_to_float64: Set to True to cast the contents of the
        features columns to float64.

    Returns:
      A tuple of Tensors in the format (features, labels)
    """
    feature_matrix = []
    label_tensor = None

    for key, value in batch.items():
      if key == label_col_name:
        label_tensor = value
        continue

      feature_matrix.append(value)

    if label_tensor is None:
      raise ValueError('Label column not found in the input table.')
    if convert_label_to_int:
      label_tensor = tf.cast(label_tensor, tf.int64)

    feature_matrix = tf.stack(feature_matrix, axis=1)
    if convert_features_to_float64:
      feature_matrix = tf.cast(feature_matrix, tf.float64)

    return feature_matrix, label_tensor

  def load_tf_dataset_from_bigquery(
      self,
      input_path: str,
      label_col_name: str,
      where_statements: Optional[List[str]] = None,
      ignore_columns: Optional[Sequence[str]] = None,
      batch_size: Optional[int] = None,
      label_column_filter_value: Optional[int] = None,
      convert_features_to_float64: bool = False,
  ) -> tf.data.Dataset:
    """Loads a TensorFlow dataset from a BigQuery Table.

    Args:
      input_path: BigQuery string in the format 'project.dataset.table'.
      label_col_name: The name of the label column in the input BigQuery table.
      where_statements: A sequence of SQL where clauses that you wish to apply
        when selecting data from BigQuery.
      ignore_columns: A sequence of columns that will not be included in the
        returned dataset. Default is None, meaning all columns are included.
      batch_size: The number of records the generator returns in a single call.
        Tune this depending on the size of the machines memory the algorithm is
        being executed on. The default value is None, which means that the
        dataset is not batched. In this case, when iterating through the
        dataset, it will yield one record per call instead of a batch of
        records.
      label_column_filter_value: An integer used when filtering the label column
        values. No value will result in all data returned from the table.
      convert_features_to_float64: Set to True to cast the contents of the
        features columns to float64.

    Returns:
      A TensorFlow dataset.
    """
    metadata_retrieval_options = None

    metadata_builder = None

    if label_column_filter_value is not None:
      where_statements = (
          list() if where_statements is None else where_statements
      )
      where_statements.append(f'{label_col_name} = {label_column_filter_value}')

    if ignore_columns is not None:
      metadata_builder = feature_metadata.BigQueryMetadataBuilder(
          project_id=self.table_parts.project_id,
          bq_dataset_name=self.table_parts.bq_dataset_name,
          bq_table_name=self.table_parts.bq_table_name,
          ignore_columns=ignore_columns,
      )

    if where_statements:
      metadata_retrieval_options = feature_metadata.MetadataRetrievalOptions(
          where_clauses=where_statements
      )

    tf_dataset, metadata = bq_dataset.get_dataset_and_metadata_for_table(
        table_path=input_path,
        batch_size=batch_size,
        with_mask=False,
        drop_remainder=True,
        metadata_options=metadata_retrieval_options,
        metadata_builder=metadata_builder,
    )

    self.input_feature_metadata = metadata

    preprocess_fn = functools.partial(
        self._preprocess,
        label_col_name=label_col_name,
        convert_label_to_int=False,
        convert_features_to_float64=convert_features_to_float64,
    )
    return tf_dataset.map(
        lambda batch: preprocess_fn(batch),  # pylint:disable=unnecessary-lambda
        num_parallel_calls=tf.data.AUTOTUNE,
    )

  def construct_schema_from_dataframe(
      self, df: pd.DataFrame
  ) -> Sequence[bigquery.SchemaField]:
    """Constructs a BigQuery schema from a Pandas dataframe."""

    def get_type(dtype: np.dtype) -> str:
      if dtype in [np.float32, np.float64]:
        return 'FLOAT'
      elif dtype in [np.int32, np.int64, np.int_]:
        return 'INTEGER'
      elif dtype in [np.bool, np.bool_]:
        return 'BOOLEAN'
      else:
        raise ValueError(f'Unsupported data type: {dtype}')

    columns = df.columns
    dtypes = df.dtypes
    return [
        bigquery.SchemaField(
            name=col, field_type=get_type(dtype), mode='NULLABLE'
        )
        for (col, dtype) in zip(columns, dtypes)
    ]

  def construct_dataframe_from_features(
      self,
      features: np.ndarray,
      labels: np.ndarray,
      convert_label_to_bool: bool = False,
  ) -> pd.DataFrame:
    """Constructs a dataframe from features and labels.

    Args:
      features: The features as a 2D Numpy array.
      labels: The labels as a 1D Numpy array.
      convert_label_to_bool: If True, convert the label to boolean.

    Returns:
      A dataframe containing the features and labels.
    """
    if not self.input_feature_metadata:
      raise ValueError(
          'Input feature metadata has not been instantiated yet, '
          'ensure that you have made a call to '
          'load_tf_dataset_from_bigquery() before this method '
          'is called.'
      )
    combined_data = np.concatenate(
        [features, labels.reshape(len(features), 1)], axis=1
    )

    column_names = list(self.input_feature_metadata.names)
    column_names.remove(self.runner_parameters.label_col_name)
    column_names.append(self.runner_parameters.label_col_name)

    complete_dataframe = pd.DataFrame(data=combined_data, columns=column_names)

    if convert_label_to_bool:
      # Adjust label column type so that users can go straight to BigQuery or
      # AutoML without having to adjust data. Both of these products require a
      # boolean or string target column, not integer.
      complete_dataframe[self.runner_parameters.label_col_name] = (
          complete_dataframe[self.runner_parameters.label_col_name].astype(
              'bool'
          )
      )
    return complete_dataframe

  def parse_gcs_folder_uri(self, gcs_uri: str) -> tuple[str, str]:
    """Parses a GCS URI of a folder."""
    # Get ID of GCS bucket from URI. The URIs are of the form
    # 'gs://bucket-name/folder/path'.
    gcs_prefix = 'gs://'
    uri_without_prefix = gcs_uri.split(gcs_prefix)[-1]
    bucket_name = uri_without_prefix.partition('/')[0]
    folder_name = uri_without_prefix.partition('/')[-1]
    return bucket_name, folder_name

  def upload_data_to_local_disk(
      self,
      file_path: str,
      features: np.ndarray,
      labels: np.ndarray,
      convert_label_to_bool: bool = False,
      use_parquet: bool = True,
  ):
    """Saves a batch of pseudo-labeled data to local disk on the VM."""
    complete_dataframe = self.construct_dataframe_from_features(
        features=features,
        labels=labels,
        convert_label_to_bool=convert_label_to_bool,
    )
    if use_parquet:
      # Send the dataframe to the file. Compression can be 'snappy', 'gzip',
      # or 'lz4'. Anecdotally, 'gzip' is 4X slower than 'snappy' or 'lz4'.
      complete_dataframe.to_parquet(
          file_path, index=False, compression='snappy'
      )
    else:
      # Send the dataframe to the file.
      complete_dataframe.to_csv(file_path, header=True, index=False)

  def _construct_file_paths_for_gcs(
      self,
      source_folder: str,
  ) -> Sequence[str]:
    """Constructs the file paths for uploading to GCS.

    Args:
      source_folder: Name of local folder from where files are to be uploaded.

    Returns:
      List of file paths of files in the source folder/
    """
    # Generate a list of paths (in string form) relative to the `directory`.
    # This can be done in a single list comprehension, but is expanded into
    # multiple lines here for clarity.

    # First, recursively get all files in `directory` as Path objects.
    directory_as_path_obj = pathlib.Path(source_folder)
    paths = directory_as_path_obj.rglob('*')

    # Filter so the list only includes files, not directories themselves.
    file_paths = [path for path in paths if path.is_file()]

    # These paths are relative to the current working directory. Next, make them
    # relative to `directory`
    relative_paths = [path.relative_to(source_folder) for path in file_paths]

    # Finally, convert them all to strings.
    string_paths = [str(path) for path in relative_paths]
    return string_paths

  def upload_folder_to_gcs(
      self,
      source_folder: str,
      gcs_uri: str,
      num_workers: int = 8,
      use_parquet: bool = True,
  ):
    """Uploads all the files in a local folder to GCS."""
    file_ext = 'parquet' if use_parquet else 'csv'

    # Parse the GCS URI. The URIs are of the form
    # 'gs://bucket-name/folder/path'.
    bucket_name, folder_name = self.parse_gcs_folder_uri(gcs_uri)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Make the folders on GCS if they don't exist.
    if not tf.io.gfile.exists(gcs_uri):
      tf.io.gfile.makedirs(gcs_uri)

    # Generate a list of paths (in string form) relative to the `directory`.
    string_paths = self._construct_file_paths_for_gcs(source_folder)
    logging.info('Found %d %s files.', len(string_paths), file_ext)

    # Start the upload.
    results = transfer_manager.upload_many_from_filenames(
        bucket,
        string_paths,
        blob_name_prefix=f'{folder_name}/',
        source_directory=source_folder,
        max_workers=num_workers,
    )

    upload_count = 0
    for name, result in zip(string_paths, results):
      # The results list is either `None` or an exception for each filename in
      # the input list, in order.
      if isinstance(result, Exception):
        logging.error('Failed to upload %s due to exception: %s', name, result)
      else:
        upload_count += 1
    logging.info(
        'Uploaded %d %s files to %s in folder %s.',
        upload_count,
        file_ext,
        bucket.name,
        folder_name,
    )

  def _get_bigquery_job_config(
      self,
      schema: Sequence[bigquery.SchemaField],
      write_disposition: Optional[str] = 'WRITE_TRUNCATE',
      use_parquet: bool = True,
  ) -> bigquery.LoadJobConfig:
    """Gets the BigQuery job configuration."""
    source_format = (
        bigquery.SourceFormat.PARQUET
        if use_parquet
        else bigquery.SourceFormat.CSV
    )
    if use_parquet:
      job_config = bigquery.LoadJobConfig(
          schema=schema,
          source_format=source_format,
          write_disposition=write_disposition,
      )
    else:
      job_config = bigquery.LoadJobConfig(
          schema=schema,
          source_format=source_format,
          skip_leading_rows=1,  # Skip the header row.
          write_disposition=write_disposition,
      )
    return job_config

  def load_bigquery_table_from_gcs_uri(
      self,
      gcs_uri: str,
      table_name: str,
      schema: Sequence[bigquery.SchemaField],
      write_disposition: Optional[str] = 'WRITE_TRUNCATE',
      use_parquet: bool = True,
  ) -> None:
    """Loads a BigQuery table from a GCS URI.

    CSV load job quotas are here:
    https://cloud.google.com/bigquery/quotas#load_jobs

    Args:
      gcs_uri: GCS URI (potentially with wildcard) from which to load data.
      table_name: Name of target BigQuery table.
      schema: Schema of the target BigQuery table.
      write_disposition: Whether to overwrite the table (WRITE_TRUNCATE) or
        append to the table (WRITE_APPEND). Default is `WRITE_TRUNCATE`.
      use_parquet: If True, use parquet files on GCS, else use CSV files.
    """
    with bigquery.Client(
        project=self.table_parts.project_id
    ) as big_query_client:
      job_config = self._get_bigquery_job_config(
          schema=schema,
          write_disposition=write_disposition,
          use_parquet=use_parquet,
      )
      file_ext = 'parquet' if use_parquet else 'csv'
      load_job = big_query_client.load_table_from_uri(
          f'{gcs_uri}/*.{file_ext}', table_name, job_config=job_config
      )
      load_job.result()  # Waits for the job to complete.

    destination_table = big_query_client.get_table(table_name)
    logging.info(
        'Loaded %d rows from %s files to table %s.',
        destination_table.num_rows,
        file_ext,
        table_name,
    )

  def upload_dataframe_as_bigquery_table(
      self,
      features: np.ndarray,
      labels: np.ndarray,
  ) -> None:
    """Uploads the dataframe to BigQuery, create or replace table.

    Args:
      features: Numpy array of features.
      labels: Numpy array of labels.
    """
    if not self.input_feature_metadata:
      raise ValueError(
          'Input feature metadata has not been instantiated yet, '
          'ensure that you have made a call to '
          'load_tf_dataset_from_bigquery() before this method '
          'is called.'
      )
    combined_data = np.concatenate(
        [features, labels.reshape(len(features), 1)], axis=1
    )

    column_names = list(self.input_feature_metadata.names)
    column_names.remove(self.runner_parameters.label_col_name)
    column_names.append(self.runner_parameters.label_col_name)

    complete_dataframe = pd.DataFrame(data=combined_data, columns=column_names)

    # Adjust label column type so that users can go straight to BigQuery or
    # AutoML without having to adjust data. Both of these products require a
    # boolean or string target column, not integer.
    complete_dataframe[self.runner_parameters.label_col_name] = (
        complete_dataframe[self.runner_parameters.label_col_name].astype('bool')
    )

    with bigquery.Client(
        project=self.table_parts.project_id
    ) as big_query_client:
      job_config = bigquery.LoadJobConfig(
          # Creates or replaces an existing table.
          write_disposition='WRITE_TRUNCATE',
      )
      # In this configuration, the schema will be auto-detected.
      big_query_client.load_table_from_dataframe(
          dataframe=complete_dataframe,
          destination=self.runner_parameters.output_bigquery_table_path,
          job_config=job_config,
      )
    if self.runner_parameters.verbose:
      logging.info(
          'Uploaded pseudo-labeled data to %s',
          self.runner_parameters.output_bigquery_table_path,
      )

  def get_metadata_for_table(
      self,
      bigquery_client: Optional[bigquery.Client] = None,
      metadata_options: Optional[
          feature_metadata.MetadataRetrievalOptions
      ] = None,
      metadata_builder: Optional[
          feature_metadata.BigQueryMetadataBuilder
      ] = None,
  ) -> feature_metadata.BigQueryTableMetadata:
    """Gets the metadata and dataset for a BigQuery table.

    Args:
      bigquery_client: The BigQuery Client object to use for getting the
        metadata.
      metadata_options: The metadata retrieval options to use.
      metadata_builder: The metadata builder to use to get the metadata.

    Returns:
      The metadata for the specified table.
    """
    if not bigquery_client:
      bigquery_client = bigquery.Client(project=self.table_parts.project_id)

    if not metadata_options:
      metadata_options = feature_metadata.MetadataRetrievalOptions.get_none()

    if not metadata_builder:
      metadata_builder = (
          feature_metadata.BigQueryMetadataBuilder.from_table_parts(
              self.table_parts, bq_client=bigquery_client
          )
      )

    all_metadata = metadata_builder.get_metadata_for_all_features(
        metadata_options
    )
    # TODO(b/333154677): Refactor code so that this extra call is not needed.
    bq_dataset.update_tf_data_types_from_bq_data_types(all_metadata)

    return all_metadata
