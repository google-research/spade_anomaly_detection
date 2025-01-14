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

"""Implements a CSV data loader."""

import collections
import csv
import dataclasses
import os
from typing import Callable, Dict, Final, Mapping, Optional, Sequence, Tuple, cast

from absl import logging
from google.cloud import storage
import numpy as np
import pandas as pd
from spade_anomaly_detection import data_loader
from spade_anomaly_detection import parameters
import tensorflow as tf


_FEATURES_TYPE: Final[str] = 'FLOAT64'
_SOURCE_LABEL_TYPE: Final[str] = 'STRING'
_SOURCE_LABEL_DEFAULT_VALUE: Final[str] = '-1'
_LABEL_TYPE: Final[str] = 'INT64'
_STRING_TO_INTEGER_LABEL_MAP: dict[str | int, int] = {
    1: 1,
    0: 0,
    -1: -1,
    '': -1,
    '-1': -1,
    '0': 0,
    '1': 1,
    'positive': 1,
    'negative': 0,
    'unlabeled': -1,
}

# Setting the shuffle buffer size to 1M seems to be necessary to get the CSV
# reader to provide a diversity of data to the model.
_SHUFFLE_BUFFER_SIZE: Final[int] = 1_000_000
_SPLIT_CHAR: Final[str] = ','

LabelColumnType = str | list[str] | int | list[int] | None


def _get_header_from_input_file(inputs_file: str) -> str:
  """Gets the header from a file of data inputs."""
  # Separate this logic so that it can be mocked easily in unit tests.
  with tf.io.gfile.GFile(inputs_file, mode='r') as f:
    header = f.readline()  # Assume that first line is the header.
  return header


def _list_files(
    bucket_name: str,
    input_blob_prefix: str,
    input_blob_suffix: Optional[str] = None,
) -> Sequence[str]:
  """Lists all files in `bucket_name` matching a prefix and an optional suffix.

  Args:
    bucket_name: GCS bucket in which to list files.
    input_blob_prefix: Prefix of files to list (inside `bucket_name`).
    input_blob_suffix: Suffix of files to list (inside `bucket_name`).

  Returns:
    Listed files, formatted as "gs://`bucket_name`/`blob.name`.
  """
  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)

  filenames = []
  blobs = bucket.list_blobs(prefix=input_blob_prefix)
  for blob in blobs:
    if blob.name.endswith('/'):
      # Skip any folders.
      continue
    filenames.append('gs://' + bucket_name + '/' + blob.name)
  if input_blob_suffix is not None:
    filenames = [
        filename
        for filename in filenames
        if filename.endswith(input_blob_suffix)
    ]
  return filenames


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str, str]:
  """Parses a GCS URI into bucket name, prefix and suffix.

  Args:
    gcs_uri: GCS URI to parse.

  Returns:
    Bucket name, prefix and suffix.

  Raises:
    ValueError: If the GCS URI is not valid.
  """
  gcs_uri_prefix = 'gs://'
  if not gcs_uri.startswith(gcs_uri_prefix):
    raise ValueError(f'GCS URI {gcs_uri} does not start with "gs://".')
  gcs_uri = gcs_uri.removeprefix(gcs_uri_prefix)
  bucket_name = gcs_uri.split('/')[0]
  rest = gcs_uri.removeprefix(f'{bucket_name}/')
  split = rest.split('*')
  if len(split) == 1:
    # Paths must be to folders, not files.
    rest = f'{rest}/' if not rest.endswith('/') else rest
    return bucket_name, rest, ''
  elif len(split) == 2:
    return bucket_name, split[0], split[1]
  else:
    raise ValueError(f"GCS URI {gcs_uri} has more than one wildcard ('*').")


@dataclasses.dataclass
class ColumnNamesInfo:
  """Information about the column names."""

  header: str
  label_column_name: str
  column_names_dict: collections.OrderedDict[str, str]
  num_features: int

  @classmethod
  def from_inputs_file(
      cls,
      inputs_file: str,
      label_column_name: str,
  ) -> 'ColumnNamesInfo':
    """Reads the column names information from one CSV file of input data.

    Inputs are in multiple CSV files on GCS. Get the column names from one of
    those CSV files. The returned `ColumnNamesInfo` instance contains the
    inputs CSV file header, the label column name and a dictionary of column
    names to column types.

    Args:
      inputs_file: Inputs file from which to read the column names.
      label_column_name: The name of the label column.

    Returns:
      An instance of ColumnNamesInfo containing the original header, the name of
      the label column, a dictionary of column names to column types and the
      number of features.
    """
    header = _get_header_from_input_file(inputs_file=inputs_file)
    header = header.replace('\n', '')
    all_columns = header.split(_SPLIT_CHAR)
    if label_column_name not in all_columns:
      raise ValueError(
          f'Label column {label_column_name} not found in the header: {header}'
      )
    features_types = [_FEATURES_TYPE] * len(all_columns)
    column_names_dict = collections.OrderedDict(
        zip(all_columns, features_types)
    )
    column_names_dict[label_column_name] = _SOURCE_LABEL_DEFAULT_VALUE
    num_features = len(all_columns) - 1
    return ColumnNamesInfo(
        column_names_dict=column_names_dict,
        header=header,
        label_column_name=label_column_name,
        num_features=num_features,
    )


@dataclasses.dataclass
class InputFilesMetadata:
  """Metadata about the set of CSV files containing the input data.

  Attributes:
    location_prefix: Prefix of location pattern on GCS where the CSV files
      containing input data are located. All CSV files recursively matching this
      patten will be selected.
    files: Names for each of the found CSV files.
    column_names_info: Instance of ColumnNamesInfo Dataclass.
  """

  location_prefix: str
  files: Sequence[str]
  column_names_info: ColumnNamesInfo


class CsvDataLoader:
  """Contains methods for interacting with CSV files using RunnerParameters."""

  def __init__(self, runner_parameters: parameters.RunnerParameters):
    self.runner_parameters = runner_parameters
    if self.runner_parameters.data_input_gcs_uri is None:
      raise ValueError(
          'Data input GCS URI is not set in the runner parameters. Please set '
          'the `data_input_gcs_uri` field in the runner parameters.'
      )
    self._label_counts = None
    self._last_read_metadata = None

    self.all_labels: Final[list[int] | list[str]] = [
        self.runner_parameters.positive_data_value,
        self.runner_parameters.negative_data_value,
        self.runner_parameters.unlabeled_data_value,
    ]
    # Add any labels that are not already in the map.
    _STRING_TO_INTEGER_LABEL_MAP[self.runner_parameters.positive_data_value] = 1
    _STRING_TO_INTEGER_LABEL_MAP[self.runner_parameters.negative_data_value] = 0
    _STRING_TO_INTEGER_LABEL_MAP[
        self.runner_parameters.unlabeled_data_value
    ] = -1

    # Construct a label remap from string labels to integers. The table is not
    # necessary for the case when the labels are all integers. But instead of
    # checking if the labels are all integers, we construct the table and use
    # it only if labels are strings. The table is lightweight so it does not
    # add overhead.
    str_labels: list[str] = [str(l) for l in self.all_labels]
    # Construct a label remap from string labels to integers.
    if self.runner_parameters.labels_are_strings:
      orig_labels: list[int] = [
          self.convert_str_to_int(l) for l in self.all_labels
      ]
    else:
      orig_labels: list[int] = cast(list[int], self.all_labels)
    # Key is always a string.
    orig_map: dict[str, int] = dict(zip(str_labels, orig_labels))
    # Add the empty string to the label remap table. `None` is not added because
    # the Data Loader will default to '' if the label is None.
    orig_map[''] = cast(
        int,
        self.convert_str_to_int(self.runner_parameters.unlabeled_data_value)
        if self.runner_parameters.labels_are_strings
        else self.runner_parameters.unlabeled_data_value,
    )
    # Convert to the Tensorflow lookup table.
    self._label_remap_table: Final[tf.lookup.StaticHashTable] = (
        self._get_label_remap_table(labels_mapping=orig_map)
    )

  @property
  def label_counts(self) -> Dict[int | str, int]:
    """Returns the label counts."""
    if not self._label_counts:
      raise ValueError(
          'Label counts have not been computed yet, ensure that you have made '
          'a call to load_tf_dataset_from_csv() before this property is called.'
      )
    return self._label_counts

  def get_inputs_metadata(
      self,
      bucket_name: str,
      location_prefix: str,
      location_suffix: str,
      label_column_name: str,
  ) -> 'InputFilesMetadata':
    """Gets information about the CSVs containing the input data.

    Args:
      bucket_name: Name of the GCS bucket where the CSV files are located.
      location_prefix: The prefix of location of the CSV files, excluding any
        trailing unique identifiers.
      location_suffix: The suffix of location of the CSV files (e.g. '.csv').
      label_column_name: The name of the label column.

    Returns:
      Return a InputFilesMetadata instance.
    """
    # Get the names of the CSV files containing the input data.
    csv_filenames = _list_files(
        bucket_name=bucket_name,
        input_blob_prefix=location_prefix,
        input_blob_suffix=location_suffix,
    )
    logging.info(
        'Collecting metadata for %d files at %s',
        len(csv_filenames),
        location_prefix,
    )
    # Get information about the columns.
    column_names_info = ColumnNamesInfo.from_inputs_file(
        csv_filenames[0],
        label_column_name,
    )
    logging.info(
        'Obtained metadata for data with CSV prefix %s (number of features=%d)',
        location_prefix,
        column_names_info.num_features,
    )
    return InputFilesMetadata(
        location_prefix=location_prefix,
        files=csv_filenames,
        column_names_info=column_names_info,
    )

  @classmethod
  def _get_filter_by_label_value_func(
      cls,
      label_column_filter_value: LabelColumnType,
      exclude_label_value: bool = False,
  ) -> Callable[[tf.Tensor, tf.Tensor], bool]:
    """Returns a function that filters a record based on the label column value.

    This function will return a function that filters a record based on the
    label column value whether it is a string or an integer.

    Args:
      label_column_filter_value: The value of the label column to use as a
        filter. If None, all records are included.
      exclude_label_value: If True, exclude records with the label column value.
        If False, include records with the label column value.

    Returns:
      A function that returns True if the label column value is equal to the
      label_column_filter_value parameter(s). If exclude_label_value is True,
      the function returns True if the label column value is not equal to the
      label_column_filter_value parameter(s).
    """

    def filter_func(features: tf.Tensor, label: tf.Tensor) -> bool:  # pylint: disable=unused-argument
      if label_column_filter_value is None:
        return True
      if (
          isinstance(label, tf.Tensor)
          and label.dtype == tf.dtypes.string
          or isinstance(label, np.ndarray)
          and label.dtype == np.str_
          or isinstance(label, list)
          and isinstance(label[0], str)
      ):
        # If the label dtype is string, convert it to an integer dtype,
        # *assuming* that the string is composed of only digits.
        try:
          label = tf.strings.to_number(
              label, tf.dtypes.as_dtype(_LABEL_TYPE.lower())
          )
        except tf.errors.InvalidArgumentError as e:
          logging.exception(
              'Failed to convert label %s to integer: %s', label, e
          )
          raise e
      label_cast = tf.cast(label[0], tf.dtypes.as_dtype(_LABEL_TYPE.lower()))
      label_column_filter_value_cast = tf.cast(
          label_column_filter_value, label_cast.dtype
      )
      broadcast_equal = tf.equal(label_column_filter_value_cast, label_cast)
      broadcast_all = tf.reduce_any(broadcast_equal)
      if exclude_label_value:
        return tf.logical_not(broadcast_all)
      return broadcast_all

    return filter_func

  @classmethod
  def convert_str_to_int(cls, value: str) -> int:
    """Converts a string integer label to an integer label."""
    if value in _STRING_TO_INTEGER_LABEL_MAP:
      return _STRING_TO_INTEGER_LABEL_MAP[value]
    else:
      raise ValueError(
          f'Label {value} of type {type(value)} is not a string integer or '
          'mappable to an integer.'
      )

  @classmethod
  def _get_label_remap_table(
      cls, labels_mapping: dict[str, int]
  ) -> tf.lookup.StaticHashTable:
    """Returns a label remap table that converts string labels to integers."""
    keys_tensor = tf.constant(
        list(labels_mapping.keys()),
        dtype=tf.dtypes.as_dtype(_SOURCE_LABEL_TYPE.lower()),
    )
    vals_tensor = tf.constant(
        list(labels_mapping.values()),
        dtype=tf.dtypes.as_dtype(_LABEL_TYPE.lower()),
    )
    label_remap_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
        default_value=-1,
    )
    return label_remap_table

  def remap_label(self, label: str | tf.Tensor) -> int | tf.Tensor:
    """Remaps the label to an integer."""
    if isinstance(label, str) or (
        isinstance(label, tf.Tensor) and label.dtype == tf.dtypes.string
    ):
      return self._label_remap_table.lookup(label)
    return label

  def load_tf_dataset_from_csv(
      self,
      input_path: str,
      label_col_name: str,
      batch_size: Optional[int] = None,
      label_column_filter_value: LabelColumnType = None,
      exclude_label_value: bool = False,
  ) -> tf.data.Dataset:
    """Convert multiple CSV files to a tf.data.Dataset.

    Multiple CSV files are read from a specified location. They are streamed
    using a tf.data.Dataset.

    Args:
      input_path: The path to the CSV files.
      label_col_name: The name of the label column.
      batch_size: The batch size to use for the dataset. If None, the batch size
        will be set to 1.
      label_column_filter_value: The value of the label column to use as a
        filter. If None, all records are included.
      exclude_label_value: If True, exclude records with the label column value.
        If False, include records with the label column value.

    Returns:
      A tf.data.Dataset.
    """
    bucket, prefix, suffix = _parse_gcs_uri(input_path)
    # Since we are reading a new set of CSV files, we need to get the metadata
    # again.
    self._last_read_metadata = self.get_inputs_metadata(
        bucket_name=bucket,
        location_prefix=prefix,
        location_suffix=suffix,
        label_column_name=label_col_name,
    )
    logging.info('Last read metadata: %s', self._last_read_metadata)
    # Get the names of the CSV files containing the data and other metadata
    filenames = self._last_read_metadata.files
    logging.info('Found %d CSV files.', len(filenames))

    column_names = list(
        self._last_read_metadata.column_names_info.column_names_dict.keys()
    )
    # Setting dtypes in the column defaults makes the columns required.
    # TODO(sinharaj): Add support for optional columns.
    column_defaults = [
        d.lower()
        for d in list(
            self._last_read_metadata.column_names_info.column_names_dict.values()
        )
    ]
    logging.info('column_defaults: %s', column_defaults)

    # Construct a single dataset out of multiple CSV files.
    # TODO(sinharaj): Remove the determinism after testing.
    dataset = tf.data.experimental.make_csv_dataset(
        filenames,
        batch_size=1,  # Initial Dataset is created with one sample at a time.
        column_names=column_names,
        column_defaults=column_defaults,
        label_name=label_col_name,
        select_columns=None,
        field_delim=_SPLIT_CHAR,
        use_quote_delim=True,
        na_value='',
        header=True,
        num_epochs=1,
        shuffle=False,
        shuffle_buffer_size=_SHUFFLE_BUFFER_SIZE,
        shuffle_seed=self.runner_parameters.random_seed,
        prefetch_buffer_size=tf.data.AUTOTUNE,
        num_parallel_reads=tf.data.AUTOTUNE,
        sloppy=False,  # Set to True for non-deterministic ordering for speed.
        num_rows_for_inference=100,
        compression_type=None,
        ignore_errors=False,
        encoding='utf-8',
    )
    if not dataset:
      raise ValueError(
          f'Dataset with prefix {self._last_read_metadata.location_prefix} not '
          'created.'
      )

    # The Dataset can have labels of type int or str. Cast them to int.
    dataset = dataset.map(
        lambda features, label: (features, self.remap_label(label)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    # Filter the dataset by label column value. Filtering is applied after
    # re-mapping the labels so that the labels are all integers.
    ds_filter_func = self._get_filter_by_label_value_func(
        label_column_filter_value=label_column_filter_value,
        exclude_label_value=exclude_label_value,
    )
    dataset = dataset.filter(ds_filter_func)

    # The Dataset returns features as a dict. Combine the features into a single
    # tensor. Also cast the features and labels to correct types.
    def combine_features_dict_into_tensor(
        features: Mapping[str, tf.Tensor],
        label: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
      """Combines the features into a single tensor."""
      feature_matrix = tf.squeeze(tf.stack(list(features.values()), axis=1))
      feature_matrix = tf.reshape(feature_matrix, (-1,))
      feature_matrix = tf.cast(
          feature_matrix,
          tf.dtypes.as_dtype(_FEATURES_TYPE.lower()),
          name='features',
      )
      label = tf.squeeze(label)
      label = tf.reshape(label, (-1,))
      label = tf.cast(
          label, tf.dtypes.as_dtype(_LABEL_TYPE.lower()), name='label'
      )
      return feature_matrix, label

    dataset = dataset.map(
        combine_features_dict_into_tensor,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    dataset = dataset.repeat(1)  # One repeat of the dataset during creation.
    if batch_size:
      dataset = dataset.batch(batch_size, deterministic=True)
      dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # This Dataset was just created. Calculate the label distribution. Any
    # string labels were already re-mapped to integers. So keys are always
    # integers and values are EagerTensors. We need to extract the value within
    # this Tensor for subsequent use.
    self._label_counts = {
        k: v.numpy() for k, v in self.counts_by_label(dataset).items()
    }

    return dataset

  def counts_by_label(self, dataset: tf.data.Dataset) -> Dict[int, tf.Tensor]:
    """Counts the number of samples in each label class in the dataset.

    When this function is called, the labels in the Dataset have already been
    re-mapped to integers. So all counting operations make this assumption.

    Args:
      dataset: The dataset to count the labels in.

    Returns:
      A dictionary of label class (as integer) to counts (as integer).
    """

    @tf.function
    def count_class(
        counts: Dict[int, int],
        batch: Tuple[tf.Tensor, tf.Tensor],
    ) -> Dict[int, int]:
      _, labels = batch
      labels = self.remap_label(labels)
      new_counts: Dict[int, int] = counts.copy()
      for i in self.all_labels:
        # This function is called after the Dataset is constructed and the
        # labels are re-mapped to integers. So convert the label to an integer.
        if isinstance(i, str):
          i_for_compare = self.convert_str_to_int(i)
        else:
          i_for_compare = i
        cc: tf.Tensor = tf.cast(labels == i_for_compare, tf.int32)
        if i_for_compare in list(new_counts.keys()):
          new_counts[i_for_compare] += tf.reduce_sum(cc)
        else:
          new_counts[i_for_compare] = tf.reduce_sum(cc)
      return new_counts

    initial_state = dict(
        (self.convert_str_to_int(i), 0) for i in self.all_labels
    )
    counts = dataset.reduce(
        initial_state=initial_state, reduce_func=count_class
    )
    return counts

  def counts_by_original_label(
      self, dataset: tf.data.Dataset
  ) -> tuple[dict[str, tf.Tensor], dict[int, tf.Tensor]]:
    """Counts the number of samples in each label class in the dataset."""

    all_int_labels = [l for l in self.all_labels if isinstance(l, int)]
    logging.info('all_int_labels: %s', all_int_labels)
    all_str_labels = [l for l in self.all_labels if isinstance(l, str)]
    logging.info('all_str_labels: %s', all_str_labels)

    @tf.function
    def count_original_class(
        counts: Dict[int | str, int],
        batch: Tuple[tf.Tensor, tf.Tensor],
    ) -> Dict[int | str, int]:
      keys_are_int = all(isinstance(k, int) for k in counts.keys())
      if keys_are_int:
        all_labels = all_int_labels
      else:
        all_labels = all_str_labels
      _, labels = batch
      new_counts: Dict[int | str, int] = counts.copy()
      for label in all_labels:
        cc: tf.Tensor = tf.cast(labels == label, tf.int32)
        if label in list(new_counts.keys()):
          new_counts[label] += tf.reduce_sum(cc)
        else:
          new_counts[label] = tf.reduce_sum(cc)
      return new_counts

    int_keys_map = {
        k: v
        for k, v in _STRING_TO_INTEGER_LABEL_MAP.items()
        if isinstance(k, int)
    }
    initial_int_state = dict((int(label), 0) for label in int_keys_map.keys())
    if initial_int_state:
      int_counts = dataset.reduce(
          initial_state=initial_int_state, reduce_func=count_original_class
      )
    else:
      int_counts = {}
    str_keys_map = {
        k: v
        for k, v in _STRING_TO_INTEGER_LABEL_MAP.items()
        if isinstance(k, str)
    }
    initial_str_state = dict((str(label), 0) for label in str_keys_map.keys())
    str_counts = dataset.reduce(
        initial_state=initial_str_state, reduce_func=count_original_class
    )
    return int_counts, str_counts

  def get_label_thresholds(self) -> Mapping[str, float]:
    """Computes positive and negative thresholds based on label ratios.

    This method is useful for setting percentile thresholds when performing
    inference on the OCC ensemble. Feature vectors can be labeled as normal
    or anomalous depending on these values. This method requires that the label
    column, and positive and negative values are set in RunnerParameters.

    Args: None.

    Returns:
      A dictionary containing 'positive_threshold' and 'negative_threshold'.

    Raises:
      ValueError: If the label counts have not been computed yet.
    """
    if not self._label_counts:
      raise ValueError(
          'Label counts have not been computed yet, '
          'ensure that you have made a call to '
          'load_tf_dataset_from_csv() before this method '
          'is called.'
      )

    positive_count = self._label_counts[
        self.convert_str_to_int(self.runner_parameters.positive_data_value)
    ]

    labeled_data_record_count = (
        self._label_counts[
            self.convert_str_to_int(self.runner_parameters.positive_data_value)
        ]
        + self._label_counts[
            self.convert_str_to_int(self.runner_parameters.negative_data_value)
        ]
    )

    positive_threshold = 100 * (positive_count / labeled_data_record_count)
    label_thresholds = {
        'positive_threshold': positive_threshold,
        'negative_threshold': 100 - positive_threshold,
    }

    if self.runner_parameters.verbose:
      logging.info('Computed label thresholds: %s', label_thresholds)

    return label_thresholds

  def upload_dataframe_to_gcs(
      self,
      batch: int,
      features: np.ndarray,
      labels: np.ndarray,
      weights: Optional[np.ndarray] = None,
      pseudolabel_flags: Optional[np.ndarray] = None,
      map_labels_to_bool: bool = False,
  ) -> None:
    """Uploads the dataframe to BigQuery, create or replace table.

    Args:
      batch: The batch number of the pseudo-labeled data.
      features: Numpy array of features.
      labels: Numpy array of labels.
      weights: Optional numpy array of weights.
      pseudolabel_flags: Optional numpy array of pseudolabel flags.
      map_labels_to_bool: If True, map labels to bool. This is useful for
        uploading data to BigQuery or AutoML for further analysis.

    Returns:
      None.

    Raises:
      ValueError: If the metadata has not been read yet or if the data output
      GCS URI is not set in the runner parameters.
    """
    if not self._last_read_metadata:
      raise ValueError(
          'No metadata has been read yet, ensure that you have made a call to '
          'load_tf_dataset_from_csv(), trained the model and performed '
          ' pseudo-labeling before this method is called.'
      )
    if not self.runner_parameters.data_output_gcs_uri:
      raise ValueError(
          'Data output GCS URI is not set in the runner parameters. Please set '
          'the `data_output_gcs_uri` field in the runner parameters.'
      )
    combined_data = features

    column_names = list(
        self._last_read_metadata.column_names_info.column_names_dict.keys()
    )
    # Save a copy of the feature column names.
    feature_column_names = column_names.copy()
    feature_column_names.remove(self.runner_parameters.label_col_name)

    # If the weights are provided, add them to the column names and to the
    # combined data.
    if weights is not None:
      column_names.append(data_loader.WEIGHT_COLUMN_NAME)
      combined_data = np.concatenate(
          [combined_data, weights.reshape(len(features), 1).astype(np.float64)],
          axis=1,
      )

    # If the pseudolabel flags are provided, add them to the column names and
    # to the combined data.
    if pseudolabel_flags is not None:
      column_names.append(data_loader.PSEUDOLABEL_FLAG_COLUMN_NAME)
      combined_data = np.concatenate(
          [
              combined_data,
              pseudolabel_flags.reshape(len(features), 1).astype(np.int64),
          ],
          axis=1,
      )

    # Make sure the label column is the last column.
    combined_data = np.concatenate(
        [combined_data, labels.reshape(len(features), 1)], axis=1
    )
    column_names.remove(self.runner_parameters.label_col_name)
    column_names.append(self.runner_parameters.label_col_name)

    complete_dataframe = pd.DataFrame(data=combined_data, columns=column_names)
    feature_column_dtypes_map = {
        c: _FEATURES_TYPE.lower() for c in feature_column_names
    }
    column_dtypes_map = {
        self.runner_parameters.label_col_name: (
            str if self.runner_parameters.labels_are_strings else int
        ),
        data_loader.WEIGHT_COLUMN_NAME: np.float64,
        data_loader.PSEUDOLABEL_FLAG_COLUMN_NAME: np.int64,
    } | feature_column_dtypes_map
    complete_dataframe = complete_dataframe.astype(column_dtypes_map)
    if map_labels_to_bool:
      # Adjust label column type so that users can go straight to BigQuery or
      # AutoML without having to adjust data. Both of these products require a
      # boolean or string target column, not integer.
      complete_dataframe[
          self.runner_parameters.label_col_name
      ] = complete_dataframe[self.runner_parameters.label_col_name].map({
          self.runner_parameters.positive_data_value: True,
          self.runner_parameters.negative_data_value: False,
      })
      complete_dataframe[self.runner_parameters.label_col_name] = (
          complete_dataframe[self.runner_parameters.label_col_name].astype(bool)
      )

    # Adjust pseudolabel flag column type.
    if pseudolabel_flags is not None:
      complete_dataframe[data_loader.PSEUDOLABEL_FLAG_COLUMN_NAME] = (
          complete_dataframe[data_loader.PSEUDOLABEL_FLAG_COLUMN_NAME].astype(
              np.int64
          )
      )

    output_path = os.path.join(
        self.runner_parameters.data_output_gcs_uri,
        f'pseudo_labeled_batch_{batch}.csv',
    )
    with tf.io.gfile.GFile(output_path, 'w') as f:
      complete_dataframe.to_csv(
          f,
          index=False,
          header=True,
          quoting=csv.QUOTE_NONNUMERIC,
      )
    if self.runner_parameters.verbose:
      logging.info('Uploaded pseudo-labeled data to %s', output_path)
