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

"""Tests for the CSV data loader."""

import collections
import dataclasses
import os
from typing import Set, Tuple
from unittest import mock

from absl.testing import parameterized
import numpy as np
import pandas as pd
from spade_anomaly_detection import csv_data_loader
from spade_anomaly_detection import parameters
import tensorflow as tf

import tensorflow_datasets as tfds

import pytest

# Test Config.
_NUMBER_OF_DECIMALS_TO_ROUND = 2


def _dataset_to_set_of_nested_tuples(
    ds: tf.data.Dataset,
) -> Set[Tuple[Tuple[float, ...], Tuple[float]]]:  # pylint: disable=g-one-element-tuple
  """Helper to convert a dataset to a tuple of tuples of tuples for tests."""
  ds_list = list(ds.as_numpy_iterator())
  new_ds_list = []
  for elem in ds_list:
    # Convert the first element to a tuple, which is the features.
    if elem[0].ndim == 1:
      new_elem_0 = tuple(
          np.round(elem[0], decimals=_NUMBER_OF_DECIMALS_TO_ROUND).tolist()
      )
    else:
      new_elem_0 = tuple(
          np.round(elem[0], decimals=_NUMBER_OF_DECIMALS_TO_ROUND)
          .reshape((-1,))
          .tolist()
      )
    # Convert second element to a tuple, which is the label.
    if elem[1].ndim == 1:
      new_elem_1 = tuple(elem[1].tolist())
    else:
      new_elem_1 = tuple(elem[1].reshape((-1,)).tolist())
    new_elem = (new_elem_0, new_elem_1)
    new_ds_list.append(new_elem)
  return set(new_ds_list)


@dataclasses.dataclass(frozen=True)
class FakeBlob:
  """Represents a fake GCS blob to be returned by bucket.list_blobs.

  Attributes:
    name: Name of the blob.
    contents: Contents to be returned when download_as_string is called.
  """

  name: str
  contents: str

  def download_as_string(self):
    return self.contents


class CsvDataUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.storage_client_mock = mock.MagicMock()
    self.bucket_mock = mock.MagicMock()
    self.storage_client_mock.return_value.bucket.return_value = self.bucket_mock

    self.header = ["x1", "x2", "y"]
    self.data1 = [[0.6, 0.2, -1], [0.1, 0.8, 0], [0.6, 0.9, 1]]
    self.data1_df = pd.DataFrame(data=self.data1, columns=self.header)
    self.csv_file1 = "/dir1/data1.csv"
    self.csv_file1_content = self.data1_df.to_csv(header=True, index=False)

  # Params to test: gcs_uri.
  @parameterized.named_parameters(
      ("single_file", "gs://bucket/dir/file.csv", "bucket", "dir/file.csv/"),
      ("folder_no_slash", "gs://bucket/dir", "bucket", "dir/"),
      ("folder_with_slash", "gs://bucket/dir/", "bucket", "dir/"),
  )
  def test_parse_gcs_uri_returns_bucket_name_and_prefix(
      self, gcs_uri, expected_bucket, expected_prefix
  ):
    bucket_name, prefix = csv_data_loader._parse_gcs_uri(gcs_uri=gcs_uri)
    self.assertEqual(bucket_name, expected_bucket)
    self.assertEqual(prefix, expected_prefix)

  def test_parse_gcs_uri_incorrect_uri_raises(self):
    gcs_uri = "bucket/dir/"
    with self.assertRaises(ValueError):
      _, _ = csv_data_loader._parse_gcs_uri(gcs_uri=gcs_uri)

  def test_list_files_returns_listed_files(self):
    self.bucket_mock.list_blobs.return_value = [
        FakeBlob(name="dir/", contents=""),
        FakeBlob(
            name="dir/file1.csv", contents="x1,x2,y\n0.1,0.2,1\n0.2,0.3,0"
        ),
        FakeBlob(
            name="dir/file2.csv", contents="x1,x2,y\n0.4,0.6,1\n0.5,0.5,0"
        ),
        FakeBlob(name="dir/file2.txt", contents="doesn't matter"),
    ]
    with mock.patch("google.cloud.storage.Client", self.storage_client_mock):
      all_files = csv_data_loader._list_files(
          bucket_name="bucket", input_blob_prefix="dir/file"
      )
    expected_files = [
        "gs://bucket/dir/file1.csv",
        "gs://bucket/dir/file2.csv",
        "gs://bucket/dir/file2.txt",
    ]

    self.assertCountEqual(all_files, expected_files)

  def test_get_header_from_input_file_returns_header(self):
    with tfds.testing.MockFs() as fs:
      fs.add_file(f"{self.csv_file1}", self.csv_file1_content)
      header = csv_data_loader._get_header_from_input_file(self.csv_file1)
    expected_header = "x1,x2,y\n"
    self.assertEqual(header, expected_header)

  def test_column_names_info_from_inputs_file_returns_column_names_info(self):
    with tfds.testing.MockFs() as fs:
      fs.add_file(f"{self.csv_file1}", self.csv_file1_content)
      column_names_info = csv_data_loader.ColumnNamesInfo.from_inputs_file(
          inputs_file=self.csv_file1, label_column_name="y"
      )
    expected_column_names_info = csv_data_loader.ColumnNamesInfo(
        header="x1,x2,y",
        label_column_name="y",
        column_names_dict=collections.OrderedDict(
            [("x1", "FLOAT64"), ("x2", "FLOAT64"), ("y", "INT64")]
        ),
        num_features=2,
    )
    self.assertEqual(column_names_info, expected_column_names_info)


class CsvDataLoaderTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.header = ["x1", "x2", "y"]
    self.dir = "dir1/"
    self.data1 = [[0.6, 0.2, -1], [0.1, 0.8, 0], [0.6, 0.9, 1]]
    self.data1_df = pd.DataFrame(data=self.data1, columns=self.header)
    self.csv_file1 = f"{self.dir}data1.csv"
    self.csv_file1_content = self.data1_df.to_csv(header=True, index=False)
    self.data2 = [[0.6, 0.7, 1], [0.6, 0.5, 0], [0.6, 0.9, 1], [0.6, 0.2, 1]]
    self.data2_df = pd.DataFrame(data=self.data2, columns=self.header)
    self.csv_file2 = f"{self.dir}data2.csv"
    self.csv_file2_content = self.data2_df.to_csv(header=True, index=False)
    self.data_df = pd.concat([self.data1_df, self.data2_df])
    self.data_df = self.data_df.astype({"y": "bool"})

  @parameterized.named_parameters(
      dict(
          testcase_name="no_label_value_no_exclude",
          label_column_filter_value=None,
          exclude_label_value=False,
          inputs=[([0.6, 0.2], [-1]), ([0.1, 0.8], [0]), ([0.6, 0.9], [1])],
          expected=[True, True, True],
      ),
      dict(
          testcase_name="positive_label_value_no_exclude",
          label_column_filter_value=1,
          exclude_label_value=False,
          inputs=[([0.6, 0.2], [-1]), ([0.1, 0.8], [0]), ([0.6, 0.9], [1])],
          expected=[False, False, True],
      ),
      dict(
          testcase_name="positive_label_value_exclude",
          label_column_filter_value=1,
          exclude_label_value=True,
          inputs=[([0.6, 0.2], [-1]), ([0.1, 0.8], [0]), ([0.6, 0.9], [1])],
          expected=[True, True, False],
      ),
      dict(
          testcase_name="pos_and_0_label_value_no_exclude",
          label_column_filter_value=[0, 1],
          exclude_label_value=False,
          inputs=[([0.6, 0.2], [-1]), ([0.1, 0.8], [0]), ([0.6, 0.9], [1])],
          expected=[False, True, True],
      ),
  )
  def test_get_filter_by_label_value_func(
      self, label_column_filter_value, exclude_label_value, inputs, expected
  ):
    runner_parameters = parameters.RunnerParameters(
        train_setting="PNU",
        input_bigquery_table_path=None,
        data_input_gcs_uri="gs://bucket/input_path",
        output_gcs_uri="gs://bucket/output_path",
        label_col_name="y",
        positive_data_value=1,
        negative_data_value=0,
        unlabeled_data_value=-1,
        positive_threshold=5,
        negative_threshold=95,
    )
    instance = csv_data_loader.CsvDataLoader(
        runner_parameters=runner_parameters
    )
    filter_func = instance._get_filter_by_label_value_func(
        label_column_filter_value=label_column_filter_value,
        exclude_label_value=exclude_label_value,
    )
    for i, ((input_f, input_l), keep) in enumerate(zip(inputs, expected)):
      with self.subTest(msg=f"input_{i}"):
        got = filter_func(input_f, input_l)
        self.assertEqual(keep, got)

  # Test the creation of a Dataset from CSV files. Only tests batch_size=1.
  @pytest.mark.skip(reason="create_tempdir is broken in pytest")
  @mock.patch.object(csv_data_loader, "_list_files", autospec=True)
  @mock.patch.object(csv_data_loader, "_parse_gcs_uri", autospec=True)
  @mock.patch.object(tf.io.gfile.GFile, "readline", autospec=True)
  def test_load_tf_dataset_from_csv_returns_expected_dataset(
      self, mock_readline, mock_parse_gcs_uri, mock_file_reader
  ):
    mock_readline.return_value = ",".join(self.header)
    tmp_dir = self.create_tempdir("tmp")
    input_path = os.path.join(tmp_dir.full_path, self.dir)
    tf.io.gfile.makedirs(input_path)
    mock_parse_gcs_uri.return_value = ("doesnt_matter", input_path)
    mock_file_reader.return_value = [
        os.path.join(tmp_dir.full_path, self.csv_file1),
        os.path.join(tmp_dir.full_path, self.csv_file2),
    ]
    # Write the test CSV files to temporary files. These CSV files will be
    # re-read when the Dataset is created. Their metadata will also be recorded
    # in the InputFilesMetadata object.
    self.data1_df.to_csv(
        os.path.join(tmp_dir.full_path, self.csv_file1),
        header=True,
        index=False,
    )
    self.data2_df.to_csv(
        os.path.join(tmp_dir.full_path, self.csv_file2),
        header=True,
        index=False,
    )
    runner_parameters = parameters.RunnerParameters(
        train_setting="PNU",
        input_bigquery_table_path=None,
        data_input_gcs_uri=input_path,
        output_gcs_uri=f"{input_path}/output",
        label_col_name="y",
        positive_data_value=1,
        negative_data_value=0,
        unlabeled_data_value=-1,
        positive_threshold=5,
        negative_threshold=95,
        verbose=True,
    )

    data_loader = csv_data_loader.CsvDataLoader(
        runner_parameters=runner_parameters
    )
    dataset = data_loader.load_tf_dataset_from_csv(
        input_path=runner_parameters.data_input_gcs_uri,
        label_col_name=runner_parameters.label_col_name,
        batch_size=1,
        label_column_filter_value=None,
        exclude_label_value=False,
    )

    expected_dataset = tf.data.Dataset.from_tensor_slices((
        tf.constant(
            [
                [0.6, 0.2],
                [0.1, 0.8],
                [0.6, 0.9],
                [0.6, 0.7],
                [0.6, 0.5],
                [0.6, 0.9],
                [0.6, 0.2],
            ],
            dtype=tf.float64,
        ),
        tf.constant([[-1], [0], [1], [1], [0], [1], [1]], dtype=tf.float64),
    ))

    expected_element_spec = (
        tf.TensorSpec(shape=(None, None), dtype=tf.float64),  # features
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),  # label
    )
    with self.subTest(msg="check_spec_equal"):
      self.assertTupleEqual(dataset.element_spec, expected_element_spec)

    expected_counts = {
        -1: 1,
        0: 2,
        1: 4,
    }
    with self.subTest(msg="check_dataset_class_counts"):
      counts = data_loader.counts_by_label(dataset)
      self.assertDictEqual(counts, expected_counts)

    expected_label_thresholds = {
        "positive_threshold": 66.6666,
        "negative_threshold": 33.3333,
    }
    with self.subTest(msg="check_dataset_label_thresholds"):
      got_label_thresholds = data_loader.get_label_thresholds()
      self.assertDictEqual(got_label_thresholds, expected_label_thresholds)

    with self.subTest(msg="check_datasets_have_same_elements"):
      self.assertSetEqual(
          _dataset_to_set_of_nested_tuples(dataset),
          _dataset_to_set_of_nested_tuples(expected_dataset),
      )

  @pytest.mark.skip(reason="create_tempdir is broken in pytest")
  def test_upload_dataframe_to_gcs(self):
    tmp_dir = self.create_tempdir("tmp")
    output_dir = os.path.join(tmp_dir.full_path, self.dir, "output_path")
    runner_parameters = parameters.RunnerParameters(
        train_setting="PNU",
        input_bigquery_table_path=None,
        data_input_gcs_uri="gs://bucket/input_path",
        output_gcs_uri="gs://bucket/model_path",
        label_col_name="y",
        positive_data_value=1,
        negative_data_value=0,
        unlabeled_data_value=-1,
        positive_threshold=5,
        negative_threshold=95,
        data_output_gcs_uri=output_dir,
    )
    data_loader = csv_data_loader.CsvDataLoader(
        runner_parameters=runner_parameters
    )
    tf.io.gfile.makedirs(output_dir)
    col_names_info = csv_data_loader.ColumnNamesInfo(
        header="x1,x2,y",
        label_column_name="y",
        column_names_dict=collections.OrderedDict(
            [("x1", "FLOAT64"), ("x2", "FLOAT64"), ("y", "INT64")]
        ),
        num_features=2,
    )
    data_loader._last_read_metadata = csv_data_loader.InputFilesMetadata(
        column_names_info=col_names_info,
        location_prefix="doesnt_matter_for_uploader",
        files=["doesnt_matter_for_uploader"],
    )
    all_features = self.data_df[["x1", "x2"]].to_numpy()
    all_labels = self.data_df["y"].to_numpy()
    # Create 2 batches of features and labels.
    # TODO(b/347332980): Update test when pseudolabel flag is added.
    features1 = all_features[0:2]
    labels1 = all_labels[0:2]
    features2 = all_features[2:]
    labels2 = all_labels[2:]
    # Upload batch 1.
    data_loader.upload_dataframe_to_gcs(
        batch=1,
        features=features1,
        labels=labels1,
    )
    # Upload batch 2.
    data_loader.upload_dataframe_to_gcs(
        batch=2,
        features=features2,
        labels=labels2,
    )
    # Sorting means batch 1 file will be first.
    files_list = sorted(tf.io.gfile.listdir(output_dir))
    self.assertLen(files_list, 2)
    expected_dfs = [
        self.data_df.iloc[0:2].reset_index(drop=True),
        self.data_df.iloc[2:].reset_index(drop=True),
    ]
    for i, file_name in enumerate(files_list):
      with self.subTest(msg=f"file_{i}"):
        file_path = os.path.join(output_dir, file_name)
        with tf.io.gfile.GFile(file_path, "r") as f:
          got_df = pd.read_csv(f, header=0)
        pd.testing.assert_frame_equal(
            got_df, expected_dfs[i], check_exact=False, check_like=True
        )


if __name__ == "__main__":
  tf.test.main()
