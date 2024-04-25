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

"""Tests for supervised models."""

from unittest import mock

from spade_anomaly_detection import data_loader
from spade_anomaly_detection import supervised_model

import tensorflow as tf
import tensorflow_decision_forests as tfdf


def _get_labeled_dataset() -> tf.data.Dataset:
  """Loads the thyroid_labeled dataset for model performance testing.

  Returns:
    TensorFlow dataset with X, y split.
  """
  features, labels = data_loader.load_dataframe('thyroid_labeled')

  labeled_dataset = (
      tf.data.Dataset.from_tensor_slices((features, labels))
      .batch(100)
      .as_numpy_iterator()
      .next()
  )

  return labeled_dataset


class SupervisedModelsTest(tf.test.TestCase):

  def test_supervised_model_create_no_error(self):
    params = supervised_model.RandomForestParameters(max_depth=20)
    model_instance = supervised_model.RandomForestModel(parameters=params)

    self.assertEqual(model_instance.supervised_parameters.max_depth, 20)

  @mock.patch.object(tfdf.keras.RandomForestModel, 'fit', autospec=True)
  def test_train_supervised_model_no_error(self, mock_fit):
    params = supervised_model.RandomForestParameters()
    model = supervised_model.RandomForestModel(parameters=params)
    features, labels = _get_labeled_dataset()

    model.train(features=features, labels=labels)

    mock_fit_actual_call_args = mock_fit.call_args.kwargs

    expected_fit_call_args = {'x': features, 'y': labels, 'sample_weight': None}

    self.assertDictEqual(mock_fit_actual_call_args, expected_fit_call_args)

  @mock.patch.object(tfdf.keras.RandomForestModel, 'save', autospec=True)
  def test_model_saving_no_error(self, mock_tfdf_save):
    job_dir = 'gs://test_bucket/test_folder/'
    params = supervised_model.RandomForestParameters()
    model = supervised_model.RandomForestModel(parameters=params)
    features, labels = _get_labeled_dataset()

    model.train(features=features, labels=labels)
    model.save(job_dir)

    mock_tfdf_save.assert_called_once_with(model.supervised_model, job_dir)


if __name__ == '__main__':
  tf.test.main()
