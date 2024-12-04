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

"""Tests for supervised models."""


from unittest import mock

from absl.testing import parameterized
from spade_anomaly_detection import data_loader
from spade_anomaly_detection import supervised_model
import tensorflow as tf
import tensorflow_decision_forests as tfdf


def _get_labeled_dataset(labels_are_strings: bool = False) -> tf.data.Dataset:
  """Loads the thyroid_labeled dataset for model performance testing.

  Args:
    labels_are_strings: Whether the labels are strings or integers.

  Returns:
    TensorFlow dataset with X, y split.
  """
  features, labels = data_loader.load_dataframe('thyroid_labeled')

  original_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
  if labels_are_strings:
    # Mimic the behavior of the CsvDataLoader by converting the labels to
    # strings. The string labels are converted back to integers in the
    # Data loader.
    # Treat the labels as strings for testing. Note that the test dataset
    # contains only integer labels.
    original_dataset = original_dataset.map(lambda x, y: (x, tf.as_string(y)))

  labeled_dataset = original_dataset.batch(100).as_numpy_iterator().next()

  return labeled_dataset


class SupervisedModelsTest(tf.test.TestCase, parameterized.TestCase):

  def test_supervised_model_create_no_error(self):
    params = supervised_model.RandomForestParameters(max_depth=20)
    model_instance = supervised_model.RandomForestModel(parameters=params)

    self.assertEqual(model_instance.supervised_parameters.max_depth, 20)

  @parameterized.named_parameters(
      ('labels_are_integers', False),
      ('labels_are_strings', True),
  )
  @mock.patch.object(tfdf.keras.RandomForestModel, 'fit', autospec=True)
  def test_train_supervised_model_no_error(
      self, labels_are_strings: bool, mock_fit: mock.Mock
  ):
    params = supervised_model.RandomForestParameters()
    model = supervised_model.RandomForestModel(parameters=params)
    features, labels = _get_labeled_dataset(labels_are_strings)

    # The CsvDataLoader converts the labels to integers, while the BQDataLoader
    # assumes that the labels are integers. So for this test, convert the labels
    # to integers here.
    model.train(features=features, labels=labels.astype(int))

    mock_fit_actual_call_args = mock_fit.call_args.kwargs

    # Convert the labels to integers for the expected call args.
    expected_fit_call_args = {
        'x': features,
        'y': labels.astype(int),
        'sample_weight': None,
    }

    self.assertDictEqual(mock_fit_actual_call_args, expected_fit_call_args)

  @parameterized.named_parameters(
      ('labels_are_integers', False),
      ('labels_are_strings', True),
  )
  @mock.patch.object(tfdf.keras.RandomForestModel, 'save', autospec=True)
  def test_model_saving_no_error(
      self, labels_are_strings: bool, mock_tfdf_save: mock.Mock
  ):
    job_dir: str = 'gs://test_bucket/test_folder/'
    params = supervised_model.RandomForestParameters()
    model = supervised_model.RandomForestModel(parameters=params)
    features, labels = _get_labeled_dataset(labels_are_strings)

    # The CsvDataLoader converts the labels to integers, while the BQDataLoader
    # assumes that the labels are integers. So for this test, convert the labels
    # to integers here.
    model.train(features=features, labels=labels.astype(int))
    model.save(job_dir)

    mock_tfdf_save.assert_called_once_with(model.supervised_model, job_dir)


if __name__ == '__main__':
  tf.test.main()
