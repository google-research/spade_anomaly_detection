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

"""Holds dataclasses and enums leveraged by the SPADE algorithm."""


import dataclasses
import enum
from typing import Final, Sequence


_RANDOM_SEED: Final[int] = 42


@enum.unique
class TrainSetting(str, enum.Enum):
  """Train setting options for the runner, mutually exclusive."""

  PU = 'PU'
  PNU = 'PNU'


# An immutable mapping of label values to their corresponding integer values.
# This supports label values that the empty string. This value is mapped to -1,
# which is the value used to denote unlabeled data.
# None is not included in this mapping, because the Data Loader will default
# to the empty string if the label column is empty.
labels_mapping: Final[dict[str | None, int]] = {
    '': -1,
    '1': 1,
    '0': 0,
    '-1': -1,
}


@enum.unique
class VotingStrategy(enum.Enum):
  """Voting strategy for the ensemble."""

  UNANIMOUS = 'unanimous'
  MAJORITY = 'majority'


@dataclasses.dataclass
class RunnerParameters:
  """Stores runner related parameters for helper functions in the module.

  Attributes:
    train_setting: The 'PNU' setting will train the supervised model using
      ground truth negative, positive data, and pseudo labeled positive and
      negative data. The 'PU' setting will only use negative data from the
      pseudo labeler along with the rest of the positive data (ground truth plus
      pseudo labeled) to train the supervised model. For model evaluation, we
      still require ground truth negative data to be in the BigQuery dataset, it
      just won't be used during training. Default is PNU.
    input_bigquery_table_path: A BigQuery table path in the format
      'project.dataset.table'. If this is the only BigQuery path provided, this
      will be used in conjunction with test_dataset_holdout_fraction parameter
      to create a train/test split.
    data_input_gcs_uri: Cloud Storage location to store the input data. If this
      is the only Cloud Storage location provided, this will be used in
      conjunction with test_dataset_holdout_fraction parameter to create a
      train/test split.
    output_gcs_uri: Cloud Storage location to store the supervised model assets.
      The location should be in the form gs://bucketname/foldername. A timestamp
      will be added to the end of the folder so that multiple runs of this won't
      overwrite previous runs.
    label_col_name: The name of the label column in the input BigQuery table.
    labels_are_strings: Whether the labels in the input dataset are strings or
      integers.
    positive_data_value: The value used in the label column to denote positive
      data - data points that are anomalous.
    negative_data_value: The value used in the label column to denote negative
      data - data points that are not anomalous.
    unlabeled_data_value: The value used in the label column to denote unlabeled
      data.
    positive_threshold: Float between [0, 100] used as the percentile for the
      one class classifier ensemble to label a point as positive. The closer to
      0 this value is set, the less positive data will be labeled. However, we
      expect an increase in precision when lowering the value, and an increase
      in recall when raising it. Equivalent to saying the given data point needs
      to be located in the top X percentile in order to be considered anomalous.
      This setting is not required, and is automatically instantiated when no
      value is set based off of ratios of positive and negative data from the
      input_bigquery_table_path.
    negative_threshold: Float between [0, 100] used as the percentile for the
      one class classifier ensemble to label a point as negative. The higher
      this value is set, the less negative data will be labeled. A value in the
      range of 50-99 is a good starting point. We expect an increase in
      precision when raising this value, and an increase in recall when lowering
      it. Equivalent to saying the given data point needs to be X percentile or
      greater in order to be considered anomalous. This setting is not required,
      and is automatically instantiated when no value is set based off of ratios
      of positive and negative data from the input_bigquery_table_path.
    ignore_columns: A list of columns located in the input table that you would
      like to ignore in both pseudo labeling and supervised model training.
    where_statements: Additional SQL where statements with correct syntax that
      can be leveraged to filter BigQuery table data. An example is "date >
      '2008-11-11'" Statements must not contain any trailing "AND" keywords. All
      statements will be combined with "AND" statements automatically along with
      the initial "WHERE" clause.
    test_bigquery_table_path: A complete BigQuery path in the form of
      'project.dataset.table' to be used for evaluating the supervised model.
      Note that the positive and negative label values must also be the same in
      this testing set. It is okay to have your test labels in that form, or use
      1 for positive and 0 for negative.
    test_label_col_name: The label column name in the test dataset.
    test_dataset_holdout_fraction: Float between 0 and 1 representing the
      fraction of samples to hold out as a test set.
    data_test_gcs_uri: Cloud Storage location to store the CSV data to be used
      for evaluating the supervised model. Note that the positive and negative
      label values must also be the same in this testing set. It is okay to have
      your test labels in that form, or use 1 for positive and 0 for negative.
    upload_only: Use this setting in conjunction with output_bigquery_table_path
      or data_output_gcs_uri. When True, the algorithm will just upload the
      pseudo labeled data to the specified table, and will skip training a
      supervised model. When set to False, the algorithm will also train a
      supervised model and upload it to a GCS location. Default is False.
    output_bigquery_table_path: A complete BigQuery path in the form of
      'project.dataset.table' to be used for uploading the pseudo labeled data.
      This includes features and new labels. By default, we will use the column
      names from the input_bigquery_table_path BigQuery table.
    data_output_gcs_uri: Cloud Storage location used for uploading the pseudo
      labeled data as CSV. This includes features and new labels. By default, we
      will use the column names from the data_input_gcs_uri table.
    voting_strategy: The voting strategy to use when determining if a data point
      is anomalous. By default, we use unanimous voting, meaning all the models
      in the ensemble need to agree in order to label a data point as anomalous.
    alpha: Sample weights for weighting the loss function, only for positively
      pseudo-labeled data from the occ ensemble. Original data that is labeled
      will have a weight of 1.0. If this is provided and
      `alpha_negative_pseudolabels` is not provided, then this value will be
      used for both positive and negative pseudo-labeled data.
    alpha_negative_pseudolabels: Sample weights for weighting the loss function,
      only for negatively pseudo-labeled data from the occ ensemble. Original
      data that is labeled will have a weight of 1.0. If this is not provided,
      then the `alpha` value will be used for both positive and negative
      pseudo-labeled data.
    batches_per_model: The number of batches to use in training each model in
      the ensemble. By default it is 1, meaning use 1/N of the entire dataset
      when training each model, where N is the number of OCCs in the ensemble.
    max_occ_batch_size: The maximum number of examples in a batch to train a
      single one-class classifiers (e.g., GMM)
    labeling_and_model_training_batch_size: The number of examples to use when
      pseudo labeling unlabeled training examples. The subset of labeled
      examples are then fed to the supervised model to perform a train step.
      When there is no batch size specified here, we will use the entire dataset
      to score, label, and train the supervised model. For large datasets, use
      the batches_per_model and this setting to reduce the amount of data stored
      locally on the training machine.
    ensemble_count: Number of one class classifiers in the ensemble use for
      pseudo labeling unlabeled data points. The more models in the ensemble,
      the less likely it is for all the models to gain consensus, and thus will
      reduce the amount of labeled data points. By default, we use 5 one class
      classifiers.
    n_components: The number of components to use in the one class classifier
      ensemble. By default, we use 1 component. Pass a single integer if all the
      ensemble models should have the same number of components. Pass a
      space-separated list of integers if you want to use different numbers of
      components for each model in the ensemble.
    covariance_type: The covariance type to use in the one class classifier
      ensemble. By default, we use 'full' covariance. Note that when there are
      many components, a 'full' covariance matrix may not be suitable.
    random_seed: The random seed to use for all random number generators in the
      algorithm.
    verbose: The amount of console logs to display during training. Use False to
      show few messages, and True for displaying many aspects of model training
      and scoring. This is useful for debugging model performance.
  """

  train_setting: TrainSetting
  input_bigquery_table_path: str
  data_input_gcs_uri: str
  output_gcs_uri: str
  label_col_name: str
  positive_data_value: int | str
  negative_data_value: int | str
  unlabeled_data_value: int | str
  labels_are_strings: bool = True
  positive_threshold: float | None = None
  negative_threshold: float | None = None
  ignore_columns: Sequence[str] | None = None
  where_statements: Sequence[str] | None = None
  test_bigquery_table_path: str | None = None
  test_label_col_name: str | None = None
  test_dataset_holdout_fraction: float = 0.2
  data_test_gcs_uri: str | None = None
  upload_only: bool = False
  output_bigquery_table_path: str | None = None
  data_output_gcs_uri: str | None = None
  voting_strategy: VotingStrategy = VotingStrategy.UNANIMOUS
  alpha: float = 1.0
  alpha_negative_pseudolabels: float | None = None
  batches_per_model: int = 1
  max_occ_batch_size: int = 50000
  labeling_and_model_training_batch_size: int | None = None
  ensemble_count: int = 5
  n_components: tuple[int, ...] = (1,)
  covariance_type: str = 'full'
  random_seed: int = _RANDOM_SEED
  verbose: bool = False

  def __post_init__(self):
    """Validates the parameters and sets default values."""
    # Parameter checks.
    if not (self.input_bigquery_table_path or self.data_input_gcs_uri):
      raise ValueError(
          '`input_bigquery_table_path` or `data_input_gcs_uri` must be set.'
      )
    if self.input_bigquery_table_path and self.data_input_gcs_uri:
      raise ValueError(
          'Both`input_bigquery_table_path` and `data_input_gcs_uri` should not '
          'be set.'
      )
    if not self.train_setting:
      raise ValueError('`train_setting` must be set.')
    if not self.output_gcs_uri:
      raise ValueError('`output_gcs_uri` must be set.')
    if not self.label_col_name:
      raise ValueError('`label_col_name` must be set.')
    if (
        (self.positive_data_value == self.negative_data_value)
        or (self.unlabeled_data_value == self.positive_data_value)
        or (self.unlabeled_data_value == self.negative_data_value)
    ):
      raise ValueError(
          '`positive_data_value`, `negative_data_value` and'
          ' `unlabeled_data_value` must all be different from each other.'
      )
    if self.labels_are_strings and not self._check_labels_are_strings():
      raise TypeError(
          '`labels_are_strings` must be True if `positive_data_value`, '
          '`negative_data_value` and `unlabeled_data_value` are strings.'
      )
    # Adjust the labels if needed.
    if not self.labels_are_strings:
      self.positive_data_value = int(self.positive_data_value)
      self.negative_data_value = int(self.negative_data_value)
      self.unlabeled_data_value = int(self.unlabeled_data_value)
    # If `alpha_negative_pseudolabels` is not set, set it to `alpha`.
    if self.alpha_negative_pseudolabels is None:
      self.alpha_negative_pseudolabels = self.alpha

  def _check_labels_are_strings(self) -> bool:
    """Returns True if the labels are strings."""
    return (
        isinstance(self.positive_data_value, str)
        and isinstance(self.negative_data_value, str)
        and isinstance(self.unlabeled_data_value, str)
    )
