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

"""Executes the semi-supervised framework with user defined options.

Runs the implementation of the SPADE framework, which is focused on
anomaly detection. In general, there is a semi-supervised step where we use
a series of one class classifiers to label unlabled data and then use the new
data set to batch fit a supervised model.
"""

# TODO(b/247116870): Change to collections when Vertex supports python 3.9
from typing import Optional, Tuple

from absl import logging
import numpy as np
import pandas as pd

from spade_anomaly_detection import data_loader
from spade_anomaly_detection import occ_ensemble
from spade_anomaly_detection import parameters
from spade_anomaly_detection import supervised_model


class Runner:
  """Runs the SPADE algorithm with values in the RunnerParameters instance.

  Attributes:
    runner_parameters: An instance of the runner_parameters dataclass. This
      holds all parameter information for running a training job.
    test_x: Evaluation features, this is None when there is no evaluation set.
    test_y: Evaluation features, this is None when there is no evaluation set.
    supervised_model_object: None when upload only is True, otherwise we
      instantiate a supervised model object to train on after pseudo labeling.
    supervised_model_metrics: Metrics for the supervised model to be evaluated
      on. By default, this is AUC.
    data_loader: An instance of the data loader, useful for performing a number
      of operations such as loading, filtering, and uploading of bigquery data.
  """

  def __init__(self, runner_parameters: parameters.RunnerParameters):
    self.runner_parameters = runner_parameters
    logging.info('Runner parameters: %s', self.runner_parameters)

    self.data_loader = data_loader.DataLoader(self.runner_parameters)

    # TODO(b/247116870): Evaluate performance implications of using a global
    # testing array - the machine may not have enough memory to store the test
    # set in addition to iterating over the training set.
    self.test_x: Optional[np.ndarray] = None
    self.test_y: Optional[np.ndarray] = None

    if not self.runner_parameters.upload_only:
      self.supervised_model_metrics: Optional[dict[str, float]] = None
      supervised_model_parameters = supervised_model.RandomForestParameters()
      self.supervised_model_object = supervised_model.RandomForestModel(
          supervised_model_parameters
      )
    else:
      self.supervised_model_object = None

    if (
        self.runner_parameters.positive_threshold is None
        or self.runner_parameters.negative_threshold is None
    ):
      input_table_statistics = self.data_loader.get_label_thresholds(
          self.runner_parameters.input_bigquery_table_path
      )

      self.runner_parameters.positive_threshold = (
          input_table_statistics['positive_threshold']
          if self.runner_parameters.positive_threshold is None
          else self.runner_parameters.positive_threshold
      )
      self.runner_parameters.negative_threshold = (
          input_table_statistics['negative_threshold']
          if self.runner_parameters.negative_threshold is None
          else self.runner_parameters.negative_threshold
      )

  def check_data_tables(
      self,
      total_record_count: int,
      unlabeled_record_count: int,
  ) -> None:
    """Runs sanity checks on the table that is passed to this algorithm.

    Args:
      total_record_count: Count of all records in the BigQuery table.
      unlabeled_record_count: Number of unlabeled records in the table.
    """
    if not total_record_count:
      raise ValueError(
          'There are no records in the table: '
          f'{self.runner_parameters.input_bigquery_table_path}'
      )
    elif total_record_count < self.runner_parameters.ensemble_count:
      raise ValueError(
          'There are not enough records in the table to fit one '
          'record per model in the ensemble. '
          f'Total records: {total_record_count}'
      )

    if total_record_count < 1000:
      logging.warning(
          (
              'Using a small number of examples to train the model, results '
              'will vary significantly between runs. Total records: '
              ' %i'
          ),
          total_record_count,
      )

    if unlabeled_record_count == total_record_count:
      raise ValueError(
          'There were no labels found in the dataset. Check the'
          'value passed in for the label value and label column'
          'are correct if your datasets has labels.'
      )

    if (
        self.runner_parameters.labeling_and_model_training_batch_size
        is not None
    ):
      if self.runner_parameters.labeling_and_model_training_batch_size < 1:
        raise ValueError(
            'labeling_and_model_training_batch_size must be set to '
            'a value greater than 0.'
        )

      if (
          self.runner_parameters.labeling_and_model_training_batch_size
          > total_record_count
      ):
        raise ValueError(
            'labeling_and_model_training_batch_size can not be greater than '
            f'the total number of examples. There are {total_record_count} '
            'examples and batch size is set to '
            f'{self.runner_parameters.labeling_and_model_training_batch_size}.'
        )

  def instantiate_and_fit_ensemble(
      self, unlabeled_record_count: int
  ) -> occ_ensemble.GmmEnsemble:
    """Creates and fits an OCC ensemble on the specified input data.

    Args:
      unlabeled_record_count: Number of unlabeled records in the table.

    Returns:
      A trained one class classifier ensemble.
    """

    ensemble_object = occ_ensemble.GmmEnsemble(
        ensemble_count=self.runner_parameters.ensemble_count,
        positive_threshold=self.runner_parameters.positive_threshold,
        negative_threshold=self.runner_parameters.negative_threshold,
    )

    records_per_occ = unlabeled_record_count // ensemble_object.ensemble_count
    batch_size = records_per_occ // self.runner_parameters.batches_per_model
    batch_size = np.min([batch_size, self.runner_parameters.max_occ_batch_size])

    logging.info('Batch size for OCC ensemble: %s', batch_size)

    unlabeled_data = self.data_loader.load_tf_dataset_from_bigquery(
        input_path=self.runner_parameters.input_bigquery_table_path,
        label_col_name=self.runner_parameters.label_col_name,
        where_statements=self.runner_parameters.where_statements,
        ignore_columns=self.runner_parameters.ignore_columns,
        batch_size=batch_size,
        label_column_filter_value=self.runner_parameters.unlabeled_data_value,
    )

    logging.info('Fitting ensemble.')
    ensemble_object.fit(
        train_x=unlabeled_data,
        batches_per_occ=self.runner_parameters.batches_per_model,
    )
    logging.info('Ensemble fit complete.')

    return ensemble_object

  def write_verbose_logs(
      self,
      features: np.ndarray,
      labels: np.ndarray,
      weights: np.ndarray,
  ) -> None:
    """Writes logs to Cloud Logging.

    Args:
      features: Set of all labeled features (after pseudo labels added).
      labels: Set of all positive and negative labels.
      weights: Weights corresponding to pseudo labels - this is the alpha
        parameter.
    """
    updated_label_counts = pd.DataFrame(labels).value_counts()
    logging.info('Updated label counts %s', updated_label_counts)

    logging.info('Test features shape: %s',
                 self.test_x.shape if self.test_x is not None else None)
    logging.info('Test labels shape: %s',
                 self.test_y.shape if self.test_y is not None else None)

    logging.info('Updated features shape: %s', features.shape)
    logging.info('Updated labels shape: %s', labels.shape)
    logging.info('Weights shape: %s', weights.shape)

    logging.info('Features sample: %s', features[:1])
    logging.info('Labels sample: %s', labels[:1])
    logging.info('Weights sample: %s', weights[:1])

  def evaluate_model(
      self,
      batch_number: Optional[int] = 0,
  ) -> None:
    """Evaluates the supervised model and writes logs to Cloud Logging.

    Args:
      batch_number: Integer corresponding to the batch of pseudo labeling and
        supervised model training the loop is in. We use this for calculating
        running averages for various metrics.
    """

    if self.test_x is None:
      raise ValueError(
          'There is no test set to evaluate on, double check '
          'the testing table or train/test split fraction.'
      )

    if self.supervised_model_object is None:
      raise ValueError(
          'Evaluate called without a trained supervised model. Ensure that '
          'run() has been called and the algorithm is not being ran in upload '
          'only mode.'
      )

    else:
      eval_results = self.supervised_model_object.supervised_model.evaluate(
          x=self.test_x, y=self.test_y, return_dict=True
      )
      if batch_number == 0:
        self.supervised_model_metrics = eval_results
        logging.info(
            'Supervised model evaluation %s',
            eval_results,
        )
      else:
        for metric_name, metric_value in eval_results.items():
          if self.supervised_model_metrics is not None:
            self.supervised_model_metrics[metric_name] += (
                metric_value - self.supervised_model_metrics[metric_name]
            ) / (batch_number + 1)

        logging.info(
            'Supervised model evaluation for current batch %s',
            eval_results,
        )

        logging.info(
            'Supervised model average evaluation: %s',
            self.supervised_model_metrics,
        )

  def _check_runner_parameters(self) -> None:
    """Checks parameter that are not related to the data tables.

    When developing workflows here, set parameters and warn users as
    needed. Throw errors when an essential parameter is missing or out of a
    required range.
    """
    if self.runner_parameters.test_dataset_holdout_fraction > 1:
      raise ValueError(
          'Can not use more than 100% of the data for the test '
          'set:'
          f' test_dataset_holdout_fraction={self.runner_parameters.test_dataset_holdout_fraction}'
      )

    if self.runner_parameters.test_dataset_holdout_fraction > 0.5:
      logging.warning(
          (
              'Using more than 50%% of the data for a test set, this can lead'
              ' to negative performance implications.'
              ' test_dataset_holdout_fraction = %s'
          ),
          self.runner_parameters.test_dataset_holdout_fraction,
      )
    if (
        self.runner_parameters.test_dataset_holdout_fraction > 0
        and self.runner_parameters.test_bigquery_table_path
    ):
      logging.warning(
          'Only a test holdout fraction and a single input table '
          'or an input and test table may be specified. Using the '
          'test table instead of the specified holdout fraction.'
      )
      self.runner_parameters.test_dataset_holdout_fraction = 0

    if self.runner_parameters.upload_only and (
        self.runner_parameters.test_dataset_holdout_fraction
        or self.runner_parameters.test_bigquery_table_path
    ):
      logging.warning(
          'A test set is not needed in upload only mode, '
          'test_dataset_holdout_fraction and test_bigquery_table_path will be '
          'ignored.'
      )
      self.runner_parameters.test_dataset_holdout_fraction = 0
      self.runner_parameters.test_bigquery_table_path = ''
      self.runner_parameters.output_gcs_uri = ''

    if (
        self.runner_parameters.upload_only
        and not self.runner_parameters.output_bigquery_table_path
    ):
      raise ValueError(
          'output_bigquery_table_path needs to be specified in '
          'upload_only mode.'
      )

  def preprocess_train_test_split(
      self,
      features: np.ndarray,
      labels: np.ndarray,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Creates the train and test sets depending on the train setting.

    Args:
      features: Testing features, numpy array.
      labels: Testing labels, numpy array.

    Returns:
      A sequence consisting of processed training and test sets in the order of
      train_x, train_y.
    """
    if len(features) != len(labels):
      raise ValueError(
          'Feature and label arrays have different lengths. '
          f'{len(features)} != {len(labels)}'
      )
    # Shuffle arrays, best practice before creating train/test splits.
    random_indices = np.random.permutation(len(features))
    features = features[random_indices]
    labels = labels[random_indices]

    if self.runner_parameters.train_setting == parameters.TrainSetting.PNU:
      ground_truth_label_indices = np.where(
          labels != self.runner_parameters.unlabeled_data_value
      )[0]
      label_count = int(
          len(ground_truth_label_indices)
          * self.runner_parameters.test_dataset_holdout_fraction
      )
      test_index_subset = np.random.choice(
          ground_truth_label_indices, size=label_count, replace=False
      )

      train_x = np.delete(features, test_index_subset, axis=0)
      train_y = np.delete(labels, test_index_subset, axis=0)

      test_x = features[test_index_subset]
      test_y = labels[test_index_subset]

    # TODO(b/247116870): Investigate the performance implications and user
    # interest for including some of the negative data in the training set
    # where we re-label as unlabeled data. This could improve the accuracy of
    # the end supervised classifier.
    elif self.runner_parameters.train_setting == parameters.TrainSetting.PU:
      # Uses all ground truth negative labels and the correct proportion of
      # positive labels for testing.
      positive_indices = np.where(
          labels == self.runner_parameters.positive_data_value
      )[0]
      negative_indices = np.where(
          labels == self.runner_parameters.negative_data_value
      )[0]

      positive_label_count = int(
          len(positive_indices)
          * self.runner_parameters.test_dataset_holdout_fraction
      )
      test_positive_indices = np.random.choice(
          positive_indices, size=positive_label_count, replace=False
      )

      train_x = np.delete(
          features,
          np.concatenate([test_positive_indices, negative_indices], axis=0),
          axis=0,
      )
      train_y = np.delete(
          labels,
          np.concatenate([test_positive_indices, negative_indices], axis=0),
          axis=0,
      )

      test_x = np.concatenate(
          [features[negative_indices], features[test_positive_indices]], axis=0
      )
      test_y = np.concatenate(
          [labels[negative_indices], labels[test_positive_indices]], axis=0
      )

    else:
      raise ValueError(
          'Unknown train setting for preparing train/test '
          f'datasets: {self.runner_parameters.train_setting}'
      )

    # TODO(b/247116870): Implement a dedicated function in the runner class
    # to load BQ test sets before evaluating the supervised model.
    if self.runner_parameters.test_bigquery_table_path:
      # Remove any unlabeled samples that may be in the test set.
      unlabeled_sample_filter = (
          f'{self.runner_parameters.test_label_col_name} != '
          f'{self.runner_parameters.unlabeled_data_value}'
      )
      if self.runner_parameters.where_statements is not None:
        unlabeled_sample_where_statements = (
            self.runner_parameters.where_statements + [unlabeled_sample_filter]
        )
      else:
        unlabeled_sample_where_statements = [unlabeled_sample_filter]
      test_dataset_size = self.data_loader.get_query_record_result_length(
          input_path=self.runner_parameters.test_bigquery_table_path,
          where_statements=unlabeled_sample_where_statements,
      )
      test_tf_dataset = self.data_loader.load_tf_dataset_from_bigquery(
          input_path=self.runner_parameters.test_bigquery_table_path,
          label_col_name=self.runner_parameters.test_label_col_name,
          where_statements=unlabeled_sample_where_statements,
          ignore_columns=self.runner_parameters.ignore_columns,
          batch_size=test_dataset_size,
      )

      test_x, test_y = test_tf_dataset.as_numpy_iterator().next()
      self.test_x = np.array(test_x)
      self.test_y = np.array(test_y)

      if not (
          np.any(test_y == self.runner_parameters.positive_data_value)
          and np.any(test_y == self.runner_parameters.negative_data_value)
      ):
        raise ValueError(
            'Positive and negative labels must be in the testing set. Please '
            'check the test table provided in the test_bigquery_table_path '
            'parameter.'
        )
    else:
      if self.test_x is not None:
        self.test_x = np.concatenate([self.test_x, test_x], axis=0)
        self.test_y = np.concatenate([self.test_y, test_y], axis=0)
      else:
        self.test_x = test_x
        self.test_y = test_y

    # Adjust the testing labels to values of 1 and 0 to align with the class
    # the supervised model is trained on.
    self.test_y[self.test_y == self.runner_parameters.positive_data_value] = 1
    self.test_y[self.test_y == self.runner_parameters.negative_data_value] = 0

    return (train_x, train_y)

  def train_supervised_model(
      self,
      *,
      features: np.ndarray,
      labels: np.ndarray,
      weights: np.ndarray,
  ) -> None:
    """Trains a supervised model.

    This function can be called in a batch manner if the supervised model
    supports batch updates.

    Args:
      features: Pseudo labeled and ground truth features.
      labels: Pseudo and ground truth labels.
      weights: Weights for pseudo labeled data.

    Raises:
      ValueError if there is the supervised model object is None.
    """
    if self.supervised_model_object is None:
      raise ValueError('No supervised model to train.')
    logging.info('Supervised model training started.')
    self.supervised_model_object.train(
        features=features, labels=labels, weights=weights
    )
    logging.info('Supervised model training completed.')

  def run(self) -> None:
    """Runs the anomaly detection algorithm on a BigQuery table."""
    logging.info('SPADE training started.')

    self._check_runner_parameters()

    total_record_count = self.data_loader.get_query_record_result_length(
        input_path=self.runner_parameters.input_bigquery_table_path,
        where_statements=self.runner_parameters.where_statements,
    )

    unlabeled_record_count_filter = (
        f'{self.runner_parameters.label_col_name} = '
        f'{self.runner_parameters.unlabeled_data_value}'
    )
    if self.runner_parameters.where_statements:
      unlabeled_record_count_where_statements = [
          self.runner_parameters.where_statements
      ] + [unlabeled_record_count_filter]
    else:
      unlabeled_record_count_where_statements = [unlabeled_record_count_filter]

    unlabeled_record_count = self.data_loader.get_query_record_result_length(
        input_path=self.runner_parameters.input_bigquery_table_path,
        where_statements=unlabeled_record_count_where_statements,
    )

    self.check_data_tables(
        total_record_count=total_record_count,
        unlabeled_record_count=unlabeled_record_count,
    )

    ensemble_object = self.instantiate_and_fit_ensemble(
        unlabeled_record_count=unlabeled_record_count
    )

    batch_size = (
        self.runner_parameters.labeling_and_model_training_batch_size
        or total_record_count
    )
    tf_dataset = self.data_loader.load_tf_dataset_from_bigquery(
        input_path=self.runner_parameters.input_bigquery_table_path,
        label_col_name=self.runner_parameters.label_col_name,
        where_statements=self.runner_parameters.where_statements,
        ignore_columns=self.runner_parameters.ignore_columns,
        batch_size=batch_size,
    )
    tf_dataset = tf_dataset.as_numpy_iterator()

    for batch_number, (features, labels) in enumerate(tf_dataset):
      logging.info(
          'Labeling and supervised model training batch number: %s',
          batch_number,
      )
      logging.info(
          'Batch size: %s',
          len(features),
      )

      train_x, train_y = self.preprocess_train_test_split(
          features=features,
          labels=labels,
      )

      logging.info('Labeling started.')
      updated_features, updated_labels, weights = ensemble_object.pseudo_label(
          features=train_x,
          labels=train_y,
          positive_data_value=self.runner_parameters.positive_data_value,
          negative_data_value=self.runner_parameters.negative_data_value,
          unlabeled_data_value=self.runner_parameters.unlabeled_data_value,
          alpha=self.runner_parameters.alpha,
          verbose=self.runner_parameters.verbose,
      )
      logging.info('Labeling completed.')

      # Upload batch of pseudo labels, will append when called more than once.
      if self.runner_parameters.output_bigquery_table_path:
        self.data_loader.upload_dataframe_as_bigquery_table(
            features=updated_features, labels=updated_labels
        )

      # TODO(b/247116870): Create two logging functions, one for batch and one
      # for the end of SPADE training (reporting job level metrics such as AUC).
      if self.runner_parameters.verbose:
        self.write_verbose_logs(
            features=updated_features,
            labels=updated_labels,
            weights=weights,
        )

      if not self.runner_parameters.upload_only:
        self.train_supervised_model(
            features=updated_features,
            labels=updated_labels,
            weights=weights,
        )

    if not self.runner_parameters.upload_only:
      self.evaluate_model()
      if self.supervised_model_object is not None:
        self.supervised_model_object.save(
            save_location=self.runner_parameters.output_gcs_uri
        )

    logging.info('SPADE training completed.')
