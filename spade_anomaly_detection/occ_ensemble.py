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

"""Creates and trains one class classifiers to use in the SPADE architecture."""

# Using typing instead of collections due to Vertex training containers
# not supporting them.

import dataclasses
from typing import Final, MutableMapping, Optional, Sequence

from absl import logging
import numpy as np
from sklearn import mixture
from spade_anomaly_detection import parameters
import tensorflow as tf


@dataclasses.dataclass
class PseudolabelsContainer:
  """Container to hold the outputs of the pseudolabeling process.

  Attributes:
    new_features: np.ndarray of features for the new pseudolabeled data.
    new_labels: np.ndarray of labels for the new pseudolabeled data.
    weights: np.ndarray of weights for the new pseudolabeled data.
    pseudolabel_flags: np.ndarray of flags indicating whether the data point is
      ground truth or pseudolabeled.
  """

  new_features: np.ndarray
  new_labels: np.ndarray
  weights: np.ndarray
  pseudolabel_flags: np.ndarray


_RANDOM_SEED: Final[int] = 42
_SHUFFLE_BUFFER_SIZE: Final[int] = 10_000
_LABEL_TYPE: Final[str] = 'INT64'


# TODO(b/247116870): Create abstract class for templating out future OCC models.
class GmmEnsemble:
  """Class for creating and training a Gaussian mixture model ensemble.

  Initializes attributes for creating single GMMs used in the ensemble, as well
  as methods for fitting models and scoring new data.

  Attributes:
    n_components: Number of mixture components.
    covariance_type: Covariance matrix to create, can be one of full, diag,
      spherical, or tied.
    init_params: Method used to initialize the input matrix. For kmeans, the k
      will be equal to the number of mixture components (n_components).
    max_iter: Maximum number of loops to run in the EM algorithm.
    ensemble_count: Number of Gaussian mixture models to create and run on the
      entire training dataset. This will be used to score samples and classify
      them as anomalies. Note that the more models in the ensemble, the harder
      it is to gain consensus.
    positive_threshold: Float between [0, 100] used as the percentile for the
      one class classifier ensemble to label a point as positive. The closer to
      0 this value is set, the less positive data will be labeled. However, we
      expect an increase in precision when lowering the value, and an increase
      in recall when raising it. Equavalent to saying the given data point needs
      to be located in the top X percentile in order to be considered anomalous.
    negative_threshold: Float between [0, 100] used as the percentile for the
      one class classifier ensemble to label a point as negative. The higher
      this value is set, the less negative data will be labeled. A value in the
      range of 60-95 is a good starting point. We expect an increase in
      precision when raising this value, and an increase in recall when lowering
      it. Equavalent to saying the given data point needs to be X percentile or
      greater in order to be considered anomalous.
    voting_strategy: The voting strategy to use when determining if a data point
      is anomalous. By default, we use unanimous voting, meaning all the models
      in the ensemble need to agree in order to label a data point as anomalous.
    unlabeled_record_count: The number of unlabeled records in the dataset.
    negative_record_count: The number of negative records in the dataset.
    unlabeled_data_value: The value used in the label column to denote unlabeled
      data.
    negative_data_value: The value used in the label column to denote negative
      data.
    verbose: Boolean denoting whether to send model performance and
      pseudo-labeling metrics to the GCP console.
    ensemble: A trained ensemble of one class classifiers.
  """

  # TODO(b/247116870): Create dataclass when another OCC is added.
  def __init__(
      self,
      n_components: tuple[int, ...] = (1,),
      covariance_type: str = 'full',
      init_params: str = 'kmeans',
      max_iter: int = 100,
      ensemble_count: int = 5,
      positive_threshold: float = 1.0,
      negative_threshold: float = 95.0,
      voting_strategy: parameters.VotingStrategy = (
          parameters.VotingStrategy.UNANIMOUS
      ),
      random_seed: int = _RANDOM_SEED,
      unlabeled_record_count: int | None = None,
      negative_record_count: int | None = None,
      unlabeled_data_value: int | None = None,
      negative_data_value: int | None = None,
      verbose: bool = False,
  ) -> None:
    self.n_components = n_components
    self.covariance_type = covariance_type
    self.init_params = init_params
    self.max_iter = max_iter
    self.ensemble_count = ensemble_count
    self.positive_threshold = positive_threshold
    self.negative_threshold = negative_threshold
    self.voting_strategy = voting_strategy
    self._random_seed = random_seed
    self.unlabeled_record_count = unlabeled_record_count
    self.negative_record_count = negative_record_count
    self.unlabeled_data_value = unlabeled_data_value
    self.negative_data_value = negative_data_value
    self.verbose = verbose

    self.ensemble = []

    self._warm_start = False

  def _get_model(self, idx: int) -> mixture.GaussianMixture:
    """Instantiates a Gaussian mixture model.

    Args:
      idx: The index of the model in the ensemble.

    Returns:
      Gaussian mixture model with class attributes.
    """
    return mixture.GaussianMixture(
        n_components=(
            self.n_components[idx]
            if len(self.n_components) == self.ensemble_count
            else self.n_components[0]
        ),
        covariance_type=self.covariance_type,
        init_params=self.init_params,
        warm_start=self._warm_start,
        max_iter=self.max_iter,
        random_state=self._random_seed,
    )

  def _get_filter_by_label_value_func(self, label_column_filter_value: int):
    """Returns a function that filters a record based on the label column value.

    Args:
      label_column_filter_value: The value of the label column to use as a
        filter. If None, all records are included.

    Returns:
      A function that returns True if the label column value is equal to the
      label_column_filter_value parameter.
    """

    def filter_func(features: tf.Tensor, label: tf.Tensor) -> bool:  # pylint: disable=unused-argument
      if label_column_filter_value is None:
        return True
      label_cast = tf.cast(label, tf.dtypes.as_dtype(_LABEL_TYPE.lower()))
      label_column_filter_value_cast = tf.cast(
          label_column_filter_value, label_cast.dtype
      )
      broadcast_equal = tf.equal(label_column_filter_value_cast, label_cast)
      return tf.reduce_all(broadcast_equal)

    return filter_func

  def is_batched(self, dataset: tf.data.Dataset) -> bool:
    """Returns True if the dataset is batched."""
    # This suffices for the current use case of the OCC ensemble.
    return len(dataset.element_spec[0].shape) == 2 and (
        dataset.element_spec[0].shape[0] is None
        or isinstance(dataset.element_spec[0].shape[0], int)
    )

  # Fit with -ve included in every batch to GMMs.
  def fit(
      self,
      train_x: tf.data.Dataset,
      batches_per_occ: int,
  ) -> Sequence[mixture.GaussianMixture]:
    """Creates and fits and ensemble of one class classifiers.

    This model should only be called using the negative and unlabeled data.

    Args:
      train_x: Tensorflow dataset to use as input in creating the OCCs. This
        dataset will be used in training all models in the ensemble.
      batches_per_occ: The number of training batches to use in fitting each
        OCC.

    Returns:
      A list containing all trained models in the ensemble.
    """
    # If there is just one batch for a single OCC, we can leave
    # warm_start=False, else we need it enabled.
    if batches_per_occ > 1:
      self._warm_start = True

    has_batches = self.is_batched(train_x)
    logging.info('has_batches is %s', has_batches)
    negative_features = None

    if (
        not self.unlabeled_record_count
        or not self.negative_record_count
        or not has_batches
        or self.unlabeled_data_value is None
        or self.negative_data_value is None
    ):
      # Either the dataset is not batched, or we don't have all the details to
      # extract the negative-labeled data. Hence we will use all the data for
      # training.
      dataset_iterator = train_x.as_numpy_iterator()
    else:
      # We unbatch the dataset so that we can separate-out the unlabeled and
      # negative data points
      ds_unbatched = train_x.unbatch()

      ds_unlabeled = ds_unbatched.filter(
          self._get_filter_by_label_value_func(self.unlabeled_data_value)
      )

      ds_negative = ds_unbatched.filter(
          self._get_filter_by_label_value_func(self.negative_data_value)
      )

      negative_features_and_labels_zip = list(
          zip(*ds_negative.as_numpy_iterator())
      )

      negative_features = (
          negative_features_and_labels_zip[0]
          if len(negative_features_and_labels_zip) == 2
          else None
      )

      if negative_features is None:
        # The negative features were not extracted. This can happen when the
        # dataset elements are not tuples of features and labels. So we will use
        # all the data for training.
        ds_batched = train_x
      else:
        # The negative features were extracted. How we can proceed with creating
        # batches of unlabeled data, to which the negative data will be added
        # before training.
        batch_size = (
            self.unlabeled_record_count // self.ensemble_count
        ) // batches_per_occ
        ds_batched = ds_unlabeled.batch(
            batch_size,
            drop_remainder=False,
        )
      dataset_iterator = ds_batched.as_numpy_iterator()

    for idx in range(self.ensemble_count):
      model = self._get_model(idx=idx)

      for _ in range(batches_per_occ):
        features, _ = dataset_iterator.next()
        all_features = (
            np.concatenate([features, negative_features], axis=0)
            if negative_features is not None
            else features
        )
        model.fit(all_features)

      self.ensemble.append(model)

    # Reset warm start in case fit is called multiple times with different
    # parameters on the same GmmEnsemble object.
    self._warm_start = False

    return self.ensemble

  def _vote(self, model_scores: np.ndarray) -> np.ndarray:
    """Votes on whether a data point is anomalous or not.

    Args:
      model_scores: The scores for each model in the ensemble for a given data
        point. Can be the positive score or the negative score.

    Returns:
      True if the data point is anomalous, False otherwise.
    """
    if self.voting_strategy == parameters.VotingStrategy.UNANIMOUS:
      return model_scores == self.ensemble_count
    elif self.voting_strategy == parameters.VotingStrategy.MAJORITY:
      return model_scores > self.ensemble_count // 2
    else:
      raise ValueError(
          f'Unsupported voting strategy: {self.voting_strategy}. Supported'
          ' strategies are UNANIMOUS and MAJORITY.'
      )

  def _score_unlabeled_data(
      self,
      unlabeled_features: np.ndarray,
  ) -> MutableMapping[str, np.ndarray]:
    """Labels data as a positive (anomalous) or negative (normal).

    Given an unlabled dataset and percentiles, determines the indices of the
    dataset that are believed to be positive or normal samples. The dataset will
    then be updated and uploaded to BigQuery where a supervised model will be
    trained and deployed. Default percentiles should be changed depending on the
    data that is passed in, and can have a significant effect on the
    performance. Also note that every model in the ensemble needs to agree in
    order for a data point to be classified as anomalous or normal.

    Args:
      unlabeled_features: Numpy array consisting of unlabeled data.

    Returns:
      A mutable mapping composed of positive indices and negative indices for
      the unlabled dataset.
    """
    # TODO(b/247116870): Update this method to take a batch size so that we can
    # perform inference on large datasets. Ideally, everything will be done
    # using indices, and data is updated in BigQuery with the new label.

    # Initialize arrays that we will use to keep the score for each row vector.
    model_scores_pos = np.zeros(len(unlabeled_features))
    model_scores_neg = np.zeros(len(unlabeled_features))

    for model in iter(self.ensemble):
      unlabeled_scores = model.score_samples(unlabeled_features)

      thresh_pos = np.percentile(unlabeled_scores, self.positive_threshold)
      thresh_neg = np.percentile(unlabeled_scores, self.negative_threshold)

      binary_scores_pos = (unlabeled_scores < thresh_pos)
      binary_scores_neg = (unlabeled_scores > thresh_neg)

      model_scores_pos += binary_scores_pos
      model_scores_neg += binary_scores_neg

    positive_indices = np.where(self._vote(model_scores_pos))[0]
    negative_indices = np.where(self._vote(model_scores_neg))[0]

    return {
        'positive_indices': positive_indices,
        'negative_indices': negative_indices
    }

  def pseudo_label(
      self,
      features: np.ndarray,
      labels: np.ndarray,
      positive_data_value: str | int,
      negative_data_value: str | int | None,
      unlabeled_data_value: str | int,
      alpha: float = 1.0,
      alpha_negative_pseudolabels: float = 1.0,
      verbose: Optional[bool] = False,
  ) -> PseudolabelsContainer:
    """Labels unlabeled data using the trained ensemble of OCCs.

    Args:
      features: A numpy array of features, already pre-processed using a min-max
        scaler.
      labels: A numpy array of labels. Strings or integers can be used for
        denoting the positive, negative, or unlabeled features.
      positive_data_value: The value used in the label column to denote positive
        data - data points that are anomalous.
      negative_data_value: The value used in the label column to denote negative
        data - data points that are not anomalous.
      unlabeled_data_value: The value used in the label column to denote
        unlabeled data.
      alpha: This value is used to adjust the influence of the positively pseudo
        labeled data in training the supervised model.
      alpha_negative_pseudolabels: This value is used to adjust the influence of
        the negatively pseudo labeled data in training the supervised model.
      verbose: Chooses the amount of logging info to display. This can be useful
        when debugging model performance.

    Returns:
      A container including updated features (features for which we now have
      labels for), updated labels (includes pseudo labeled positive and negative
      values, as well as ground truth), the weights (correct alpha values)
      for the new pseudo labeled data points, a binary flag that indicates
      whether the data point is newly pseudolabeled, or ground truth. Labels are
      in the format of 1 for positive and 0 for negative. Flag is 1 for
      pseudo-labeled and 0 for ground truth.
    """
    original_positive_idx = np.where(labels == positive_data_value)[0]
    original_negative_idx = np.where(labels == negative_data_value)[0]
    original_unlabeled_idx = np.where(labels == unlabeled_data_value)[0]

    updated_indices = self._score_unlabeled_data(
        unlabeled_features=features[original_unlabeled_idx, :])

    new_positive_indices = updated_indices['positive_indices']
    new_negative_indices = updated_indices['negative_indices']

    new_positive_features = features[
        original_unlabeled_idx[new_positive_indices]]
    new_negative_features = features[
        original_unlabeled_idx[new_negative_indices]]
    positive_features = features[original_positive_idx]
    negative_features = features[original_negative_idx]

    # Build the new feature, label, and weight set to send to a supervised
    # model.
    new_features = np.concatenate([
        new_positive_features,
        new_negative_features,
        positive_features,
        negative_features,
    ],
                                  axis=0)
    new_labels = np.concatenate(
        [
            np.full(len(new_positive_indices), positive_data_value),
            np.full(len(new_negative_indices), negative_data_value),
            np.full(len(original_positive_idx), positive_data_value),
            np.full(len(original_negative_idx), negative_data_value),
        ],
        axis=0,
    )
    weights = np.concatenate(
        [
            np.repeat(alpha, len(new_positive_indices)),
            np.repeat(alpha_negative_pseudolabels, len(new_negative_indices)),
            np.ones([len(original_positive_idx)]),
            np.ones([len(original_negative_idx)]),
        ],
        axis=0,
    )
    pseudolabel_flags = np.concatenate(
        [
            np.ones(len(new_positive_indices)),
            np.ones(len(new_negative_indices)),
            np.zeros(len(original_positive_idx)),
            np.zeros(len(original_negative_idx)),
        ],
        axis=0,
    )

    if verbose:
      logging.info('Number of new positive labels: %s',
                   len(new_positive_indices))
      logging.info('Number of new negative labels: %s',
                   len(new_negative_indices))

    return PseudolabelsContainer(
        new_features=new_features,
        new_labels=new_labels,
        weights=weights,
        pseudolabel_flags=pseudolabel_flags,
    )
