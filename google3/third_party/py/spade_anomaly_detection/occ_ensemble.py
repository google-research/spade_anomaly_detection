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

from typing import MutableMapping, Sequence, Optional

from absl import logging
import numpy as np
from sklearn import mixture
import tensorflow as tf


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
    verbose: Boolean denoting whether to send model performance and
      pseudo-labeling metrics to the GCP console.
    ensemble: A trained ensemble of one class classifiers.
  """

  # TODO(b/247116870): Create dataclass when another OCC is added.
  def __init__(
      self,
      n_components: int = 1,
      covariance_type: str = 'full',
      init_params: str = 'kmeans',
      max_iter: int = 100,
      ensemble_count: int = 5,
      positive_threshold: float = 1.0,
      negative_threshold: float = 95.0,
      verbose: bool = False,
  ) -> None:
    self.n_components = n_components
    self.covariance_type = covariance_type
    self.init_params = init_params
    self.max_iter = max_iter
    self.ensemble_count = ensemble_count
    self.positive_threshold = positive_threshold
    self.negative_threshold = negative_threshold
    self.verbose = verbose

    self.ensemble = []

    self._warm_start = False

  def _get_model(self) -> mixture.GaussianMixture:
    """Instantiates a Gaussian mixture model.

    Returns:
      Gaussian mixture model with class attributes.
    """
    return mixture.GaussianMixture(
        n_components=self.n_components,
        covariance_type=self.covariance_type,
        init_params=self.init_params,
        warm_start=self._warm_start,
        max_iter=self.max_iter,
    )

  def fit(
      self, train_x: tf.data.Dataset, batches_per_occ: int
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

    dataset_iterator = train_x.as_numpy_iterator()

    for _ in range(self.ensemble_count):
      model = self._get_model()

      for _ in range(batches_per_occ):
        features, labels = dataset_iterator.next()
        del labels  # Not needed for this task.
        model.fit(features)

      self.ensemble.append(model)

    # Reset warm start in case fit is called multiple times with different
    # parameters on the same GmmEnsemble object.
    self._warm_start = False

    return self.ensemble

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

    positive_indices = np.where(model_scores_pos == self.ensemble_count)[0]
    negative_indices = np.where(model_scores_neg == self.ensemble_count)[0]

    return {
        'positive_indices': positive_indices,
        'negative_indices': negative_indices
    }

  def pseudo_label(self,
                   features: np.ndarray,
                   labels: np.ndarray,
                   positive_data_value: int,
                   negative_data_value: Optional[int],
                   unlabeled_data_value: int,
                   alpha: float = 1.0,
                   verbose: Optional[bool] = False) -> Sequence[np.ndarray]:
    """Labels unlabeled data using the trained ensemble of OCCs.

    Args:
      features: A numpy array of features, already pre-processed using a min-max
        scaler.
      labels: A numpy array of labels. Strings or integers can be used for
        denoting the positive, negative, or unlabeled features.
      positive_data_value: The value used in the label column to denote
        positive data - data points that are anomalous.
      negative_data_value: The value used in the label column to denote
        negative data - data points that are not anomalous.
      unlabeled_data_value: The value used in the label column to denote
        unlabeled data.
      alpha: This value is used to adjust the influence of the pseudo labeled
        data in training the supervised model.
      verbose: Chooses the amount of logging info to display. This can be useful
        when debugging model performance.

    Returns:
      A sequence including updated features (features for which we now have
      labels for), updated labels (includes pseudo labeled positive and negative
      values, as well as ground truth), and the weights (correct alpha values)
      for the new pseudo labeled data points. Labels are in the format of 1 for
      positive and 0 for negative.
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
    new_labels = np.concatenate([
        np.ones(len(new_positive_indices)),
        np.zeros(len(new_negative_indices)),
        np.ones(len(original_positive_idx)),
        np.zeros(len(original_negative_idx))
    ],
                                axis=0)
    weights = np.concatenate([
        np.repeat(alpha, len(new_positive_indices)),
        np.repeat(alpha, len(new_negative_indices)),
        np.ones([len(original_positive_idx)]),
        np.ones([len(original_negative_idx)])
    ],
                             axis=0)

    if verbose:
      logging.info('Number of new positive labels: %s',
                   len(new_positive_indices))
      logging.info('Number of new negative labels: %s',
                   len(new_negative_indices))

    return new_features, new_labels, weights
