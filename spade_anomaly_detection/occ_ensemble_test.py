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

"""Tests for the one class classifier ensemble."""

from absl.testing import parameterized
import numpy as np

from spade_anomaly_detection import data_loader
from spade_anomaly_detection import occ_ensemble
from spade_anomaly_detection import parameters
import tensorflow as tf


class OccEnsembleTest(tf.test.TestCase, parameterized.TestCase):

  def test_ensemble_initialization_no_error(self):
    gmm_ensemble = occ_ensemble.GmmEnsemble(n_components=1, ensemble_count=10)

    with self.subTest(name='ObjectInit'):
      self.assertIsInstance(gmm_ensemble, occ_ensemble.GmmEnsemble)
    with self.subTest(name='ObjectAttributes'):
      self.assertEqual(gmm_ensemble.ensemble_count, 10)

  # Params to test: n_components, ensemble_count, covariance_type.
  @parameterized.named_parameters(
      ('components_1_ensemble_10_full', (1,), 10, 'full'),
      ('components_3_ensemble_5_full', (3,), 5, 'full'),
      ('components_3_ensemble_5_tied', (3,), 5, 'tied'),
      (
          'components_1_2_3_ensemble_3_tied',
          (
              1,
              2,
              3,
          ),
          5,
          'tied',
      ),
  )
  def test_ensemble_training_no_error(
      self, n_components, ensemble_count, covariance_type
  ):
    batches_per_occ = 10

    ensemble_obj = occ_ensemble.GmmEnsemble(
        n_components=n_components,
        ensemble_count=ensemble_count,
        covariance_type=covariance_type,
    )

    tf_dataset = data_loader.load_tf_dataset_from_csv(
        dataset_name='covertype_pu_labeled', batch_size=None
    )

    features_len = tf_dataset.cardinality().numpy()

    records_per_occ = features_len // ensemble_obj.ensemble_count
    batch_size = records_per_occ // batches_per_occ

    tf_dataset = tf_dataset.shuffle(batch_size).batch(
        batch_size, drop_remainder=True
    )

    ensemble_models = ensemble_obj.fit(tf_dataset, batches_per_occ)

    self.assertLen(
        ensemble_models,
        ensemble_obj.ensemble_count,
        msg='Model count in ensemble not equal to specified ensemble size.',
    )

  # Params to test: n_components, ensemble_count, covariance_type.
  @parameterized.named_parameters(
      ('components_1_ensemble_10_full', (1,), 10, 'full'),
      ('components_1_ensemble_5_tied', (1,), 5, 'tied'),
      ('components_1and3_ensemble_5_tied', (1, 3), 5, 'tied'),
  )
  def test_ensemble_training_unlabeled_negative_no_error(
      self, n_components, ensemble_count, covariance_type
  ):
    batches_per_occ = 10
    negative_data_value = 0
    unlabeled_data_value = -1

    ensemble_obj = occ_ensemble.GmmEnsemble(
        n_components=n_components,
        ensemble_count=ensemble_count,
        covariance_type=covariance_type,
        negative_data_value=negative_data_value,
        unlabeled_data_value=unlabeled_data_value,
    )

    tf_dataset = data_loader.load_tf_dataset_from_csv(
        dataset_name='covertype_pnu_100000', batch_size=None
    )
    # These are the actual counts of unlabeled and negative records in the
    # dataset.
    unlabeled_record_count = 94950
    negative_record_count = 4333
    ensemble_obj.unlabeled_record_count = unlabeled_record_count
    ensemble_obj.negative_record_count = negative_record_count

    features_len = tf_dataset.cardinality().numpy()
    records_per_occ = features_len // ensemble_obj.ensemble_count
    batch_size = records_per_occ // batches_per_occ

    tf_dataset = tf_dataset.shuffle(batch_size).batch(
        batch_size, drop_remainder=True
    )

    ensemble_models = ensemble_obj.fit(tf_dataset, batches_per_occ)

    self.assertLen(
        ensemble_models,
        ensemble_obj.ensemble_count,
        msg='Model count in ensemble not equal to specified ensemble size.',
    )

  def test_dataset_filtering(self):
    positive_data_value = 1
    negative_data_value = 0
    unlabeled_data_value = -1
    gmm_ensemble = occ_ensemble.GmmEnsemble(n_components=1, ensemble_count=10)

    tf_dataset = data_loader.load_tf_dataset_from_csv(
        dataset_name='covertype_pnu_100000', batch_size=None
    )
    tf_unlabeled_dataset = tf_dataset.filter(
        gmm_ensemble._get_filter_by_label_value_func(unlabeled_data_value)
    )
    tf_negative_dataset = tf_dataset.filter(
        gmm_ensemble._get_filter_by_label_value_func(negative_data_value)
    )
    tf_positive_dataset = tf_dataset.filter(
        gmm_ensemble._get_filter_by_label_value_func(positive_data_value)
    )
    self.assertEqual(
        tf_unlabeled_dataset.reduce(0, lambda x, _: x + 1).numpy(),
        94950,
    )
    self.assertEqual(
        tf_negative_dataset.reduce(0, lambda x, _: x + 1).numpy(),
        4333,
    )
    self.assertEqual(
        tf_positive_dataset.reduce(0, lambda x, _: x + 1).numpy(),
        715,
    )

  # Parameters to test: label_type, voting_strategy.
  @parameterized.named_parameters(
      ('labels_are_integers_unanimous_voting', False, 'unanimous'),
      ('labels_are_strings_unanimous_voting', True, 'unanimous'),
      ('labels_are_integers_majority_voting', False, 'majority'),
      ('labels_are_strings_majority_voting', True, 'majority'),
  )
  def test_score_unlabeled_data_no_error(
      self, labels_are_strings: bool, voting_strategy: str
  ):
    batches_per_occ = 1
    positive_threshold = 2
    negative_threshold = 90
    alpha = 0.8
    alpha_negative_pseudolabels = 0.8

    if labels_are_strings:
      positive_data_value = b'1'
      negative_data_value = b'0'
      unlabeled_data_value = b'-1'
    else:
      positive_data_value = 1
      negative_data_value = 0
      unlabeled_data_value = -1

    occ_train_dataset = data_loader.load_tf_dataset_from_csv(
        dataset_name='drug_train_pu_labeled',
        batch_size=None,
        # Coerce `unlabeled_data_value` to int since the test dataset contains
        # only integer labels.
        filter_label_value=int(unlabeled_data_value),
    )
    features_len = occ_train_dataset.cardinality().numpy()
    if labels_are_strings:
      # Treat the labels as strings for testing. Note that the test dataset
      # contains only integer labels.
      occ_train_dataset = occ_train_dataset.map(
          lambda x, y: (x, tf.as_string(y))
      )

    ensemble_obj = occ_ensemble.GmmEnsemble(
        n_components=(1,),
        ensemble_count=5,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        voting_strategy=parameters.VotingStrategy(voting_strategy),
    )

    records_per_occ = features_len // ensemble_obj.ensemble_count
    occ_train_dataset = occ_train_dataset.batch(
        records_per_occ // batches_per_occ, drop_remainder=True
    )
    ensemble_obj.fit(occ_train_dataset, batches_per_occ)

    occ_train_dataset = data_loader.load_tf_dataset_from_csv(
        dataset_name='drug_train_pu_labeled',
        batch_size=500,
        filter_label_value=None,
    )
    if labels_are_strings:
      # Treat the labels as strings for testing. Note that the test dataset
      # contains only integer labels.
      occ_train_dataset = occ_train_dataset.map(
          lambda x, y: (x, tf.as_string(y))
      )
    features, labels = occ_train_dataset.as_numpy_iterator().next()

    label_count_before_labeling = len(
        np.where(
            (labels == negative_data_value) | (labels == positive_data_value)
        )[0]
    )

    pseudolabels_container = ensemble_obj.pseudo_label(
        features=features,
        labels=labels,
        alpha=alpha,
        alpha_negative_pseudolabels=alpha_negative_pseudolabels,
        positive_data_value=positive_data_value,
        negative_data_value=negative_data_value,
        unlabeled_data_value=unlabeled_data_value,
    )
    updated_features = pseudolabels_container.new_features
    updated_labels = pseudolabels_container.new_labels
    weights = pseudolabels_container.weights
    pseudolabel_flags = pseudolabels_container.pseudolabel_flags

    label_count_after_labeling = len(
        np.where(
            (updated_labels == negative_data_value)
            | (updated_labels == positive_data_value)
        )[0]
    )

    new_label_count = label_count_after_labeling - label_count_before_labeling
    updated_weights = np.concatenate(
        (
            np.where(
                (pseudolabel_flags == 1)
                & (updated_labels == positive_data_value)
            )[0],
            np.where(
                (pseudolabel_flags == 1)
                & (updated_labels == negative_data_value)
            )[0],
        ),
        axis=0,
    )
    updated_weight_count = len(updated_weights)

    with self.subTest(name='AlphaValuesLabels'):
      self.assertEqual(
          new_label_count,
          updated_weight_count,
          msg=(
              'The number of alpha values is not equal to the number of new '
              'labels.'
          ),
      )

    with self.subTest(name='MoreLabelsCreated'):
      # We could enforce that X labels are created here, revisit this later if
      # needed.
      self.assertGreater(
          label_count_after_labeling,
          label_count_before_labeling,
          msg='Label count after labeling was not more than before.',
      )

    with self.subTest(name='AlphaValuesCorrespondToPseudoLabels'):
      # Note that this test will fail if the alpha value is 1.0 (the ground
      # truth weight).
      positive_cond = (pseudolabel_flags == 1) & (
          updated_labels == positive_data_value
      )
      negative_cond = (pseudolabel_flags == 1) & (
          updated_labels == negative_data_value
      )
      weights_are_alpha_pos = np.where(positive_cond)[0]
      weights_are_alpha_neg = np.where(negative_cond)[0]
      weights_are_alpha = np.concatenate(
          (weights_are_alpha_pos, weights_are_alpha_neg), axis=0
      )
      pseudolabel_flags_are_1 = np.where(pseudolabel_flags == 1)[0]
      self.assertNDArrayNear(
          weights_are_alpha,
          pseudolabel_flags_are_1,
          err=1e-6,
          msg=(
              'The data samples where the weights are equal to the alpha '
              'values are not the same as the samples where the pseudolabel '
              'flags are equal to 1.'
          ),
      )

    with self.subTest(name='LabelFeatureArraysEqual'):
      self.assertLen(updated_labels, len(updated_features))
    with self.subTest(name='LabelWeightArraysEqualLen'):
      self.assertLen(updated_labels, len(weights))
    with self.subTest(name='PseudolabelWeightArraysEqualLen'):
      self.assertLen(pseudolabel_flags, len(weights))


if __name__ == '__main__':
  tf.test.main()
