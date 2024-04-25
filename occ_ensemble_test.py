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

import numpy as np

from spade_anomaly_detection import data_loader
from spade_anomaly_detection import occ_ensemble

import tensorflow as tf


class OccEnsembleTest(tf.test.TestCase):

  def test_ensemble_initialization_no_error(self):
    gmm_ensemble = occ_ensemble.GmmEnsemble(n_components=1, ensemble_count=10)

    with self.subTest(name='ObjectInit'):
      self.assertIsInstance(gmm_ensemble, occ_ensemble.GmmEnsemble)
    with self.subTest(name='ObjectAttributes'):
      self.assertEqual(gmm_ensemble.ensemble_count, 10)

  def test_ensemble_training_no_error(self):
    batches_per_occ = 10
    ensemble_count = 5
    n_components = 1

    ensemble_obj = occ_ensemble.GmmEnsemble(
        n_components=n_components, ensemble_count=ensemble_count
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

  def test_score_unlabeled_data_no_error(self):
    batches_per_occ = 1
    positive_threshold = 2
    negative_threshold = 90
    positive_data_value = 1
    negative_data_value = 0
    unlabeled_data_value = -1
    alpha = 0.8

    occ_train_dataset = data_loader.load_tf_dataset_from_csv(
        dataset_name='drug_train_pu_labeled',
        batch_size=None,
        filter_label_value=unlabeled_data_value,
    )
    features_len = occ_train_dataset.cardinality().numpy()

    ensemble_obj = occ_ensemble.GmmEnsemble(
        n_components=1,
        ensemble_count=5,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
    )

    records_per_occ = features_len // ensemble_obj.ensemble_count
    occ_train_dataset = occ_train_dataset.batch(
        records_per_occ // batches_per_occ, drop_remainder=True
    )
    ensemble_obj.fit(occ_train_dataset, batches_per_occ)

    features, labels = (
        data_loader.load_tf_dataset_from_csv(
            dataset_name='drug_train_pu_labeled',
            batch_size=500,
            filter_label_value=None,
        )
        .as_numpy_iterator()
        .next()
    )

    label_count_before_labeling = len(
        np.where((labels == 0) | (labels == 1))[0]
    )

    updated_features, updated_labels, weights = ensemble_obj.pseudo_label(
        features=features,
        labels=labels,
        alpha=alpha,
        positive_data_value=positive_data_value,
        negative_data_value=negative_data_value,
        unlabeled_data_value=unlabeled_data_value,
    )

    label_count_after_labeling = len(
        np.where((updated_labels == 0) | (updated_labels == 1))[0]
    )

    new_label_count = label_count_after_labeling - label_count_before_labeling
    updated_weight_count = len(np.where(weights == alpha)[0])

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

    with self.subTest(name='LabelFeatureArraysEqual'):
      self.assertLen(updated_labels, len(updated_features))
    with self.subTest(name='LabelWeightArraysEqual'):
      self.assertLen(updated_labels, len(weights))


if __name__ == '__main__':
  tf.test.main()
