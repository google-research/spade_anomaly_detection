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

"""Supervised models for SPADE."""
import abc
import contextlib
import dataclasses
from typing import Any, Optional

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_decision_forests as tfdf


# TODO(b/247116870): Add models such as MLP so that we can update the parameters
# incrementally via batches of data. TFDF does not support batch training.
class Model(abc.ABC):
  """Abstract class to ensure a consistent API surface for future models."""

  def __init__(self, parameters: Any) -> None:
    self.supervised_parameters = parameters
    self.supervised_model = None

  @abc.abstractmethod
  def train(
      self,
      features: np.ndarray,
      labels: np.ndarray,
      weights: Optional[np.ndarray],
  ) -> tf.keras.callbacks.History:
    pass

  def save(self, save_location: str) -> None:
    """Saves TensorFlow model assets to a GCS location.

    Args:
      save_location: String denoting a Google Cloud Storage location, or local
        disk path. Note that local assets will be deleted when the VM running
        this container is shutdown at the end of the training job.
    """
    if self.supervised_model is not None:
      self.supervised_model.save(save_location)
      logging.info('Saved model assets to %s', save_location)
    else:
      logging.warning('No model to save.')


@dataclasses.dataclass
class RandomForestParameters:
  """Dataclass for the RandomForest TF model.

  Attributes:
    task: Keras task - I have defaulted to classification since we are
      predicting a discrete label.
    num_trees: The number of individual decision trees in the forest.
    min_examples: Minimum number of examples per node.
    max_num_nodes: Maximum number of nodes in a tree. There is no max as a
      default.
    max_depth: Maximum depth of each tree, a depth of 1 would make all nodes
      roots for example.
    random_seed: Random seed for training the model. Learners are deterministic
      when you pass in the same seed.
  """

  task: tfdf.keras.Task = tfdf.keras.Task.CLASSIFICATION
  num_trees: int = 300
  min_examples: int = 5
  max_num_nodes: Optional[int] = None
  max_depth: int = 16
  random_seed: int = 0


class RandomForestModel(Model):
  """RandomForest implemented in TensorFlow."""

  def __init__(
      self,
      parameters: RandomForestParameters,
      distribution_strategy: Optional[tf.distribute.Strategy] = None,
  ):
    """Initializes a random forest model.

    Args:
      parameters: An initialized RandomForestParameters object set with the
        desired values for a RandomForest model.
      distribution_strategy: TensorFlow distribution strategy to use when
        training the random forest model. This is useful on larger datasets,
        where training time is not satisfactory. Refer to the Vertex Custom
        training jobs and TensorFlow distribution strategy for choosing the
        right method here. Default is no strategy, ie use a single worker node.
        It may be worthwhile to attempt to scale up before scaling out, since
        there is typically higher overhead when scaling out.
    """

    super().__init__(parameters)

    # Note on performance: As of tfdf 1.0.0, hardware accelerators are not yet
    # supported. CPU performance for training and inference should be sufficient
    # for most use cases.
    with (
        distribution_strategy.scope()
        if distribution_strategy
        else contextlib.nullcontext()
    ):
      self.supervised_model = tfdf.keras.RandomForestModel(
          **dataclasses.asdict(self.supervised_parameters)
      )
      self.supervised_model.compile(
          metrics=[
              tf.keras.metrics.AUC(name='Supervised_Model_AUC'),
              tf.keras.metrics.Precision(
                  thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  name='Supervised_Model_Precision',
              ),
              tf.keras.metrics.Recall(
                  thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  name='Supervised_Model_Recall',
              ),
          ]
      )

  def train(
      self,
      features: np.ndarray,
      labels: np.ndarray,
      weights: Optional[np.ndarray] = None,
  ) -> tf.keras.callbacks.History:
    """Trains a random forest model.

    Args:
      features: Numpy array of input features - numerical and scaled values
        only.
      labels: Array of labels corresponding to rows of features. The numerical
        label we are trying to predict for the given task.
      weights: Array of sample weights containing values between 0 and 1, where
        1 has the most influence and 0 the least influence on training. These
        values also correspond to rows of features.

    Returns:
      The callback history from the call to fit. This can be useful for
      inspecting loss and model convergence.
    """
    # Note: Batch fitting is not supported in TFDF Random Forest, even though
    # it is in the public documentation.
    return self.supervised_model.fit(
        x=features,
        y=labels,
        sample_weight=weights,
    )
