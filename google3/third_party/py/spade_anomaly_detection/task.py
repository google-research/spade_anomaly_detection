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

"""This is the entry point for the Docker container to execute the SPADE algo.

See spade/scripts for examples on launching a Vertex training job and using this
script as an entry point.
"""
import logging
from typing import Sequence

from absl import app
from absl import flags

from spade_anomaly_detection import parameters
from spade_anomaly_detection import runner

_TRAIN_SETTING = flags.DEFINE_enum_class(
    "train_setting",
    default="PNU",
    required=False,
    enum_class=parameters.TrainSetting,
    help=(
        "The 'PNU' setting will train the supervised model using ground truth"
        " negative, positive data, and pseudo labeled positive and negative"
        " data. The 'PU' setting will only use negative data from the pseudo"
        " labeler along with the rest of the positive data (ground truth plus"
        " pseudo labeled) to train the supervised model. For model"
        " evaluation,we still require ground truth negative data to be in the"
        " BigQuery dataset, it just won't be used during training."
    ),
)

_INPUT_BIGQUERY_TABLE_PATH = flags.DEFINE_string(
    "input_bigquery_table_path",
    default=None,
    required=True,
    help=(
        "BigQuery table path used for training the anomaly detection model."
        " This needs to be the format 'project.dataset.table'."
    ),
)

_OUTPUT_GCS_URI = flags.DEFINE_string(
    "output_gcs_uri",
    default=None,
    required=True,
    help=(
        "Cloud Storage location to store the supervised model assets. The"
        "location should be in the form gs://bucketname/foldername. A timestamp"
        "will be added to the end of the folder so that multiple runs of this"
        "won't overwrite previous runs."
    ),
)

_LABEL_COL_NAME = flags.DEFINE_string(
    "label_col_name",
    default=None,
    required=True,
    help="The name of the label column in BigQuery.",
)

_POSITIVE_DATA_VALUE = flags.DEFINE_integer(
    "positive_data_value",
    default=None,
    required=True,
    help="The column value used to define an anomalous (positive) data point.",
)

_NEGATIVE_DATA_VALUE = flags.DEFINE_integer(
    "negative_data_value",
    default=None,
    required=True,
    help="The column value used to define a normal (negative) data point.",
)

_UNLABELED_DATA_VALUE = flags.DEFINE_integer(
    "unlabeled_data_value",
    default=None,
    required=True,
    help="The column value used to define an unlabeled data point.",
)

_POSITIVE_THRESHOLD = flags.DEFINE_float(
    "positive_threshold",
    default=None,
    required=False,
    help=(
        "Integer used as the percentile for the one class classifier ensemble "
        "to label a point as positive. The closer to 0 this value is set, the "
        "less positive data will be labeled. However, we expect an increase in "
        "precision when lowering the value, and an increase in recall when "
        "raising it. Equavalent to saying the given data point needs to be "
        "located in the top X percentile in order to be considered anomalous."
    ),
)

_NEGATIVE_THRESHOLD = flags.DEFINE_float(
    "negative_threshold",
    default=None,
    required=False,
    help=(
        "Integer used as the percentile for the one class classifier ensemble"
        " to label a point as negative. The higher this value is set, the less"
        " negative data will be labeled. A value in the range of 50-99 is a"
        " good starting point. We expect an increase in precision when raising"
        " this value, and an increase in recall when lowering it. Equavalent to"
        " saying the given data point needs to be X percentile or greater in"
        " order to be considered anomalous."
    ),
)

_IGNORE_COLUMNS = flags.DEFINE_list(
    "ignore_columns",
    default=None,
    required=False,
    help=(
        "A list of columns located in the input table that you would "
        "like to ignore in both pseudo labeling and supervised model training."
    ),
)

_WHERE_STAEMENTS = flags.DEFINE_list(
    "where_statements",
    default=None,
    required=False,
    help=(
        "Additional SQL where statements with correct syntax that can be "
        "leveraged to filter BigQuery table data. An example is "
        "\"WHERE date > '2008-11-11'\"."
    ),
)

_TEST_BIGQUERY_TABLE_PATH = flags.DEFINE_string(
    "test_bigquery_table_path",
    default=None,
    required=False,
    help=(
        "A complete BigQuery path in the form of 'project.dataset.table' to be "
        "used for evaluating the supervised model. Note that the positive and "
        "negative label values must also be the same in this testing set. It "
        "is okay to to have your test labels in that form, or use 1 for "
        "positive and 0 for negative."
    ),
)

_TEST_LABEL_COL_NAME = flags.DEFINE_string(
    "test_label_col_name",
    default=None,
    required=False,
    help="The label column name in the test dataset.",
)

_TEST_DATASET_HOLDOUT_FRACTION = flags.DEFINE_float(
    "test_dataset_holdout_fraction",
    default=0.3,
    required=False,
    help=(
        "Float between 0 and 1 representing the fraction of samples to hold "
        "out as a test set."
    ),
)

_OUTPUT_BIGQUERY_TABLE_PATH = flags.DEFINE_string(
    "output_bigquery_table_path",
    default=None,
    required=False,
    help=(
        "A complete BigQuery path in the form of 'project.dataset.table' to be"
        " used for uploading the pseudo labeled data. This includes features"
        " and new labels. By default, we will use the column names from the"
        " input_bigquery_table_path BigQuery table."
    ),
)

_ALPHA = flags.DEFINE_float(
    "alpha",
    default=1.0,
    required=False,
    help=(
        "Sample weights for weighting the loss function, only for"
        " pseudo-labeled data from the occ ensemble. Original data that is"
        " labeled will have a weight of 1.0."
    ),
)

_BATCHES_PER_MODEL = flags.DEFINE_integer(
    "batches_per_model",
    default=1,
    required=False,
    help=(
        "The number of batches to use when fitting a single model in the"
        " ensemble. Default is one, meaning that the dataset is divided into"
        " 1/ensemble_count pieces that are used to fit different models."
        " Increasing this value is useful for large datasets that can not fit a"
        " 1/ensemble_count shard in memory."
    ),
)

_MAX_OCC_BATCH_SIZE = flags.DEFINE_integer(
    "max_occ_batch_size",
    default=50000,
    required=False,
    help=(
        "The maximum number of examples in a batch to use when fitting a single"
        "occ model"
    ),
)

_BATCH_SIZE = flags.DEFINE_integer(
    "labeling_and_model_training_batch_size",
    default=None,
    required=False,
    help=(
        "The number of examples to use when pseudo labeling unlabeled training"
        " examples. The subset of labeled examples are then fed to the"
        " supervised model to perform a train step. When there is no batch size"
        " specified here, we will use the entire dataset to score, label, and"
        " train the supervised model. For large datasets, use the"
        " batches_per_model and this setting to reduce the amount of data"
        " stored locally on the training machine."
    ),
)

_ENSEMBLE_COUNT = flags.DEFINE_integer(
    "ensemble_count",
    default=5,
    required=False,
    help=(
        "The number of models to use in the labeling ensemble. The default is"
        " 5, and the more that are added to the ensemble decreases the"
        " probability of all of them agreeing on negative and positive samples."
    ),
)

_VERBOSE = flags.DEFINE_bool(
    "verbose",
    default=False,
    required=False,
    help=(
        "The amount of console logs to display during training. Use False to "
        "show few messages, and True for displaying many aspects of model "
        "training and scoring. This is useful for debugging model performance."
    ),
)

_UPLOAD_ONLY = flags.DEFINE_bool(
    "upload_only",
    default=False,
    required=False,
    help=(
        "Use this setting in conjunction with output_bigquery_table_path. When"
        " True, the algorithm will just upload the pseudo labeled data to the"
        " specified table, and will skip training a supervised model. When set"
        " to False, the algorithm will also train a supervised model and upload"
        " to a GCS endpoint. Default is False."
    ),
)

# TODO(b/247116870) Implement the rest of the input parameters.


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  runner_parameters = parameters.RunnerParameters(
      train_setting=_TRAIN_SETTING.value,
      input_bigquery_table_path=_INPUT_BIGQUERY_TABLE_PATH.value,
      output_gcs_uri=_OUTPUT_GCS_URI.value,
      label_col_name=_LABEL_COL_NAME.value,
      positive_data_value=_POSITIVE_DATA_VALUE.value,
      negative_data_value=_NEGATIVE_DATA_VALUE.value,
      unlabeled_data_value=_UNLABELED_DATA_VALUE.value,
      positive_threshold=_POSITIVE_THRESHOLD.value,
      negative_threshold=_NEGATIVE_THRESHOLD.value,
      ignore_columns=_IGNORE_COLUMNS.value,
      where_statements=_WHERE_STAEMENTS.value,
      test_bigquery_table_path=_TEST_BIGQUERY_TABLE_PATH.value,
      test_label_col_name=_TEST_LABEL_COL_NAME.value,
      test_dataset_holdout_fraction=_TEST_DATASET_HOLDOUT_FRACTION.value,
      upload_only=_UPLOAD_ONLY.value,
      output_bigquery_table_path=_OUTPUT_BIGQUERY_TABLE_PATH.value,
      alpha=_ALPHA.value,
      batches_per_model=_BATCHES_PER_MODEL.value,
      max_occ_batch_size=_MAX_OCC_BATCH_SIZE.value,
      labeling_and_model_training_batch_size=_BATCH_SIZE.value,
      ensemble_count=_ENSEMBLE_COUNT.value,
      verbose=_VERBOSE.value,
  )
  runner_obj = runner.Runner(runner_parameters)
  runner_obj.run()


if __name__ == "__main__":
  try:
    app.run(main)
  except Exception as e:
    logging.shutdown()
    raise e
