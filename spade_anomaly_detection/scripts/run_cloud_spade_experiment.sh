#!/bin/bash
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

# Builds a docker container, pushes it to Google Cloud, and executes it as a
# training job on Vertex.

# Run this from the base project directory.
# Note: Replace parameters in [] brackets with your values.

set -x

PROJECT_ID=${1:-"[insert-project-id]"}
DATETIME=$(date '+%Y%m%d_%H%M%S')

#Args
TRAIN_SETTING=${2:-"PNU"}

# Use either Bigquery or GCS for input/output/test data.
INPUT_BIGQUERY_TABLE_PATH=${3:-"${PROJECT_ID}.[bq-dataset].[bq-input-table]"}
DATA_INPUT_GCS_URI=${4:-""}
OUTPUT_BIGQUERY_TABLE_PATH=${5:-"${PROJECT_ID}.[bq-dataset].[bq-output-table]"}
DATA_OUTPUT_GCS_URI=${6:-""}
OUTPUT_GCS_URI=${7:-"gs://[gcs-bucket]/[model-folder]"}
LABEL_COL_NAME=${8:-"y"}
# The label column is of type float, these must match in order for array
# filtering to work correctly.
POSITIVE_DATA_VALUE=${9:-"1"}
NEGATIVE_DATA_VALUE=${10:-"0"}
UNLABELED_DATA_VALUE=${11:-"-1"}
POSITIVE_THRESHOLD=${12:-".1"}
NEGATIVE_THRESHOLD=${13:-"95"}
TEST_BIGQUERY_TABLE_PATH=${14:-"${PROJECT_ID}.[bq-dataset].[bq-test-table]"}
DATA_TEST_GCS_URI=${15:-""}
TEST_LABEL_COL_NAME=${16:-"y"}
ALPHA=${17:-"1.0"}
BATCHES_PER_MODEL=${18:-"1"}
ENSEMBLE_COUNT=${19:-"5"}
N_COMPONENTS=${20:-"1"}
COVARIANCE_TYPE=${21:-"full"}
MAX_OCC_BATCH_SIZE=${22:-"50000"}
LABELING_AND_MODEL_TRAINING_BATCH_SIZE=${23:-"100000"}
VERBOSE=${24:-"True"}
UPLOAD_ONLY=${25:-"False"}

# Give a unique name to your training job.
TRIAL_NAME="spade_${USER}_${DATETIME}"

# Image name and location
IMAGE_NAME="spade"
IMAGE_TAG=${26:-"latest-oss"}
# Project image (use this for testing)
IMAGE_URI="us-docker.pkg.dev/${PROJECT_ID}/spade/${IMAGE_NAME}:${IMAGE_TAG}"
echo "IMAGE_URI = ${IMAGE_URI}"

BUILD=${27:-"TRUE"}

if [[ "${BUILD}" == "TRUE" ]]; then
  /bin/bash ./scripts/build_and_push_image.sh "${IMAGE_TAG}" "${IMAGE_NAME}" "${PROJECT_ID}" || exit
fi

REGION="us-central1"
WORKER_MACHINE="machine-type=n1-standard-16"


# Launch the job.
gcloud ai custom-jobs create \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --display-name="${TRIAL_NAME}" \
  --worker-pool-spec="${WORKER_MACHINE}",replica-count=1,container-image-uri="${IMAGE_URI}" \
  --args=--train_setting="${TRAIN_SETTING}" \
  --args=--input_bigquery_table_path="${INPUT_BIGQUERY_TABLE_PATH}" \
  --args=--data_input_gcs_uri="${DATA_INPUT_GCS_URI}" \
  --args=--output_bigquery_table_path="${OUTPUT_BIGQUERY_TABLE_PATH}" \
  --args=--data_output_gcs_uri="${DATA_OUTPUT_GCS_URI}" \
  --args=--output_gcs_uri="${OUTPUT_GCS_URI}" \
  --args=--label_col_name="${LABEL_COL_NAME}" \
  --args=--positive_data_value="${POSITIVE_DATA_VALUE}" \
  --args=--negative_data_value="${NEGATIVE_DATA_VALUE}" \
  --args=--unlabeled_data_value="${UNLABELED_DATA_VALUE}" \
  --args=--positive_threshold="${POSITIVE_THRESHOLD}" \
  --args=--negative_threshold="${NEGATIVE_THRESHOLD}" \
  --args=--test_bigquery_table_path="${TEST_BIGQUERY_TABLE_PATH}" \
  --args=--data_test_gcs_uri="${DATA_TEST_GCS_URI}" \
  --args=--test_label_col_name="${TEST_LABEL_COL_NAME}" \
  --args=--alpha="${ALPHA}" \
  --args=--batches_per_model="${BATCHES_PER_MODEL}" \
  --args=--ensemble_count="${ENSEMBLE_COUNT}" \
  --args=--n_components="${N_COMPONENTS}" \
  --args=--covariance_type="${COVARIANCE_TYPE}" \
  --args=--max_occ_batch_size="${MAX_OCC_BATCH_SIZE}" \
  --args=--labeling_and_model_training_batch_size="${LABELING_AND_MODEL_TRAINING_BATCH_SIZE}" \
  --args=--upload_only="${UPLOAD_ONLY}" \
  --args=--verbose="${VERBOSE}"
