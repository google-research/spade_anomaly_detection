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

# Note: Replace parameters in [] brackets with your values.

set -x

IMAGE_TAG=${1:-"latest-oss"}
IMAGE_NAME=${2:-"spade"}
PROJECT_ID=${3:-"[insert-project-id]"}
REPO_NAME=${5:-"spade"}
IMAGE_BASE_URI=${4:-"us-docker.pkg.dev"}

# Image for team testing.
IMAGE_URI_ML="${IMAGE_BASE_URI}/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

BUILD_IMAGE="true"

if "${BUILD_IMAGE}"; then
  DOCKER_BUILDKIT=1 docker build -f ./Dockerfile -t "${IMAGE_URI_ML}" .
fi

docker push "${IMAGE_URI_ML}"

echo "Built and pushed ${IMAGE_URI_ML}"

