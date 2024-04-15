# Google Cloud's optimized Tensorflow image
FROM gcr.io/deeplearning-platform-release/tf-cpu.2-10

# Alternative Tensorflow image with GPU.
# FROM gcr.io/deeplearning-platform-release/tf-gpu.2-10:latest

# Alternative vanilla Tensorflow image
# FROM tensorflow/tensorflow:2.8.1-gpu

WORKDIR /

RUN python -m pip install --upgrade pip

ARG BASE_ML_DIR=.

# Make sure `pip install ...` happens before `COPY`. This makes
# sure `pip install ...` is cached, so the docker build speed is improved. It's
# important to move as much heavy step (such as `pip install`) before
# `COPY` to cache as much build steps as possible.
ARG PIP_REQUIREMENTS=/spade_anomaly_detection/${BASE_ML_DIR}/requirements.txt
COPY ${BASE_ML_DIR}/requirements.txt ${PIP_REQUIREMENTS}
RUN pip install --no-cache-dir -r ${PIP_REQUIREMENTS}

# TODO(b/333154677): Figure out how to do this based off BUILD files.
COPY ${BASE_ML_DIR}/data_utils /spade_anomaly_detection/${BASE_ML_DIR}/data_utils
COPY ${BASE_ML_DIR} /spade_anomaly_detection/${BASE_ML_DIR}

# TODO(b/333154677): Figure out if these are needed.
# Enable userspace DNS cache
ENV GCS_RESOLVE_REFRESH_SECS=60
ENV GCS_REQUEST_CONNECTION_TIMEOUT_SECS=300
ENV GCS_METADATA_REQUEST_TIMEOUT_SECS=300
ENV GCS_READ_REQUEST_TIMEOUT_SECS=300
ENV GCS_WRITE_REQUEST_TIMEOUT_SECS=600
# Each opened GCS file takes GCS_READ_CACHE_BLOCK_SIZE_MB of RAM, reduce the
# value from the default 64MB to 8MB to decrease memory footprint.
ENV GCS_READ_CACHE_BLOCK_SIZE_MB=8

# TODO(b/333154677): Figure out if this is needed.
RUN mkdir /model
RUN mkdir -p /export

# Change pythonpath so that it prefers the full path. This was needed to
# get pytest to work from the base directory of the container.
ENV PYTHONPATH "${WORKDIR}:${PYTHONPATH}"

ENTRYPOINT ["python", "-m", "spade_anomaly_detection.task"]
