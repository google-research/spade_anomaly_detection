load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(
    default_applicable_licenses = ["//third_party/py/spade_anomaly_detection:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# File build rules.
pytype_strict_library(
    name = "bq_dataset",
    srcs = ["bq_dataset.py"],
    deps = [
        ":bq_utils",
        ":feature_metadata",
        "//third_party/py/google/cloud/bigquery",
        "//third_party/py/google/cloud/bigquery_storage",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "bq_utils",
    srcs = ["bq_utils.py"],
)

pytype_strict_library(
    name = "feature_metadata",
    srcs = ["feature_metadata.py"],
    deps = [
        ":bq_utils",
        "//third_party/py/google/cloud/bigquery",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "bq_dataset_test_utils",
    srcs = ["bq_dataset_test_utils.py"],
    deps = ["//third_party/py/pandas"],
)

# Test build rules.
py_strict_test(
    name = "bq_dataset_test",
    srcs = ["bq_dataset_test.py"],
    deps = [
        ":bq_dataset",
        ":bq_dataset_test_utils",
        ":bq_utils",
        ":feature_metadata",
        "//third_party/py/google/cloud/bigquery",
        "//third_party/py/google/cloud/bigquery_storage",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/parameterized",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "bq_utils_test",
    srcs = ["bq_utils_test.py"],
    deps = [
        ":bq_utils",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/parameterized",
    ],
)

py_strict_test(
    name = "feature_metadata_test",
    srcs = ["feature_metadata_test.py"],
    deps = [
        ":feature_metadata",
        "//third_party/py/frozendict",
        "//third_party/py/google/cloud/bigquery",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/numpy",
        "//third_party/py/parameterized",
    ],
)
