# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

This allow to:

* auto-parsing release notes during the automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Have clickable headers in the rendered markdown

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [2.0.0] - YYYY-MM-DD` header and add the current
  `[Unreleased]` notes.
* At the end of the file:
  * Define the new link url:
  `[2.0.0]: https://github.com/google-research/spade_anomaly_detection/compare/v1.0.0...v2.0.0`
  * Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v2.0.0...HEAD`
* If updating the PyPi version, also update the `__version__` variable in the
  `__init__.py` file at the root of the module.
-->

## [Unreleased]

## [0.3.3] - 2024-08-05

* Add support for wildcards in GCS URIs in CSV data loader.
* Upgrade Pandas to 2.2.2.

## [0.3.2] - 2024-07-16

* Exposes the `n_component` and `covariance_type` parameters of the one-class classifier.

## [0.3.1] - 2024-07-13

* Now writes out the pseudolabel weights and a flag that indicates whether a sample has a ground truth label (0) or a pseudolabel (1).

## [0.3.0] - 2024-07-10

* Add the ability to use CSV files on GCS as data input/output/test sources.
* Miscellaneous bugfixes.

## [0.2.2] - 2024-05-19

* Update the OCC training to use negative and unlabeled samples for training.

## [0.2.1] - 2024-05-18

* Updates to data loaders. Label column filter can now be a list of integers.

## [0.2.0] - 2024-05-05

* Add PyPi support. Minor reorganization of repository.

## [0.1.0] - 2024-04-17

* Initial release

[Unreleased]: https://github.com/google-research/spade_anomaly_detection/compare/v0.3.3...HEAD
[0.3.3]: https://github.com/google-research/spade_anomaly_detection/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/google-research/spade_anomaly_detection/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/google-research/spade_anomaly_detection/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/google-research/spade_anomaly_detection/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/google-research/spade_anomaly_detection/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/google-research/spade_anomaly_detection/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/google-research/spade_anomaly_detection/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/google-research/spade_anomaly_detection/releases/tag/v0.1.0
