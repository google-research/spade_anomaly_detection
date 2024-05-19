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

-->

## [Unreleased]

## [0.2.1] - 2024-05-18

* Updates to data loaders. Label column filter can now be a list of integers.

## [0.2.0] - 2024-05-05

* Add PyPi support. Minor reorganization of repository.

## [0.1.0] - 2024-04-17

* Initial release

[Unreleased]: https://github.com/google-research/spade_anomaly_detection/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/google-research/spade_anomaly_detection/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/google-research/spade_anomaly_detection/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/google-research/spade_anomaly_detection/releases/tag/v0.1.0
