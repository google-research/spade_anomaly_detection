# Example Data

## Dataset Licenses

Datasets in `example_data/*` are made available under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/legalcode.txt).

- Covertype Dataset:
  - [Link](https://archive.ics.uci.edu/ml/datasets/covertype)
  - License: Creative Commons Attribution 4.0 International (CC BY 4.0) license.
- Drug Consumption Dataset:
  - [Link](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29)
  - License: Creative Commons Attribution 4.0 International (CC BY 4.0) license.
- Thyroid Disease Dataset:
  - [Link](https://archive.ics.uci.edu/ml/datasets/thyroid+disease)
  - License: Creative Commons Attribution 4.0 International (CC BY 4.0) license.


## Labeling and Data Normalization
- PU is for positive and unlabeled in the label column. 1 is used for positive
and 0 is used for everything else in this case.
- A PNU dataset would have 1 for positive, 0 for unlabeled and -1 for negative
data.
- All datasets have been preprocessed through minmax scaling.

## Thyroid dataset
- The thyroid dataset is a smaller one, and not advised for larger testing. It
is used in the paper however, and will be included here.
- This dataset has already been labeled by the ensemble, and is used for testing
the supervised model.

## Covertype PNU datasets
- covertype_pnu_10000, covertype_pnu_100000: These are labeled with positive,
negative, and unlabeled features. The number at the end of the file name denotes
the number of rows in the file. These are randomly sampled from the original
dataset of ~580k records.
