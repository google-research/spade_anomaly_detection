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

import re
import unittest

import parameterized

from spade_anomaly_detection.data_utils import bq_utils


class BQTablePathPartsTest(unittest.TestCase):

  def test_raises_for_missing_project(self):
    with self.assertRaises(ValueError):
      bq_utils.BQTablePathParts.from_full_path('dataset.table')

  def test_from_full_path(self):
    full_bq_table_path = 'project_id.dataset.table'

    parts = bq_utils.BQTablePathParts.from_full_path(full_bq_table_path)

    self.assertEqual('project_id', parts.project_id)
    self.assertEqual('dataset', parts.bq_dataset_name)
    self.assertEqual('table', parts.bq_table_name)
    self.assertEqual(full_bq_table_path, parts.full_table_id)

  def test_from_full_path_escaped_path(self):
    full_bq_table_path = '`project_id.dataset.table`'

    parts = bq_utils.BQTablePathParts.from_full_path(full_bq_table_path)

    self.assertEqual('project_id', parts.project_id)
    self.assertEqual('dataset', parts.bq_dataset_name)
    self.assertEqual('table', parts.bq_table_name)
    self.assertEqual(full_bq_table_path, parts.escaped_table_id)


@parameterized.parameterized_class([
    {
        'conjunction': 'and'
    },
    {
        'conjunction': 'or'
    },
])
class WhereStatementFromClausesTest(unittest.TestCase):

  def test_properly_handles_no_clauses(self):
    self.assertEqual(
        '',
        bq_utils.where_statement_from_clauses([], conjunction=self.conjunction))

  def test_properly_handles_one_clauses(self):
    clause = 'a = 1'
    self.assertRegex(
        bq_utils.where_statement_from_clauses([clause],
                                              conjunction=self.conjunction),
        re.compile(r'\s+WHERE\s+\(a = 1\)+\s+'),
        'The message should contain where and the clause with variable spacing.'
    )

  def test_joins_multiple_clauses_with_conjunctions(self):
    clauses = ['a = 1', 'b=2', 'c=b']
    self.assertRegex(
        bq_utils.where_statement_from_clauses(
            clauses, conjunction=self.conjunction),
        re.compile(rf'\s+WHERE\s+\(a = 1\)\s+{self.conjunction}'
                   rf'\s+\(b=2\)\s+{self.conjunction}\s+\(c=b\)\s+'),
    )


if __name__ == '__main__':
  unittest.main()
