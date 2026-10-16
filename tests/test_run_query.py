from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_query import run_query


class RunQueryTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.db = self.root / "trace.duckdb"
        with duckdb.connect(str(self.db)) as connection:
            connection.execute(
                "CREATE TABLE ev AS SELECT * FROM (VALUES (0, 'k', 10.0), (1, 'k', 30.0)) t(rank, name, dur)"
            )
        self.sql = self.root / "01-dur.sql"
        self.sql.write_text(
            "-- comment\nSET VARIABLE rank = 0;\n"
            "SELECT rank, sum(dur) AS total FROM ev WHERE rank = getvariable('rank') GROUP BY rank;\n",
            encoding="utf-8",
        )

    def test_runs_set_lines_then_returns_last_result(self) -> None:
        self.assertEqual(run_query(self.db, self.sql, []), (["rank", "total"], [(0, 10.0)]))

    def test_override_wins_over_file_variable(self) -> None:
        self.assertEqual(run_query(self.db, self.sql, [("rank", "1")]), (["rank", "total"], [(1, 30.0)]))

    def test_cli_writes_csv_next_to_sql(self) -> None:
        import subprocess

        subprocess.run(
            [sys.executable, "-I", str(SCRIPTS_DIR / "run_query.py"), str(self.sql), "--db", str(self.db)],
            check=True, capture_output=True, text=True,
        )
        with self.sql.with_suffix(".csv").open(newline="") as handle:
            self.assertEqual(list(csv.reader(handle)), [["rank", "total"], ["0", "10.0"]])


if __name__ == "__main__":
    unittest.main()
