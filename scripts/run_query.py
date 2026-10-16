#!/usr/bin/env python3
"""Run a saved trace query against a DuckDB trace database and save its result as CSV.

The SQL file may contain several statements (for example `SET VARIABLE` lines before the final
query); the result of the last statement is written next to the SQL file with a `.csv` suffix,
so every number cited from the evidence pack can be rerun unchanged on another trace.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import duckdb


def parse_assignment(value: str) -> tuple[str, str]:
    name, separator, literal = value.partition("=")
    if not separator or not name.strip().isidentifier():
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {value!r}")
    return name.strip(), literal.strip()


def run_query(
    database: Path, sql_path: Path, assignments: list[tuple[str, str]]
) -> tuple[list[str], list[tuple]]:
    """Execute the file's statements, apply `--set` overrides, and return the last result."""
    connection = duckdb.connect(str(database), read_only=True)
    try:
        statements = connection.extract_statements(sql_path.read_text(encoding="utf-8"))
        if not statements:
            raise ValueError(f"{sql_path} contains no SQL statements")
        for statement in statements[:-1]:
            connection.execute(statement)
        # Overrides win over the file's own SET VARIABLE lines.
        for name, literal in assignments:
            connection.execute(f"SET VARIABLE {name} = {literal}")
        connection.execute(statements[-1])
        if connection.description is None:
            raise ValueError(f"the last statement in {sql_path} returns no rows")
        columns = [column[0] for column in connection.description]
        return columns, connection.fetchall()
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sql", type=Path, help="saved query, e.g. evidence/r0/queries/03-gaps.sql")
    parser.add_argument("--db", type=Path, required=True, help="DuckDB file (trace-sql.md section 1)")
    parser.add_argument(
        "--set",
        dest="assignments",
        action="append",
        type=parse_assignment,
        default=[],
        help="override a variable after the file's own SET VARIABLE lines, e.g. --set rank=1",
    )
    parser.add_argument("--output", type=Path, help="CSV path; default: the SQL path with .csv")
    parser.add_argument("--print-rows", type=int, default=10, help="rows to echo to stdout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.sql.with_suffix(".csv")
    try:
        columns, rows = run_query(args.db, args.sql, args.assignments)
    except (OSError, ValueError, duckdb.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"{len(rows):,} rows -> {output}")
    print(",".join(columns))
    for row in rows[: args.print_rows]:
        print(",".join("" if value is None else str(value) for value in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
