from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import Iterable, Iterator


DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d")
ODDS_PAIRS = {
    "ps": (("PSW", "PSL"),),
    "b365": (("B365W", "B365L"),),
    "avg": (("AvgW", "AvgL"),),
    "max": (("MaxW", "MaxL"),),
    "best-available": (("PSW", "PSL"), ("B365W", "B365L"), ("AvgW", "AvgL"), ("MaxW", "MaxL")),
}
VALID_MATCH_COMMENTS = {"", "completed"}
XLSX_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
EXCEL_EPOCH = datetime(1899, 12, 30)


@dataclass(slots=True)
class MatchRecord:
    date: datetime
    tour: str
    surface: str
    player_a: str
    player_b: str
    player_a_odds: float
    player_b_odds: float
    player_a_won: bool
    tournament: str = ""
    series: str = ""
    round_name: str = ""
    court: str = ""
    best_of: int | None = None
    rank_a: int | None = None
    rank_b: int | None = None
    points_a: int | None = None
    points_b: int | None = None
    comment: str = ""
    odds_source: str = ""
    source_file: str = ""


def _parse_date(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("Empty date value")

    try:
        serial = float(text)
    except ValueError:
        serial = None

    if serial is not None:
        return EXCEL_EPOCH + timedelta(days=serial)

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {value!r}")


def _parse_optional_int(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _normalize_surface(value: str) -> str:
    surface = value.strip().title()
    if not surface:
        return "Unknown"
    aliases = {"Carpet Indoor": "Carpet", "Indoor Hard": "Hard", "Outdoor Hard": "Hard"}
    return aliases.get(surface, surface)


def _select_odds(
    row: dict[str, str],
    odds_source: str,
) -> tuple[float, float, str]:
    if odds_source not in ODDS_PAIRS:
        raise ValueError(f"Unsupported odds source: {odds_source}")

    for winner_col, loser_col in ODDS_PAIRS[odds_source]:
        winner_value = row.get(winner_col, "").strip()
        loser_value = row.get(loser_col, "").strip()
        if not winner_value or not loser_value:
            continue
        try:
            winner_odds = float(winner_value)
            loser_odds = float(loser_value)
        except ValueError:
            continue
        if winner_odds <= 1.0 or loser_odds <= 1.0:
            continue
        return winner_odds, loser_odds, winner_col[:-1].lower()
    raise ValueError("No supported odds pair found in row")


def _infer_tour(row: dict[str, str], path: Path) -> str:
    stem = path.stem.casefold()
    if stem.startswith("wta"):
        return "wta"
    if stem.startswith("atp"):
        return "atp"
    if "WTA" in row:
        return "wta"
    if "ATP" in row:
        return "atp"
    return "unknown"


def _column_index(cell_ref: str) -> int:
    letters = []
    for char in cell_ref:
        if char.isalpha():
            letters.append(char.upper())
        else:
            break
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for string_item in root.findall("x:si", XLSX_NS):
        parts = [text_node.text or "" for text_node in string_item.findall(".//x:t", XLSX_NS)]
        values.append("".join(parts))
    return values


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        return "".join(text_node.text or "" for text_node in cell.findall(".//x:t", XLSX_NS))

    value_node = cell.find("x:v", XLSX_NS)
    if value_node is None or value_node.text is None:
        return ""

    value = value_node.text
    if cell_type == "s":
        return shared_strings[int(value)]
    return value


def _load_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row:
                yield {key: (value or "") for key, value in row.items()}


def _load_xlsx_rows(path: Path) -> Iterator[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = _xlsx_shared_strings(archive)
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = sheet.findall(".//x:sheetData/x:row", XLSX_NS)

        headers: dict[int, str] = {}
        for row_index, row in enumerate(rows):
            values_by_index: dict[int, str] = {}
            for cell in row.findall("x:c", XLSX_NS):
                cell_ref = cell.get("r", "")
                if not cell_ref:
                    continue
                values_by_index[_column_index(cell_ref)] = _xlsx_cell_value(cell, shared_strings)

            if row_index == 0:
                headers = {
                    index: value.strip()
                    for index, value in values_by_index.items()
                    if value.strip()
                }
                continue

            if not headers:
                continue

            yield {header: values_by_index.get(index, "") for index, header in headers.items()}


def _iter_rows(path: Path) -> Iterator[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from _load_csv_rows(path)
        return
    if suffix == ".xlsx":
        yield from _load_xlsx_rows(path)
        return


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(candidate) for candidate in glob(pattern)]
        if matches:
            paths.extend(matches)
            continue
        path = Path(pattern)
        if path.exists():
            paths.append(path)
    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    return sorted(unique_paths)


def load_matches(
    paths: Iterable[str | Path],
    odds_source: str = "best-available",
    completed_only: bool = True,
) -> list[MatchRecord]:
    records: list[MatchRecord] = []
    for raw_path in paths:
        path = Path(raw_path)
        for row in _iter_rows(path):
            winner = row.get("Winner", "").strip()
            loser = row.get("Loser", "").strip()
            if not row.get("Date") or not winner or not loser:
                continue

            comment = row.get("Comment", "").strip()
            if completed_only and comment.casefold() not in VALID_MATCH_COMMENTS:
                continue

            try:
                winner_odds, loser_odds, matched_odds_source = _select_odds(row, odds_source)
                match_date = _parse_date(row["Date"])
            except ValueError:
                continue

            winner_rank = _parse_optional_int(row.get("WRank", ""))
            loser_rank = _parse_optional_int(row.get("LRank", ""))
            winner_points = _parse_optional_int(row.get("WPts", ""))
            loser_points = _parse_optional_int(row.get("LPts", ""))
            best_of = _parse_optional_int(row.get("Best of", ""))
            tour = _infer_tour(row, path)

            if winner <= loser:
                player_a = winner
                player_b = loser
                player_a_odds = winner_odds
                player_b_odds = loser_odds
                player_a_won = True
                rank_a, rank_b = winner_rank, loser_rank
                points_a, points_b = winner_points, loser_points
            else:
                player_a = loser
                player_b = winner
                player_a_odds = loser_odds
                player_b_odds = winner_odds
                player_a_won = False
                rank_a, rank_b = loser_rank, winner_rank
                points_a, points_b = loser_points, winner_points

            records.append(
                MatchRecord(
                    date=match_date,
                    tour=tour,
                    surface=_normalize_surface(row.get("Surface", "")),
                    player_a=player_a,
                    player_b=player_b,
                    player_a_odds=player_a_odds,
                    player_b_odds=player_b_odds,
                    player_a_won=player_a_won,
                    tournament=row.get("Tournament", "").strip(),
                    series=(row.get("Series", "") or row.get("Tier", "")).strip(),
                    round_name=row.get("Round", "").strip(),
                    court=row.get("Court", "").strip(),
                    best_of=best_of,
                    rank_a=rank_a,
                    rank_b=rank_b,
                    points_a=points_a,
                    points_b=points_b,
                    comment=comment,
                    odds_source=matched_odds_source,
                    source_file=str(path),
                )
            )
    return sorted(records, key=lambda match: (match.date, match.tour, match.player_a, match.player_b))
