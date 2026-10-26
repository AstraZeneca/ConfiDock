"""Tests for confidock.rcsb module."""

from __future__ import annotations

import json
import tempfile
import urllib.error
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from confidock.rcsb import (
    RCSB_GRAPHQL_URL,
    fetch_deposit_dates,
    filter_inputs_by_date,
    pdb_id_from_stem,
)


class TestPdbIdFromStem:
    """Tests for pdb_id_from_stem."""

    def test_single_underscore(self):
        """Test PDB ID extraction with single underscore."""
        assert pdb_id_from_stem("7anq_B_A") == "7ANQ"

    def test_lowercase_input(self):
        """Test lowercase input is uppercased."""
        assert pdb_id_from_stem("8qf5_H_L") == "8QF5"

    def test_no_underscore(self):
        """Test PDB ID without underscores."""
        assert pdb_id_from_stem("7anq") == "7ANQ"

    def test_multiple_underscores(self):
        """Test PDB ID with multiple underscores."""
        assert pdb_id_from_stem("1abc_X_Y_Z") == "1ABC"


class TestGraphQLQuery:
    """Tests for internal GraphQL query construction."""

    @patch("confidock.rcsb.urllib.request.urlopen")
    def test_fetch_deposit_dates_constructs_query(self, mock_urlopen):
        """Verify GraphQL query structure."""
        # Mock response
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            {
                "data": {
                    "entries": [
                        {"rcsb_id": "7ANQ", "rcsb_accession_info": {"deposit_date": "2020-10-12T00:00:00.000+00:00"}}
                    ]
                }
            }
        ).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        fetch_deposit_dates(["7anq"])

        # Verify request was made
        assert mock_urlopen.called
        call_args = mock_urlopen.call_args
        request = call_args[0][0]

        # Check URL
        assert request.full_url == RCSB_GRAPHQL_URL

        # Check headers
        assert request.headers["Content-type"] == "application/json"
        assert request.headers["Accept"] == "application/json"

        # Check payload contains proper GraphQL query
        payload = json.loads(request.data.decode())
        assert "query" in payload
        assert "entries" in payload["query"]
        assert "7ANQ" in payload["query"]


class TestFetchDepositDates:
    """Tests for fetch_deposit_dates."""

    @patch("confidock.rcsb.urllib.request.urlopen")
    def test_single_entry(self, mock_urlopen):
        """Test fetching single entry."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            {
                "data": {
                    "entries": [
                        {"rcsb_id": "7ANQ", "rcsb_accession_info": {"deposit_date": "2020-10-12T00:00:00.000+00:00"}}
                    ]
                }
            }
        ).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = fetch_deposit_dates(["7anq"])

        assert "7ANQ" in result
        assert result["7ANQ"] == date(2020, 10, 12)

    @patch("confidock.rcsb.urllib.request.urlopen")
    def test_multiple_entries(self, mock_urlopen):
        """Test fetching multiple entries."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            {
                "data": {
                    "entries": [
                        {"rcsb_id": "7ANQ", "rcsb_accession_info": {"deposit_date": "2020-10-12T00:00:00.000+00:00"}},
                        {"rcsb_id": "8QF5", "rcsb_accession_info": {"deposit_date": "2023-07-15T00:00:00.000+00:00"}},
                    ]
                }
            }
        ).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = fetch_deposit_dates(["7anq", "8qf5"])

        assert len(result) == 2
        assert result["7ANQ"] == date(2020, 10, 12)
        assert result["8QF5"] == date(2023, 7, 15)

    @patch("confidock.rcsb.urllib.request.urlopen")
    def test_missing_deposit_date(self, mock_urlopen):
        """Test entry without deposit date."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            {"data": {"entries": [{"rcsb_id": "7ANQ", "rcsb_accession_info": {}}]}}
        ).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = fetch_deposit_dates(["7anq"])

        assert result["7ANQ"] is None

    @patch("confidock.rcsb.urllib.request.urlopen")
    def test_entry_not_in_response(self, mock_urlopen):
        """Test PDB ID not returned by API."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({"data": {"entries": []}}).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = fetch_deposit_dates(["FAKE"])

        assert result["FAKE"] is None

    @patch("confidock.rcsb.urllib.request.urlopen")
    @patch("confidock.rcsb.time.sleep")
    def test_retry_logic(self, mock_sleep, mock_urlopen):
        """Test retry on network failure."""
        # Fail twice, succeed on third attempt
        mock_urlopen.side_effect = [
            urllib.error.URLError("Network error"),
            urllib.error.URLError("Network error"),
            self._mock_successful_response(),
        ]

        result = fetch_deposit_dates(["7anq"])

        assert mock_urlopen.call_count == 3
        assert mock_sleep.call_count == 2
        assert "7ANQ" in result

    @patch("confidock.rcsb.urllib.request.urlopen")
    def test_retry_exhausted(self, mock_urlopen):
        """Test failure after exhausting retries."""
        mock_urlopen.side_effect = urllib.error.URLError("Network error")

        with pytest.raises(urllib.error.URLError):
            fetch_deposit_dates(["7anq"])

    @patch("confidock.rcsb.urllib.request.urlopen")
    def test_batching(self, mock_urlopen):
        """Test batching of large requests."""
        # Create 150 IDs (should result in 2 batches with _BATCH_SIZE=100)
        pdb_ids = [f"ID{i:03d}" for i in range(150)]

        mock_response = Mock()
        mock_response.read.return_value = json.dumps({"data": {"entries": []}}).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        fetch_deposit_dates(pdb_ids)

        # Should make 2 API calls
        assert mock_urlopen.call_count == 2

    @staticmethod
    def _mock_successful_response():
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            {
                "data": {
                    "entries": [
                        {"rcsb_id": "7ANQ", "rcsb_accession_info": {"deposit_date": "2020-10-12T00:00:00.000+00:00"}}
                    ]
                }
            }
        ).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        return mock_response


class TestFilterInputsByDate:
    """Tests for filter_inputs_by_date."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary input and output directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            yield input_dir, output_dir

    def _create_test_files(self, input_dir: Path, stems: list[str]):
        """Create test files with given stems."""
        for stem in stems:
            (input_dir / f"{stem}.yaml").write_text("test: data\n")

    @patch("confidock.rcsb.fetch_deposit_dates")
    def test_filter_after_cutoff(self, mock_fetch, temp_dirs):
        """Test filtering entries deposited after cutoff."""
        input_dir, output_dir = temp_dirs
        self._create_test_files(input_dir, ["7anq_A_B", "8qf5_H_L"])

        mock_fetch.return_value = {"7ANQ": date(2020, 10, 12), "8QF5": date(2023, 7, 15)}

        kept, excluded, _ = filter_inputs_by_date(input_dir, output_dir, date(2023, 6, 1), before=False)

        assert len(kept) == 1
        assert kept[0].stem == "8qf5_H_L"
        assert len(excluded) == 1
        assert excluded[0].stem == "7anq_A_B"

    @patch("confidock.rcsb.fetch_deposit_dates")
    def test_filter_before_cutoff(self, mock_fetch, temp_dirs):
        """Test filtering entries deposited on or before cutoff."""
        input_dir, output_dir = temp_dirs
        self._create_test_files(input_dir, ["7anq_A_B", "8qf5_H_L"])

        mock_fetch.return_value = {"7ANQ": date(2020, 10, 12), "8QF5": date(2023, 7, 15)}

        kept, excluded, _ = filter_inputs_by_date(input_dir, output_dir, date(2023, 6, 1), before=True)

        assert len(kept) == 1
        assert kept[0].stem == "7anq_A_B"
        assert len(excluded) == 1
        assert excluded[0].stem == "8qf5_H_L"

    @patch("confidock.rcsb.fetch_deposit_dates")
    def test_copy_files(self, mock_fetch, temp_dirs):
        """Test that kept files are actually copied."""
        input_dir, output_dir = temp_dirs
        self._create_test_files(input_dir, ["8qf5_H_L"])

        mock_fetch.return_value = {"8QF5": date(2023, 7, 15)}

        _, _, _ = filter_inputs_by_date(input_dir, output_dir, date(2023, 6, 1), before=False)

        assert (output_dir / "8qf5_H_L.yaml").exists()

    @patch("confidock.rcsb.fetch_deposit_dates")
    def test_dry_run(self, mock_fetch, temp_dirs):
        """Test dry run mode doesn't copy files."""
        input_dir, output_dir = temp_dirs
        self._create_test_files(input_dir, ["8qf5_H_L"])

        mock_fetch.return_value = {"8QF5": date(2023, 7, 15)}

        kept, _, _ = filter_inputs_by_date(input_dir, output_dir, date(2023, 6, 1), before=False, dry_run=True)

        assert len(kept) == 1
        assert not output_dir.exists()

    @patch("confidock.rcsb.fetch_deposit_dates")
    def test_multiple_extensions(self, mock_fetch, temp_dirs):
        """Test filtering multiple file extensions."""
        input_dir, output_dir = temp_dirs
        (input_dir / "7anq_A_B.yaml").write_text("test")
        (input_dir / "7anq_A_B.pdb").write_text("test")
        (input_dir / "7anq_A_B.json").write_text("test")

        mock_fetch.return_value = {"7ANQ": date(2023, 7, 15)}

        kept, _, _ = filter_inputs_by_date(input_dir, output_dir, date(2023, 6, 1), before=False)

        assert len(kept) == 3

    @patch("confidock.rcsb.fetch_deposit_dates")
    def test_unknown_deposit_date(self, mock_fetch, temp_dirs):
        """Test handling of entries with unknown deposit dates."""
        input_dir, output_dir = temp_dirs
        self._create_test_files(input_dir, ["7anq_A_B"])

        mock_fetch.return_value = {"7ANQ": None}

        kept, excluded, _ = filter_inputs_by_date(input_dir, output_dir, date(2023, 6, 1), before=False)

        # Entries with unknown dates should be excluded
        assert len(kept) == 0
        assert len(excluded) == 1

    def test_empty_input_directory(self, temp_dirs):
        """Test handling of empty input directory."""
        input_dir, output_dir = temp_dirs

        kept, excluded, dates = filter_inputs_by_date(input_dir, output_dir, date(2023, 6, 1), before=False)

        assert len(kept) == 0
        assert len(excluded) == 0
        assert len(dates) == 0

    @patch("confidock.rcsb.fetch_deposit_dates")
    def test_custom_extensions(self, mock_fetch, temp_dirs):
        """Test filtering with custom extensions."""
        input_dir, output_dir = temp_dirs
        (input_dir / "7anq_A_B.pdb").write_text("test")
        (input_dir / "7anq_A_B.yaml").write_text("test")

        mock_fetch.return_value = {"7ANQ": date(2023, 7, 15)}

        kept, _, _ = filter_inputs_by_date(input_dir, output_dir, date(2023, 6, 1), before=False, extensions=(".pdb",))

        # Only .pdb should be processed
        assert len(kept) == 1
        assert kept[0].suffix == ".pdb"

    @patch("confidock.rcsb.fetch_deposit_dates")
    def test_deduplication_of_pdb_ids(self, mock_fetch, temp_dirs):
        """Test that duplicate PDB IDs are only fetched once."""
        input_dir, output_dir = temp_dirs
        # Same PDB ID in multiple files
        self._create_test_files(input_dir, ["7anq_A_B", "7anq_C_D"])

        mock_fetch.return_value = {"7ANQ": date(2023, 7, 15)}

        filter_inputs_by_date(input_dir, output_dir, date(2023, 6, 1), before=False)

        # fetch_deposit_dates should be called with deduplicated IDs
        mock_fetch.assert_called_once()
        called_ids = mock_fetch.call_args[0][0]
        assert len(called_ids) == 1
        assert called_ids[0].upper() == "7ANQ"
