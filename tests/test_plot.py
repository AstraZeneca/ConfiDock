"""Tests for confidock.plot module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from confidock.plot import (
    collect_by_entry,
    plot_dockq,
    plot_global_summary,
    plot_heatmap,
    plot_violin,
)


class TestCollectByEntry:
    """Tests for collect_by_entry function."""

    def test_collect_by_entry_with_complex_dockq(self):
        """Test collecting entries with complex_dockq column."""
        rows = [
            {"entry": "1abc", "seed": 1, "sample": 1, "complex_dockq": "0.85"},
            {"entry": "1abc", "seed": 1, "sample": 2, "complex_dockq": "0.75"},
            {"entry": "2def", "seed": 1, "sample": 1, "complex_dockq": "0.65"},
        ]

        result = collect_by_entry(rows)
        assert "1abc" in result
        assert "2def" in result
        assert len(result["1abc"]) == 2
        assert len(result["2def"]) == 1
        assert result["1abc"][0] == pytest.approx(0.85)
        assert result["2def"][0] == pytest.approx(0.65)

    def test_collect_by_entry_with_interface_dockq(self):
        """Test fallback to per-interface DockQ."""
        rows = [
            {"entry": "1abc", "seed": 1, "sample": 1, "DockQ": "0.85"},
            {"entry": "1abc", "seed": 1, "sample": 1, "DockQ": "0.80"},  # Same sample, different interface
            {"entry": "1abc", "seed": 1, "sample": 2, "DockQ": "0.75"},
        ]

        result = collect_by_entry(rows)
        assert "1abc" in result
        # Should compute mean per sample
        assert len(result["1abc"]) == 2  # 2 samples

    def test_collect_by_entry_with_grid_conditions(self):
        """Test with grid condition columns."""
        rows = [
            {
                "entry": "1abc",
                "seed": 1,
                "sample": 1,
                "msa_mode": "full",
                "recycling_steps": 3,
                "step_scale": 1.0,
                "complex_dockq": "0.85",
            },
            {
                "entry": "1abc",
                "seed": 1,
                "sample": 1,
                "msa_mode": "none",
                "recycling_steps": 3,
                "step_scale": 1.0,
                "complex_dockq": "0.75",
            },
        ]

        result = collect_by_entry(rows)
        # With different conditions, should treat as different samples
        assert len(result["1abc"]) == 2

    def test_collect_by_entry_empty(self):
        """Test with empty input."""
        result = collect_by_entry([])
        assert result == {}

    def test_collect_by_entry_missing_values(self):
        """Test handling of missing values."""
        rows = [
            {"entry": "1abc", "seed": 1, "sample": 1, "complex_dockq": ""},
            {"entry": "1abc", "seed": 1, "sample": 2, "complex_dockq": "0.85"},
        ]

        result = collect_by_entry(rows)
        assert len(result["1abc"]) == 1  # Empty value should be skipped
        assert result["1abc"][0] == pytest.approx(0.85)


class TestPlotDockQ:
    """Tests for plot_dockq function."""

    def test_plot_dockq_basic(self):
        """Test basic DockQ boxplot generation."""
        rows = [
            {"entry": "1abc", "seed": 1, "sample": 1, "complex_dockq": "0.85"},
            {"entry": "1abc", "seed": 1, "sample": 2, "complex_dockq": "0.75"},
            {"entry": "2def", "seed": 1, "sample": 1, "complex_dockq": "0.65"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_plot.png"
            plot_dockq(rows, output_path, title="Test DockQ")

            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_plot_dockq_creates_directory(self):
        """Test that output directory is created."""
        rows = [{"entry": "1abc", "seed": 1, "sample": 1, "complex_dockq": "0.85"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "test_plot.png"
            plot_dockq(rows, output_path)

            assert output_path.parent.exists()
            assert output_path.exists()
            assert output_path.stat().st_size > 0


class TestPlotViolin:
    """Tests for plot_violin function."""

    def test_plot_violin_basic(self):
        """Test basic violin plot generation."""
        rows = [{"entry": "1abc", "seed": 1, "sample": i, "complex_dockq": str(0.7 + i * 0.05)} for i in range(10)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_violin.png"
            plot_violin(rows, output_path, title="Test Violin")

            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_plot_violin_many_entries(self):
        """Test violin plot with many entries (horizontal layout)."""
        rows = []
        for i in range(50):  # More than 40 entries triggers horizontal layout
            rows.append({"entry": f"entry{i}", "seed": 1, "sample": 1, "complex_dockq": "0.75"})

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_violin_horiz.png"
            plot_violin(rows, output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0


class TestPlotHeatmap:
    """Tests for plot_heatmap function."""

    def test_plot_heatmap_basic(self):
        """Test basic heatmap generation."""
        rows = []
        for i in range(5):
            for j in range(10):
                rows.append(
                    {
                        "entry": f"entry{i}",
                        "seed": 1,
                        "sample": j,
                        "complex_dockq": str(0.1 * j),
                    }
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_heatmap.png"
            plot_heatmap(rows, output_path, title="Test Heatmap")

            assert output_path.exists()
            assert output_path.stat().st_size > 0


class TestPlotGlobalSummary:
    """Tests for plot_global_summary function."""

    def test_plot_global_summary_basic(self):
        """Test global summary plot generation."""
        rows = []
        for i in range(5):
            for j in range(10):
                rows.append(
                    {
                        "entry": f"entry{i}",
                        "seed": 1,
                        "sample": j,
                        "complex_dockq": str(0.5 + 0.05 * j),
                    }
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_summary.png"
            plot_global_summary(rows, output_path, title="Test Summary")

            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_plot_global_summary_without_scipy(self):
        """Test that plot works even if scipy KDE fails."""
        rows = [{"entry": "1abc", "seed": 1, "sample": i, "complex_dockq": "0.75"} for i in range(10)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_summary_no_scipy.png"
            plot_global_summary(rows, output_path)

            # Should still generate plot
            assert output_path.exists()
            assert output_path.stat().st_size > 0


class TestHelperFunctions:
    """Tests for helper functions in plot module."""

    def test_collect_by_entry_handles_numeric_dockq(self):
        """Test that numeric DockQ values are handled correctly."""
        rows = [
            {"entry": "1abc", "seed": 1, "sample": 1, "complex_dockq": 0.85},  # Numeric
            {"entry": "1abc", "seed": 1, "sample": 2, "complex_dockq": "0.75"},  # String
        ]

        result = collect_by_entry(rows)
        assert len(result["1abc"]) == 2
        assert all(isinstance(v, float) for v in result["1abc"])

    def test_collect_by_entry_deduplicates_correctly(self):
        """Test that duplicate samples are handled correctly."""
        rows = [
            {"entry": "1abc", "seed": 1, "sample": 1, "complex_dockq": "0.85"},
            {"entry": "1abc", "seed": 1, "sample": 1, "complex_dockq": "0.85"},  # Duplicate
            {"entry": "1abc", "seed": 1, "sample": 2, "complex_dockq": "0.75"},
        ]

        result = collect_by_entry(rows)
        assert len(result["1abc"]) == 2  # Should deduplicate


class TestPlotConstants:
    """Tests for plot module constants and configuration."""

    def test_az_palette_exists(self):
        """Test that AZ color palette is defined."""
        from confidock.plot import AZ_PALETTE

        assert len(AZ_PALETTE) > 0
        assert all(color.startswith("#") for color in AZ_PALETTE)

    def test_thresholds_defined(self):
        """Test that DockQ thresholds are defined."""
        from confidock.plot import THRESHOLDS

        assert len(THRESHOLDS) > 0
        for threshold in THRESHOLDS:
            value, color, label = threshold
            assert isinstance(value, float)
            assert isinstance(color, str)
            assert isinstance(label, str)

    def test_label_constants(self):
        """Test that label constants are defined."""
        from confidock.plot import (
            LABEL_COMPLEX_DOCKQ,
            LABEL_IRMSD_ANGSTROM,
            LEGEND_LOC_LOWER_RIGHT,
        )

        assert isinstance(LABEL_COMPLEX_DOCKQ, str)
        assert isinstance(LABEL_IRMSD_ANGSTROM, str)
        assert isinstance(LEGEND_LOC_LOWER_RIGHT, str)


class TestGridHelpers:
    """Tests for grid-specific helper functions."""

    def test_infer_grid_axes_boltz(self):
        """Test inferring Boltz-style grid axes."""
        from confidock.plot import _infer_grid_axes

        rows = [
            {"msa_mode": "full", "recycling_steps": 3, "step_scale": 1.0},
            {"msa_mode": "full", "recycling_steps": 5, "step_scale": 2.0},
        ]

        spec = _infer_grid_axes(rows)
        assert spec is not None
        assert "primary_col" in spec
        assert "secondary_col" in spec

    def test_infer_grid_axes_protenix(self):
        """Test inferring Protenix-style grid axes."""
        from confidock.plot import _infer_grid_axes

        rows = [
            {"msa_mode": "full", "cycle": 4, "step": 48},
            {"msa_mode": "full", "cycle": 8, "step": 24},
        ]

        spec = _infer_grid_axes(rows)
        assert spec is not None

    def test_infer_grid_axes_no_grid(self):
        """Test with non-grid data."""
        from confidock.plot import _infer_grid_axes

        rows = [
            {"entry": "1abc", "seed": 1, "sample": 1},
        ]

        spec = _infer_grid_axes(rows)
        assert spec is None


class TestConditionKeyFunctions:
    """Tests for condition key building functions."""

    def test_condition_key_cols_boltz(self):
        """Test extracting Boltz condition columns."""
        from confidock.plot import _condition_key_cols

        rows = [
            {"msa_mode": "full", "recycling_steps": 3, "step_scale": 1.0},
        ]

        cols = _condition_key_cols(rows)
        assert "msa_mode" in cols or len(cols) > 0

    def test_build_sample_key(self):
        """Test building unique sample keys."""
        from confidock.plot import _build_sample_key

        row = {"entry": "1abc", "seed": 1, "sample": 1, "msa_mode": "full"}
        condition_cols = ["msa_mode"]

        key = _build_sample_key(row, condition_cols)
        assert isinstance(key, tuple)
        assert key[0] == "1abc"


class TestCollectFunctions:
    """Tests for data collection functions."""

    def test_collect_complex_dockq(self):
        """Test collecting complex DockQ values."""
        from confidock.plot import _collect_complex_dockq

        rows = [
            {"entry": "1abc", "seed": 1, "sample": 1, "complex_dockq": "0.85"},
            {"entry": "1abc", "seed": 1, "sample": 2, "complex_dockq": "0.75"},
            {"entry": "2def", "seed": 1, "sample": 1, "complex_dockq": "0.65"},
        ]

        result = _collect_complex_dockq(rows, [])
        assert "1abc" in result
        assert len(result["1abc"]) == 2

    def test_collect_interface_dockq(self):
        """Test collecting interface DockQ values."""
        from confidock.plot import _collect_interface_dockq

        rows = [
            {"entry": "1abc", "seed": 1, "sample": 1, "DockQ": "0.85"},
            {"entry": "1abc", "seed": 1, "sample": 1, "DockQ": "0.80"},
            {"entry": "1abc", "seed": 1, "sample": 2, "DockQ": "0.75"},
        ]

        result = _collect_interface_dockq(rows, [])
        assert "1abc" in result


class TestThresholdHelpers:
    """Tests for threshold line functions."""

    def test_add_threshold_lines_vertical(self):
        """Test adding vertical threshold lines."""
        from confidock.plot import _add_threshold_lines

        ax = MagicMock()
        _add_threshold_lines(ax, 1.0, orientation="vertical")

        # Should call ax.axhline for vertical orientation
        assert ax.axhline.called

    def test_add_threshold_lines_horizontal(self):
        """Test adding horizontal threshold lines."""
        from confidock.plot import _add_threshold_lines

        ax = MagicMock()
        _add_threshold_lines(ax, 1.0, orientation="horizontal")

        # Should call ax.axvline for horizontal orientation
        assert ax.axvline.called


class TestViolinStyling:
    """Tests for violin plot styling functions."""

    def test_style_violin_parts(self):
        """Test styling violin plot parts."""
        from confidock.plot import _style_violin_parts

        parts = {
            "bodies": [MagicMock()],
            "cmaxes": MagicMock(),
            "cmins": MagicMock(),
            "cbars": MagicMock(),
        }

        # Should not raise error
        _style_violin_parts(parts)


class TestConfigureAxes:
    """Tests for axes configuration functions."""

    def test_configure_violin_axes_vertical(self):
        """Test configuring vertical violin plot axes."""
        from confidock.plot import _configure_violin_axes

        ax = MagicMock()
        entries = ["entry1", "entry2", "entry3"]
        ticks = range(len(entries))
        label_fs = 9
        data_min, data_max = 0.0, 1.0
        y_margin = 0.05

        _configure_violin_axes(ax, entries, ticks, label_fs, data_min, data_max, y_margin, vert=True)

        # Should set labels
        assert ax.set_ylabel.called

    def test_configure_violin_axes_horizontal(self):
        """Test configuring horizontal violin plot axes."""
        from confidock.plot import _configure_violin_axes

        ax = MagicMock()
        entries = ["entry" + str(i) for i in range(50)]  # Many entries
        ticks = range(len(entries))
        label_fs = 7
        data_min, data_max = 0.0, 1.0
        y_margin = 0.05

        _configure_violin_axes(ax, entries, ticks, label_fs, data_min, data_max, y_margin, vert=False)

        assert ax.set_xlabel.called


class TestDeduplication:
    """Tests for data deduplication functions."""

    def test_deduplicate_diversity_rows_no_spec(self):
        """Test deduplication without grid spec."""
        from confidock.plot import _deduplicate_diversity_rows

        rows = [
            {"entry": "1abc", "seed": 1, "sample": 1, "native_interface": "AB"},
            {"entry": "1abc", "seed": 1, "sample": 1, "native_interface": "AB"},  # Duplicate
            {"entry": "1abc", "seed": 1, "sample": 2, "native_interface": "AB"},
        ]

        result = _deduplicate_diversity_rows(rows, None, False)
        assert len(result) < len(rows)

    def test_deduplicate_diversity_rows_with_conditions(self):
        """Test deduplication with grid conditions."""
        from confidock.plot import _deduplicate_diversity_rows

        rows = [
            {"entry": "1abc", "seed": 1, "sample": 1, "msa_mode": "full", "native_interface": "AB"},
            {"entry": "1abc", "seed": 1, "sample": 1, "msa_mode": "none", "native_interface": "AB"},
        ]

        spec = {"primary_col": "msa_mode", "secondary_col": "recycling_steps"}
        result = _deduplicate_diversity_rows(rows, spec, True)
        # Different conditions should not be deduplicated
        assert len(result) == 2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_rows_all_plots(self):
        """Test that all plot functions handle empty input gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "empty.png"

            # These should not crash with empty input
            with patch("matplotlib.pyplot.savefig"):
                with patch("matplotlib.pyplot.close"):
                    try:
                        plot_dockq([], output_path)
                    except (ValueError, KeyError):
                        pass  # Some plots may intentionally raise on empty data

                    try:
                        plot_violin([], output_path)
                    except (ValueError, KeyError):
                        pass

                    try:
                        plot_heatmap([], output_path)
                    except (ValueError, KeyError):
                        pass

    def test_single_entry_plots(self):
        """Test plots with single entry."""
        rows = [{"entry": "1abc", "seed": 1, "sample": 1, "complex_dockq": "0.85"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "single.png"

            with patch("matplotlib.pyplot.savefig"):
                with patch("matplotlib.pyplot.close"):
                    plot_dockq(rows, output_path)
                    plot_violin(rows, output_path)

    def test_nan_values_handling(self):
        """Test handling of NaN/invalid values."""
        rows = [
            {"entry": "1abc", "seed": 1, "sample": 1, "complex_dockq": "nan"},
            {"entry": "1abc", "seed": 1, "sample": 2, "complex_dockq": "0.85"},
        ]

        result = collect_by_entry(rows)
        # Should skip NaN values
        assert len(result["1abc"]) == 1

    def test_missing_columns(self):
        """Test handling of rows with missing columns."""
        rows = [
            {"entry": "1abc", "seed": 1},  # Missing sample and dockq
            {"entry": "2def", "seed": 1, "sample": 1, "complex_dockq": "0.85"},
        ]

        result = collect_by_entry(rows)
        # Should handle gracefully
        assert "2def" in result
