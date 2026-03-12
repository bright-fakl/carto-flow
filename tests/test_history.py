"""
Test suite for history management classes.

This module tests the functionality of the history management system including:
- BaseSnapshot abstract base class
- CartogramInternalsSnapshot for internal algorithm state
- CartogramSnapshot for algorithm statistics
- History class for managing snapshot collections
"""

import numpy as np
import pytest

from carto_flow.flow_cartogram.errors import MorphErrors
from carto_flow.flow_cartogram.history import (
    BaseSnapshot,
    CartogramInternalsSnapshot,
    CartogramSnapshot,
    ConvergenceHistory,
    ErrorRecord,
    History,
)


def make_mock_errors(mean_log: float = 0.1, max_log: float = 0.2) -> MorphErrors:
    """Create a mock MorphErrors object for testing."""
    log_errors = np.array([mean_log, max_log, mean_log * 0.5])
    return MorphErrors(
        log_errors=log_errors,
        mean_log_error=mean_log,
        max_log_error=max_log,
        errors_pct=np.sign(log_errors) * (2 ** np.abs(log_errors) - 1) * 100,
        mean_error_pct=(2**mean_log - 1) * 100,
        max_error_pct=(2**max_log - 1) * 100,
    )


class ConcreteSnapshot(BaseSnapshot):
    """Concrete implementation of BaseSnapshot for testing."""

    def __init__(self, iteration: int, **kwargs):
        super().__init__(iteration=iteration)
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestBaseSnapshot:
    """Test cases for BaseSnapshot abstract base class."""

    def test_has_variable_with_existing_attribute(self):
        """Test has_variable returns True for existing non-None attributes."""
        snapshot = ConcreteSnapshot(iteration=0, test_var="test_value", none_var=None, numeric_var=42)

        assert snapshot.has_variable("test_var") is True
        assert snapshot.has_variable("numeric_var") is True
        assert snapshot.has_variable("none_var") is False
        assert snapshot.has_variable("nonexistent") is False

    def test_get_variable_with_existing_attribute(self):
        """Test get_variable returns correct values for existing attributes."""
        snapshot = ConcreteSnapshot(iteration=0, test_var="test_value", numeric_var=42)

        assert snapshot.get_variable("test_var") == "test_value"
        assert snapshot.get_variable("numeric_var") == 42

    def test_get_variable_with_nonexistent_attribute(self):
        """Test get_variable raises AttributeError for non-existent attributes."""
        snapshot = ConcreteSnapshot(iteration=0)

        with pytest.raises(AttributeError, match="Variable 'nonexistent' not found"):
            snapshot.get_variable("nonexistent")

    def test_get_all_variables(self):
        """Test get_all_variables returns all non-None variables."""
        snapshot = ConcreteSnapshot(
            iteration=0, var1="value1", var2=42, var3=np.array([1, 2, 3]), none_var=None, _private_var="hidden"
        )

        variables = snapshot.get_all_variables()

        assert "var1" in variables
        assert "var2" in variables
        assert "var3" in variables
        assert "none_var" not in variables
        assert "_private_var" not in variables
        assert "iteration" not in variables

        assert variables["var1"] == "value1"
        assert variables["var2"] == 42
        assert np.array_equal(variables["var3"], np.array([1, 2, 3]))


class TestCartogramInternalsSnapshot:
    """Test cases for CartogramInternalsSnapshot."""

    def test_creation_with_minimal_data(self):
        """Test creating snapshot with only required iteration field."""
        snapshot = CartogramInternalsSnapshot(iteration=0)

        assert snapshot.iteration == 0
        assert snapshot.rho is None
        assert snapshot.vx is None
        assert snapshot.vy is None

    def test_creation_with_velocity_fields(self):
        """Test creating snapshot with velocity field data."""
        rho = np.random.rand(10, 10)
        vx = np.random.rand(10, 10)
        vy = np.random.rand(10, 10)

        snapshot = CartogramInternalsSnapshot(iteration=5, rho=rho, vx=vx, vy=vy)

        assert snapshot.iteration == 5
        assert np.array_equal(snapshot.rho, rho)
        assert np.array_equal(snapshot.vx, vx)
        assert np.array_equal(snapshot.vy, vy)

    def test_variable_access_through_base_class_methods(self):
        """Test that snapshot variables are accessible through BaseSnapshot methods."""
        snapshot = CartogramInternalsSnapshot(iteration=1, rho=np.ones((5, 5)), vx=np.zeros((5, 5)))

        assert snapshot.has_variable("rho") is True
        assert snapshot.has_variable("vx") is True
        assert snapshot.has_variable("vy") is False  # None value

        rho_value = snapshot.get_variable("rho")
        assert np.array_equal(rho_value, np.ones((5, 5)))


class TestCartogramSnapshot:
    """Test cases for CartogramSnapshot."""

    def test_creation_with_minimal_data(self):
        """Test creating snapshot with only required iteration field."""
        snapshot = CartogramSnapshot(iteration=0)

        assert snapshot.iteration == 0
        assert snapshot.geometry is None
        assert snapshot.errors is None
        assert snapshot.density is None

    def test_creation_with_error_data(self):
        """Test creating snapshot with error statistics via MorphErrors."""
        errors = make_mock_errors(mean_log=0.1167, max_log=0.2)

        snapshot = CartogramSnapshot(
            iteration=10,
            errors=errors,
        )

        assert snapshot.iteration == 10
        assert snapshot.errors is not None
        assert snapshot.errors.mean_log_error == 0.1167
        assert snapshot.errors.max_log_error == 0.2
        assert snapshot.geometry is None

    def test_creation_with_geometry(self):
        """Test creating snapshot with geometry data."""
        # Mock geometry object
        geometry = "mock_geodataframe"
        errors = make_mock_errors(mean_log=0.05)

        snapshot = CartogramSnapshot(
            iteration=5,
            geometry=geometry,
            errors=errors,
        )

        assert snapshot.iteration == 5
        assert snapshot.geometry == geometry
        assert snapshot.errors.mean_log_error == 0.05

    def test_variable_access(self):
        """Test variable access through BaseSnapshot interface."""
        errors = make_mock_errors(mean_log=0.1, max_log=0.15)
        snapshot = CartogramSnapshot(
            iteration=2,
            errors=errors,
        )

        assert snapshot.has_variable("errors") is True
        assert snapshot.has_variable("geometry") is False  # None value

        assert snapshot.get_variable("errors").mean_log_error == 0.1
        assert snapshot.get_variable("errors").max_log_error == 0.15


class TestHistory:
    """Test cases for History class."""

    def test_initialization(self):
        """Test creating empty history."""
        history = History()

        assert len(history) == 0
        assert history.snapshots == []
        assert history.get_iterations() == []

    def test_add_single_snapshot(self):
        """Test adding a single snapshot."""
        history = History()
        errors = make_mock_errors(mean_log=0.1)
        snapshot = CartogramSnapshot(iteration=0, errors=errors)

        history.add_snapshot(snapshot)

        assert len(history) == 1
        assert history.get_iterations() == [0]

    def test_add_multiple_snapshots(self):
        """Test adding multiple snapshots in order."""
        history = History()

        snapshots = [
            CartogramSnapshot(iteration=0, errors=make_mock_errors(mean_log=0.2)),
            CartogramSnapshot(iteration=1, errors=make_mock_errors(mean_log=0.15)),
            CartogramSnapshot(iteration=2, errors=make_mock_errors(mean_log=0.1)),
        ]

        for snapshot in snapshots:
            history.add_snapshot(snapshot)

        assert len(history) == 3
        assert history.get_iterations() == [0, 1, 2]

    def test_get_snapshot_existing(self):
        """Test retrieving existing snapshots."""
        history = History()
        snapshot0 = CartogramSnapshot(iteration=0, errors=make_mock_errors(mean_log=0.2))
        snapshot1 = CartogramSnapshot(iteration=1, errors=make_mock_errors(mean_log=0.15))

        history.add_snapshot(snapshot0)
        history.add_snapshot(snapshot1)

        retrieved = history.get_snapshot(0)
        assert retrieved is snapshot0
        assert retrieved.errors.mean_log_error == 0.2

        retrieved = history.get_snapshot(1)
        assert retrieved is snapshot1
        assert retrieved.errors.mean_log_error == 0.15

    def test_get_snapshot_nonexistent(self):
        """Test retrieving non-existent snapshots returns None."""
        history = History()
        history.add_snapshot(CartogramSnapshot(iteration=0, errors=make_mock_errors(mean_log=0.1)))

        assert history.get_snapshot(1) is None
        assert history.get_snapshot(-1) is None

    def test_get_variable_history(self):
        """Test retrieving variable history across snapshots."""
        history = History()

        # Add snapshots with errors
        history.add_snapshot(CartogramSnapshot(iteration=0, errors=make_mock_errors(mean_log=0.2)))
        history.add_snapshot(CartogramSnapshot(iteration=1))  # No errors
        history.add_snapshot(CartogramSnapshot(iteration=2, errors=make_mock_errors(mean_log=0.1)))

        errors_history = history.get_variable_history("errors")
        assert len(errors_history) == 3
        assert errors_history[0].mean_log_error == 0.2
        assert errors_history[1] is None
        assert errors_history[2].mean_log_error == 0.1

        # Test non-existent variable
        empty_history = history.get_variable_history("nonexistent")
        assert empty_history == [None, None, None]

    def test_get_variable_at_iteration(self):
        """Test getting specific variable at specific iteration."""
        history = History()
        errors = make_mock_errors(mean_log=0.05, max_log=0.1)
        snapshot = CartogramSnapshot(iteration=5, errors=errors)
        history.add_snapshot(snapshot)

        retrieved_errors = history.get_variable_at_iteration("errors", 5)
        assert retrieved_errors.mean_log_error == 0.05
        assert retrieved_errors.max_log_error == 0.1

    def test_get_variable_at_iteration_nonexistent_snapshot(self):
        """Test error when requesting variable from non-existent iteration."""
        history = History()
        history.add_snapshot(CartogramSnapshot(iteration=0, errors=make_mock_errors(mean_log=0.1)))

        with pytest.raises(ValueError, match="No snapshot found for iteration 1"):
            history.get_variable_at_iteration("errors", 1)

    def test_getitem_operator(self):
        """Test using [] operator to access snapshots by index."""
        history = History()
        snapshot0 = CartogramSnapshot(iteration=0, errors=make_mock_errors(mean_log=0.1))
        snapshot1 = CartogramSnapshot(iteration=10, errors=make_mock_errors(mean_log=0.08))
        history.add_snapshot(snapshot0)
        history.add_snapshot(snapshot1)

        # Access by index, not iteration
        assert history[0] is snapshot0
        assert history[1] is snapshot1
        assert history[-1] is snapshot1  # Negative indexing

    def test_getitem_slice(self):
        """Test using slice notation to access multiple snapshots."""
        history = History()
        snapshots = [
            CartogramSnapshot(iteration=0, errors=make_mock_errors(mean_log=0.3)),
            CartogramSnapshot(iteration=10, errors=make_mock_errors(mean_log=0.2)),
            CartogramSnapshot(iteration=20, errors=make_mock_errors(mean_log=0.1)),
        ]
        for s in snapshots:
            history.add_snapshot(s)

        result = history[0:2]
        assert len(result) == 2
        assert result[0] is snapshots[0]
        assert result[1] is snapshots[1]

    def test_getitem_out_of_range(self):
        """Test IndexError when accessing out-of-range index."""
        history = History()
        history.add_snapshot(CartogramSnapshot(iteration=0, errors=make_mock_errors(mean_log=0.1)))

        with pytest.raises(IndexError):
            _ = history[5]

    def test_iteration(self):
        """Test iterating over snapshots."""
        history = History()

        snapshots = [
            CartogramSnapshot(iteration=0, errors=make_mock_errors(mean_log=0.2)),
            CartogramSnapshot(iteration=1, errors=make_mock_errors(mean_log=0.15)),
            CartogramSnapshot(iteration=2, errors=make_mock_errors(mean_log=0.1)),
        ]

        for snapshot in snapshots:
            history.add_snapshot(snapshot)

        # Test iteration
        iterations = [snapshot.iteration for snapshot in history]
        assert iterations == [0, 1, 2]

        mean_errors = [snapshot.errors.mean_log_error for snapshot in history]
        assert mean_errors == [0.2, 0.15, 0.1]

    def test_latest_snapshot(self):
        """Test getting the latest snapshot."""
        history = History()

        # Empty history
        assert history.latest() is None

        # Single snapshot
        snapshot0 = CartogramSnapshot(iteration=0, errors=make_mock_errors(mean_log=0.2))
        history.add_snapshot(snapshot0)
        assert history.latest() is snapshot0

        # Multiple snapshots
        snapshot1 = CartogramSnapshot(iteration=1, errors=make_mock_errors(mean_log=0.15))
        snapshot2 = CartogramSnapshot(iteration=2, errors=make_mock_errors(mean_log=0.1))
        history.add_snapshot(snapshot1)
        history.add_snapshot(snapshot2)

        assert history.latest() is snapshot2
        assert history.latest().errors.mean_log_error == 0.1

    def test_variable_summary_scalars(self):
        """Test variable summary for scalar values via density."""
        history = History()

        # Add snapshots with density data (scalar arrays)
        for i in range(3):
            density = np.array([0.1 * (3 - i)])  # Decreasing: 0.3, 0.2, 0.1
            snapshot = CartogramSnapshot(
                iteration=i,
                density=density,
            )
            history.add_snapshot(snapshot)

        density_summary = history.variable_summary("density")
        assert density_summary["available"] is True
        assert density_summary["count"] == 3
        assert density_summary["shape_consistent"] is True

    def test_variable_summary_arrays(self):
        """Test variable summary for array values."""
        history = History()

        # Add snapshots with array data
        for i in range(2):
            snapshot = CartogramInternalsSnapshot(
                iteration=i,
                rho=np.ones((10, 10)) * (i + 1),  # Different values
            )
            history.add_snapshot(snapshot)

        rho_summary = history.variable_summary("rho")
        assert rho_summary["available"] is True
        assert rho_summary["count"] == 2
        assert rho_summary["shape_consistent"] is True
        assert rho_summary["shapes"] == (10, 10)
        assert rho_summary["dtype"] == "float64"

    def test_variable_summary_nonexistent(self):
        """Test variable summary for non-existent variable."""
        history = History()
        history.add_snapshot(CartogramSnapshot(iteration=0, errors=make_mock_errors(mean_log=0.1)))

        summary = history.variable_summary("nonexistent")
        assert summary["available"] is False
        assert summary["count"] == 0

    def test_variable_summary_with_none_values(self):
        """Test variable summary handles None values correctly."""
        history = History()

        # Mix of valid and None values using density (array type)
        history.add_snapshot(CartogramSnapshot(iteration=0, density=np.array([0.2])))
        history.add_snapshot(CartogramSnapshot(iteration=1))  # No density
        history.add_snapshot(CartogramSnapshot(iteration=2, density=np.array([0.1])))

        summary = history.variable_summary("density")
        assert summary["available"] is True
        assert summary["count"] == 2  # Only non-None values count

    def test_mixed_snapshot_types(self):
        """Test history works with different snapshot types."""
        history = History()

        # Add different types of snapshots
        internals = CartogramInternalsSnapshot(iteration=0, rho=np.ones((5, 5)), vx=np.zeros((5, 5)))
        stats = CartogramSnapshot(
            iteration=1,
            errors=make_mock_errors(mean_log=0.1, max_log=0.2),
        )

        history.add_snapshot(internals)
        history.add_snapshot(stats)

        assert len(history) == 2
        assert history.get_iterations() == [0, 1]

        # Test accessing variables from different snapshot types
        rho_history = history.get_variable_history("rho")
        assert len(rho_history) == 2
        assert np.array_equal(rho_history[0], np.ones((5, 5)))
        assert rho_history[1] is None

        errors_history = history.get_variable_history("errors")
        assert errors_history[0] is None
        assert errors_history[1].mean_log_error == 0.1


class TestIntegration:
    """Integration tests for the complete history system."""

    def test_full_workflow(self):
        """Test complete workflow of creating, adding, and querying snapshots."""
        history = History()

        # Create initial snapshot
        initial = CartogramSnapshot(
            iteration=0,
            geometry="initial_geometry",
            errors=make_mock_errors(mean_log=0.5, max_log=1.0),
        )
        history.add_snapshot(initial)

        # Create intermediate snapshots with decreasing errors
        test_cases = [(1, 0.5, 1.0), (2, 0.25, 0.5), (3, 1.0 / 6.0, 1.0 / 3.0)]

        for i, mean_log, max_log in test_cases:
            snapshot = CartogramSnapshot(
                iteration=i,
                errors=make_mock_errors(mean_log=mean_log, max_log=max_log),
            )
            history.add_snapshot(snapshot)

        # Verify history length and iterations
        assert len(history) == 4
        assert history.get_iterations() == [0, 1, 2, 3]

        # Test variable history
        errors_history = history.get_variable_history("errors")
        expected_means = [0.5, 0.5, 0.25, 1.0 / 6.0]
        assert len(errors_history) == 4
        for actual, expected in zip(errors_history, expected_means, strict=False):
            assert actual.mean_log_error == pytest.approx(expected, abs=1e-10)

        # Test latest snapshot
        latest = history.latest()
        assert latest.iteration == 3
        assert latest.errors.mean_log_error == pytest.approx(1.0 / 6.0, abs=1e-10)

        # Test variable access (variable_summary doesn't work on non-numeric types like MorphErrors)
        errors_history = history.get_variable_history("errors")
        assert len([e for e in errors_history if e is not None]) == 4

    def test_error_handling_integration(self):
        """Test error handling in realistic scenarios."""
        history = History()

        # Add some snapshots
        for i in range(3):
            snapshot = CartogramSnapshot(iteration=i, errors=make_mock_errors(mean_log=0.1 * i))
            history.add_snapshot(snapshot)

        # Test various error conditions
        with pytest.raises(ValueError, match="No snapshot found for iteration 5"):
            history.get_variable_at_iteration("errors", 5)

        with pytest.raises(IndexError):
            _ = history[5]

        with pytest.raises(AttributeError, match="Variable 'nonexistent' not found"):
            history.get_snapshot(0).get_variable("nonexistent")


class TestErrorRecord:
    """Test cases for ErrorRecord dataclass."""

    def test_creation(self):
        """Test ErrorRecord can be created with all fields."""
        record = ErrorRecord(
            iteration=5,
            mean_log_error=0.1,
            max_log_error=0.2,
            mean_error_pct=7.2,
            max_error_pct=14.9,
        )
        assert record.iteration == 5
        assert record.mean_log_error == 0.1
        assert record.max_log_error == 0.2
        assert record.mean_error_pct == 7.2
        assert record.max_error_pct == 14.9

    def test_repr(self):
        """Test ErrorRecord has informative repr."""
        record = ErrorRecord(
            iteration=1,
            mean_log_error=0.1,
            max_log_error=0.2,
            mean_error_pct=7.2,
            max_error_pct=14.9,
        )
        repr_str = repr(record)
        assert "ErrorRecord" in repr_str
        assert "iter=1" in repr_str
        assert "mean_log=" in repr_str


class TestConvergenceHistory:
    """Test cases for ConvergenceHistory class."""

    def test_initialization_empty(self):
        """Test ConvergenceHistory initializes as empty."""
        history = ConvergenceHistory()
        assert len(history) == 0
        assert len(history.iterations) == 0
        assert len(history.mean_log_errors) == 0

    def test_add_single_record(self):
        """Test adding a single error record."""
        history = ConvergenceHistory()
        errors = make_mock_errors(mean_log=0.1, max_log=0.2)
        history.add(1, errors)

        assert len(history) == 1
        assert history.iterations[0] == 1
        assert history.mean_log_errors[0] == pytest.approx(0.1)
        assert history.max_log_errors[0] == pytest.approx(0.2)

    def test_add_multiple_records(self):
        """Test adding multiple error records."""
        history = ConvergenceHistory()
        for i in range(100):
            errors = make_mock_errors(mean_log=0.1 * (100 - i) / 100)
            history.add(i + 1, errors)

        assert len(history) == 100
        assert history.iterations[0] == 1
        assert history.iterations[-1] == 100

    def test_getitem_positive_index(self):
        """Test accessing record by positive index."""
        history = ConvergenceHistory()
        errors = make_mock_errors(mean_log=0.15, max_log=0.25)
        history.add(5, errors)

        record = history[0]
        assert isinstance(record, ErrorRecord)
        assert record.iteration == 5
        assert record.mean_log_error == pytest.approx(0.15)
        assert record.max_log_error == pytest.approx(0.25)

    def test_getitem_negative_index(self):
        """Test accessing record by negative index."""
        history = ConvergenceHistory()
        history.add(1, make_mock_errors(mean_log=0.2))
        history.add(2, make_mock_errors(mean_log=0.1))

        record = history[-1]
        assert record.iteration == 2
        assert record.mean_log_error == pytest.approx(0.1)

    def test_iteration(self):
        """Test iterating over ConvergenceHistory."""
        history = ConvergenceHistory()
        for i in range(5):
            history.add(i + 1, make_mock_errors(mean_log=0.1))

        iterations = [r.iteration for r in history]
        assert iterations == [1, 2, 3, 4, 5]

    def test_get_by_iteration_found(self):
        """Test getting record by iteration number when found."""
        history = ConvergenceHistory()
        history.add(10, make_mock_errors(mean_log=0.15))
        history.add(20, make_mock_errors(mean_log=0.10))

        record = history.get_by_iteration(10)
        assert record is not None
        assert record.mean_log_error == pytest.approx(0.15)

    def test_get_by_iteration_not_found(self):
        """Test getting record by iteration number when not found."""
        history = ConvergenceHistory()
        history.add(10, make_mock_errors(mean_log=0.1))

        record = history.get_by_iteration(5)
        assert record is None

    def test_to_dict(self):
        """Test converting to dictionary of arrays."""
        history = ConvergenceHistory()
        history.add(1, make_mock_errors(mean_log=0.1))
        history.add(2, make_mock_errors(mean_log=0.05))

        d = history.to_dict()
        assert "iteration" in d
        assert "mean_log_error" in d
        assert "max_log_error" in d
        assert "mean_error_pct" in d
        assert "max_error_pct" in d
        assert len(d["iteration"]) == 2

    def test_repr_empty(self):
        """Test repr of empty ConvergenceHistory."""
        history = ConvergenceHistory()
        assert "empty" in repr(history)

    def test_repr_with_data(self):
        """Test repr of ConvergenceHistory with data."""
        history = ConvergenceHistory()
        history.add(1, make_mock_errors())
        history.add(100, make_mock_errors())

        r = repr(history)
        assert "ConvergenceHistory" in r
        assert "n=2" in r
        assert "1..100" in r
