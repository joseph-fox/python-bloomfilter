"""Comprehensive test suite for pybloom_live Bloom Filter implementations.

This module tests both BloomFilter and ScalableBloomFilter classes,
including edge cases, error conditions, and all public APIs.

Test Coverage:
- Hash function selection and generation
- Basic operations (add, contains, len)
- Set operations (union, intersection)
- Serialization/deserialization
- Error handling and validation
- Edge cases and boundary conditions
- ScalableBloomFilter scaling behavior
"""
import io
import pickle
import random
import tempfile

import pytest

from pybloom_live.pybloom import BloomFilter, ScalableBloomFilter, make_hashfuncs


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def basic_filter():
    """Create a basic BloomFilter for testing.

    Returns:
        BloomFilter: A filter with capacity=100, error_rate=0.01
    """
    return BloomFilter(capacity=100, error_rate=0.01)


@pytest.fixture
def populated_filter():
    """Create a BloomFilter with some data for testing.

    Returns:
        BloomFilter: A filter containing lowercase letters a-z
    """
    bf = BloomFilter(capacity=100, error_rate=0.01)
    for char in 'abcdefghijklmnopqrstuvwxyz':
        bf.add(char)
    return bf


@pytest.fixture
def scalable_filter():
    """Create a ScalableBloomFilter for testing.

    Returns:
        ScalableBloomFilter: A filter with default parameters
    """
    return ScalableBloomFilter(initial_capacity=100, error_rate=0.001)


# =============================================================================
# Hash Function Tests
# =============================================================================

class TestHashFunctions:
    """Test hash function selection and generation."""

    def test_xxhash_selection(self):
        """Test that xxHash is selected for small bit counts.

        Purpose:
            Verify that for ≤128 bits, the fastest non-cryptographic hash
            (xxh3_128) is selected.

        Expected:
            Hash function name should be 'xxh3_128'
        """
        make_hashes, hashfn = make_hashfuncs(num_slices=5, num_bits=1)
        assert hashfn.__name__ == 'xxh3_128', \
            "Should use xxHash for small bit counts (fastest option)"

    def test_sha1_selection(self):
        """Test that SHA-1 is selected for 129-160 bit requirements.

        Purpose:
            Verify correct hash function selection based on total bits needed.

        Expected:
            Hash function name should be 'openssl_sha1'
        """
        make_hashes, hashfn = make_hashfuncs(num_slices=10, num_bits=2)
        assert hashfn.__name__ == 'openssl_sha1'

    def test_sha256_selection(self):
        """Test that SHA-256 is selected for 161-256 bit requirements.

        Purpose:
            Verify correct hash function selection for medium bit counts.

        Expected:
            Hash function name should be 'openssl_sha256'
        """
        make_hashes, hashfn = make_hashfuncs(num_slices=15, num_bits=2)
        assert hashfn.__name__ == 'openssl_sha256'

    def test_sha384_selection(self):
        """Test that SHA-384 is selected for 257-384 bit requirements.

        Purpose:
            Verify correct hash function selection for larger bit counts.

        Expected:
            Hash function name should be 'openssl_sha384'
        """
        make_hashes, hashfn = make_hashfuncs(num_slices=20, num_bits=3)
        assert hashfn.__name__ == 'openssl_sha384'

    def test_sha512_selection(self):
        """Test that SHA-512 is selected for >384 bit requirements.

        Purpose:
            Verify correct hash function selection for very large bit counts.

        Expected:
            Hash function name should be 'openssl_sha512'
        """
        make_hashes, hashfn = make_hashfuncs(num_slices=100, num_bits=20)
        assert hashfn.__name__ == 'openssl_sha512'

    def test_hash_maker_generates_correct_count(self):
        """Test that hash maker generates the requested number of hashes.

        Purpose:
            Verify that the hash generator yields exactly num_slices values.

        Expected:
            Generator should produce exactly num_slices hash values.
        """
        num_slices = 10
        make_hashes, _ = make_hashfuncs(num_slices=num_slices, num_bits=100)
        hashes = list(make_hashes("test_key"))
        assert len(hashes) == num_slices, \
            f"Should generate exactly {num_slices} hashes"

    def test_hash_maker_produces_valid_indices(self):
        """Test that hash values are valid bit array indices.

        Purpose:
            Verify all hash values are within the valid range [0, num_bits).

        Expected:
            All hash values should be >= 0 and < num_bits.
        """
        num_bits = 1000
        make_hashes, _ = make_hashfuncs(num_slices=10, num_bits=num_bits)
        hashes = list(make_hashes("test_key"))

        for h in hashes:
            assert 0 <= h < num_bits, \
                f"Hash value {h} is out of range [0, {num_bits})"


# =============================================================================
# BloomFilter Basic Operations Tests
# =============================================================================

class TestBloomFilterBasics:
    """Test basic BloomFilter operations."""

    def test_initialization_default_error_rate(self):
        """Test BloomFilter initialization with default error rate.

        Purpose:
            Verify that BloomFilter can be created with just capacity,
            using default error_rate of 0.001.

        Expected:
            Filter should be created successfully with error_rate=0.001.
        """
        bf = BloomFilter(capacity=1000)
        assert bf.capacity == 1000
        assert bf.error_rate == 0.001
        assert len(bf) == 0

    def test_initialization_custom_error_rate(self):
        """Test BloomFilter initialization with custom error rate.

        Purpose:
            Verify that custom error rates are properly stored and used.

        Expected:
            Filter should store the provided error_rate value.
        """
        bf = BloomFilter(capacity=500, error_rate=0.05)
        assert bf.capacity == 500
        assert bf.error_rate == 0.05

    def test_initialization_invalid_error_rate_zero(self):
        """Test that zero error rate raises ValueError.

        Purpose:
            Validate input: error rate must be between 0 and 1 (exclusive).

        Expected:
            ValueError should be raised for error_rate=0.
        """
        with pytest.raises(ValueError, match="Error_Rate must be between 0 and 1"):
            BloomFilter(capacity=100, error_rate=0)

    def test_initialization_invalid_error_rate_one(self):
        """Test that error rate of 1.0 raises ValueError.

        Purpose:
            Validate input: error rate must be less than 1.

        Expected:
            ValueError should be raised for error_rate=1.0.
        """
        with pytest.raises(ValueError, match="Error_Rate must be between 0 and 1"):
            BloomFilter(capacity=100, error_rate=1.0)

    def test_initialization_invalid_error_rate_negative(self):
        """Test that negative error rate raises ValueError.

        Purpose:
            Validate input: error rate must be positive.

        Expected:
            ValueError should be raised for negative error_rate.
        """
        with pytest.raises(ValueError, match="Error_Rate must be between 0 and 1"):
            BloomFilter(capacity=100, error_rate=-0.1)

    def test_initialization_invalid_capacity_zero(self):
        """Test that zero capacity raises ValueError.

        Purpose:
            Validate input: capacity must be positive.

        Expected:
            ValueError should be raised for capacity=0.
        """
        with pytest.raises(ValueError, match="Capacity must be > 0"):
            BloomFilter(capacity=0, error_rate=0.01)

    def test_initialization_invalid_capacity_negative(self):
        """Test that negative capacity raises ValueError.

        Purpose:
            Validate input: capacity must be positive.

        Expected:
            ValueError should be raised for negative capacity.
        """
        with pytest.raises(ValueError, match="Capacity must be > 0"):
            BloomFilter(capacity=-100, error_rate=0.01)

    def test_add_returns_false_for_new_element(self, basic_filter):
        """Test that add() returns False for new elements.

        Purpose:
            Verify that add() correctly indicates when an element is new.

        Expected:
            add() should return False when adding a new element.
        """
        result = basic_filter.add("new_element")
        assert result is False, "Should return False for new elements"

    def test_add_returns_true_for_duplicate(self, basic_filter):
        """Test that add() returns True for duplicate elements.

        Purpose:
            Verify that add() detects likely duplicates.

        Expected:
            add() should return True when adding an element that's already present.
        """
        basic_filter.add("test")
        result = basic_filter.add("test")
        assert result is True, "Should return True for duplicate elements"

    def test_add_increments_count(self, basic_filter):
        """Test that add() increments the element count.

        Purpose:
            Verify that the count property tracks additions correctly.

        Expected:
            len() should increase by 1 for each new element added.
        """
        initial_count = len(basic_filter)
        basic_filter.add("element1")
        assert len(basic_filter) == initial_count + 1
        basic_filter.add("element2")
        assert len(basic_filter) == initial_count + 2

    def test_add_duplicate_does_not_increment_count(self, basic_filter):
        """Test that adding duplicates doesn't increment count.

        Purpose:
            Verify that count represents unique additions (detected as new).

        Expected:
            Count should not increase when adding detected duplicates.
        """
        basic_filter.add("test")
        count_after_first = len(basic_filter)
        basic_filter.add("test")  # Duplicate
        assert len(basic_filter) == count_after_first, \
            "Count should not increase for detected duplicates"

    def test_add_with_skip_check(self, basic_filter):
        """Test add() with skip_check=True for bulk insertions.

        Purpose:
            Verify that skip_check parameter works and improves performance
            by skipping duplicate detection.

        Expected:
            add() should always return False with skip_check=True,
            and count should always increment.
        """
        basic_filter.add("test")
        result = basic_filter.add("test", skip_check=True)
        assert result is False, "skip_check=True should always return False"

    def test_contains_returns_true_for_added_elements(self, populated_filter):
        """Test that __contains__ returns True for added elements.

        Purpose:
            Verify membership test works correctly for elements that were added.

        Expected:
            'in' operator should return True for all added elements.
        """
        for char in 'abcdefghijklmnopqrstuvwxyz':
            assert char in populated_filter, \
                f"'{char}' should be in the filter"

    def test_contains_returns_false_for_absent_elements(self, populated_filter):
        """Test that __contains__ returns False for elements not added.

        Purpose:
            Verify membership test correctly identifies absent elements.

        Expected:
            'in' operator should return False for elements never added.
        """
        assert 'A' not in populated_filter, "Uppercase 'A' should not be in filter"
        assert '1' not in populated_filter, "Digit '1' should not be in filter"
        assert '@' not in populated_filter, "Symbol '@' should not be in filter"

    def test_len_returns_zero_for_empty_filter(self, basic_filter):
        """Test that len() returns 0 for empty filter.

        Purpose:
            Verify length starts at zero for new filters.

        Expected:
            len() should return 0 for newly created filter.
        """
        assert len(basic_filter) == 0

    def test_len_returns_correct_count(self):
        """Test that len() returns accurate count of additions.

        Purpose:
            Verify length tracking is accurate across multiple operations.

        Expected:
            len() should match the number of new (non-duplicate) additions.
        """
        bf = BloomFilter(capacity=100, error_rate=0.01)
        expected_count = 0

        # Add 10 unique elements
        for i in range(10):
            bf.add(f"element_{i}")
            expected_count += 1
            assert len(bf) == expected_count

    def test_capacity_exceeded_raises_error(self):
        """Test that exceeding capacity raises IndexError.

        Purpose:
            Verify that the filter prevents additions beyond capacity
            to maintain error rate guarantees.

        Expected:
            IndexError should be raised when count reaches capacity.
        """
        bf = BloomFilter(capacity=5, error_rate=0.01)

        # Add up to capacity
        for i in range(5):
            bf.add(f"item_{i}")

        # Next addition should raise error
        with pytest.raises(IndexError, match="BloomFilter is at capacity"):
            bf.add("overflow_item")

    def test_add_different_data_types(self, basic_filter):
        """Test add() with various data types.

        Purpose:
            Verify that the filter handles strings, bytes, and integers.

        Expected:
            All types should be successfully added and retrieved.
        """
        # String
        basic_filter.add("string_test")
        assert "string_test" in basic_filter

        # Bytes
        basic_filter.add(b"bytes_test")
        assert b"bytes_test" in basic_filter

        # Integer (converted to string)
        basic_filter.add(42)
        assert 42 in basic_filter


# =============================================================================
# BloomFilter Set Operations Tests
# =============================================================================

class TestBloomFilterSetOperations:
    """Test union and intersection operations on Bloom filters."""

    def test_union_combines_elements(self):
        """Test that union contains elements from both filters.

        Purpose:
            Verify union (OR) operation creates a filter containing
            all elements from both input filters.

        Expected:
            Resulting filter should contain elements from both filters.
        """
        bf1 = BloomFilter(capacity=100, error_rate=0.01)
        bf2 = BloomFilter(capacity=100, error_rate=0.01)

        # bf1 gets lowercase letters a-m
        for char in 'abcdefghijklm':
            bf1.add(char)

        # bf2 gets lowercase letters n-z
        for char in 'nopqrstuvwxyz':
            bf2.add(char)

        # Union should have all letters
        bf_union = bf1.union(bf2)

        for char in 'abcdefghijklmnopqrstuvwxyz':
            assert char in bf_union, f"Union should contain '{char}'"

    def test_union_operator_syntax(self):
        """Test that | operator works for union.

        Purpose:
            Verify that the bitwise OR operator is correctly implemented.

        Expected:
            bf1 | bf2 should produce same result as bf1.union(bf2).
        """
        bf1 = BloomFilter(capacity=50, error_rate=0.01)
        bf2 = BloomFilter(capacity=50, error_rate=0.01)

        bf1.add("apple")
        bf2.add("banana")

        bf_union = bf1 | bf2

        assert "apple" in bf_union
        assert "banana" in bf_union

    def test_union_requires_matching_capacity(self):
        """Test that union fails with mismatched capacities.

        Purpose:
            Validate that union only works on compatible filters.

        Expected:
            ValueError should be raised for different capacities.
        """
        bf1 = BloomFilter(capacity=100, error_rate=0.01)
        bf2 = BloomFilter(capacity=200, error_rate=0.01)

        with pytest.raises(ValueError, match="same capacity and error rate"):
            bf1.union(bf2)

    def test_union_requires_matching_error_rate(self):
        """Test that union fails with mismatched error rates.

        Purpose:
            Validate that union only works on compatible filters.

        Expected:
            ValueError should be raised for different error rates.
        """
        bf1 = BloomFilter(capacity=100, error_rate=0.01)
        bf2 = BloomFilter(capacity=100, error_rate=0.001)

        with pytest.raises(ValueError, match="same capacity and error rate"):
            bf1.union(bf2)

    def test_intersection_contains_common_elements(self):
        """Test that intersection contains only common elements.

        Purpose:
            Verify intersection (AND) operation creates a filter containing
            only elements present in both input filters.

        Expected:
            Resulting filter should contain only overlapping elements.
        """
        bf1 = BloomFilter(capacity=100, error_rate=0.01)
        bf2 = BloomFilter(capacity=100, error_rate=0.01)

        # bf1 gets a-p
        for char in 'abcdefghijklmnop':
            bf1.add(char)

        # bf2 gets j-z (overlap is j-p)
        for char in 'jklmnopqrstuvwxyz':
            bf2.add(char)

        bf_intersect = bf1.intersection(bf2)

        # Should contain overlap (j-p)
        for char in 'jklmnop':
            assert char in bf_intersect, \
                f"Intersection should contain '{char}' (in both filters)"

        # Should NOT contain elements only in bf1 (a-i)
        for char in 'abcdefghi':
            assert char not in bf_intersect, \
                f"Intersection should not contain '{char}' (only in bf1)"

        # Should NOT contain elements only in bf2 (q-z)
        for char in 'qrstuvwxyz':
            assert char not in bf_intersect, \
                f"Intersection should not contain '{char}' (only in bf2)"

    def test_intersection_operator_syntax(self):
        """Test that & operator works for intersection.

        Purpose:
            Verify that the bitwise AND operator is correctly implemented.

        Expected:
            bf1 & bf2 should produce same result as bf1.intersection(bf2).
        """
        bf1 = BloomFilter(capacity=50, error_rate=0.01)
        bf2 = BloomFilter(capacity=50, error_rate=0.01)

        bf1.add("apple")
        bf1.add("banana")
        bf2.add("banana")
        bf2.add("cherry")

        bf_intersect = bf1 & bf2

        assert "banana" in bf_intersect  # In both
        assert "apple" not in bf_intersect  # Only in bf1
        assert "cherry" not in bf_intersect  # Only in bf2

    def test_intersection_requires_matching_capacity(self):
        """Test that intersection fails with mismatched capacities.

        Purpose:
            Validate that intersection only works on compatible filters.

        Expected:
            ValueError should be raised for different capacities.
        """
        bf1 = BloomFilter(capacity=100, error_rate=0.01)
        bf2 = BloomFilter(capacity=200, error_rate=0.01)

        with pytest.raises(ValueError, match="equal capacity and error rate"):
            bf1.intersection(bf2)

    def test_intersection_requires_matching_error_rate(self):
        """Test that intersection fails with mismatched error rates.

        Purpose:
            Validate that intersection only works on compatible filters.

        Expected:
            ValueError should be raised for different error rates.
        """
        bf1 = BloomFilter(capacity=100, error_rate=0.01)
        bf2 = BloomFilter(capacity=100, error_rate=0.001)

        with pytest.raises(ValueError, match="equal capacity and error rate"):
            bf1.intersection(bf2)


# =============================================================================
# BloomFilter Copy Tests
# =============================================================================

class TestBloomFilterCopy:
    """Test BloomFilter copy functionality."""

    def test_copy_creates_independent_filter(self):
        """Test that copy creates an independent filter.

        Purpose:
            Verify that modifications to a copy don't affect the original.

        Expected:
            Original and copy should be independent after copying.
        """
        original = BloomFilter(capacity=100, error_rate=0.01)
        original.add("original_element")

        copied = original.copy()
        copied.add("copied_element")

        # Original should not have element added to copy
        assert "original_element" in original
        assert "copied_element" not in original

        # Copy should have both elements
        assert "original_element" in copied
        assert "copied_element" in copied

    def test_copy_preserves_count(self):
        """Test that copy preserves the element count.

        Purpose:
            Verify that the count property is correctly copied.

        Expected:
            Copy should have same count as original at time of copy.
        """
        original = BloomFilter(capacity=100, error_rate=0.01)
        for i in range(10):
            original.add(f"item_{i}")

        copied = original.copy()
        assert len(copied) == len(original) == 10

    def test_copy_preserves_parameters(self):
        """Test that copy preserves capacity and error rate.

        Purpose:
            Verify that filter parameters are correctly copied.

        Expected:
            Copy should have same capacity and error_rate as original.
        """
        original = BloomFilter(capacity=500, error_rate=0.05)
        copied = original.copy()

        assert copied.capacity == original.capacity
        assert copied.error_rate == original.error_rate


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    """Test serialization and deserialization of Bloom filters."""

    @pytest.mark.parametrize("capacity,error_rate", [
        (100, 0.01),
        (1000, 0.001),
        (10000, 0.1),
    ])
    def test_bloomfilter_file_serialization(self, capacity, error_rate):
        """Test BloomFilter serialization to file.

        Purpose:
            Verify that filters can be saved to files and restored
            with all data intact.

        Expected:
            Deserialized filter should contain all elements from original.
        """
        original = BloomFilter(capacity=capacity, error_rate=error_rate)

        # Add test data
        test_elements = [f"element_{i}" for i in range(min(50, capacity))]
        for elem in test_elements:
            original.add(elem)

        # Serialize to temporary file
        with tempfile.TemporaryFile(mode='w+b') as f:
            original.tofile(f)
            f.seek(0)
            restored = BloomFilter.fromfile(f)

        # Verify all elements present
        for elem in test_elements:
            assert elem in restored, f"'{elem}' should be in restored filter"

        # Verify parameters preserved
        assert restored.capacity == original.capacity
        assert restored.error_rate == original.error_rate
        assert len(restored) == len(original)

    def test_bloomfilter_bytesio_serialization(self):
        """Test BloomFilter serialization to BytesIO.

        Purpose:
            Verify that filters work with in-memory binary streams.

        Expected:
            BytesIO serialization should work like file serialization.
        """
        original = BloomFilter(capacity=100, error_rate=0.01)
        original.add("test_element")

        # Serialize to BytesIO
        buffer = io.BytesIO()
        original.tofile(buffer)
        buffer.seek(0)
        restored = BloomFilter.fromfile(buffer)

        assert "test_element" in restored

    def test_scalablebloomfilter_serialization(self):
        """Test ScalableBloomFilter serialization.

        Purpose:
            Verify that scalable filters with multiple internal filters
            can be serialized and restored correctly.

        Expected:
            All internal filters and their data should be preserved.
        """
        original = ScalableBloomFilter(initial_capacity=10)

        # Add enough data to trigger scaling (multiple internal filters)
        for i in range(100):
            original.add(f"item_{i}")

        # Should have created multiple internal filters
        assert len(original.filters) > 1, "Should have scaled to multiple filters"

        # Serialize and restore
        with tempfile.TemporaryFile(mode='w+b') as f:
            original.tofile(f)
            f.seek(0)
            restored = ScalableBloomFilter.fromfile(f)

        # Verify all elements
        for i in range(100):
            assert f"item_{i}" in restored

        # Verify structure preserved
        assert len(restored.filters) == len(original.filters)
        assert restored.capacity == original.capacity

    def test_pickle_support(self):
        """Test that BloomFilter can be pickled.

        Purpose:
            Verify pickle serialization works via __getstate__/__setstate__.

        Expected:
            Pickled and unpickled filter should work identically.
        """
        original = BloomFilter(capacity=100, error_rate=0.01)
        original.add("pickle_test")

        # Pickle and unpickle
        pickled = pickle.dumps(original)
        restored = pickle.loads(pickled)

        assert "pickle_test" in restored
        assert len(restored) == len(original)


# =============================================================================
# ScalableBloomFilter Tests
# =============================================================================

class TestScalableBloomFilter:
    """Test ScalableBloomFilter specific functionality."""

    def test_initialization_defaults(self):
        """Test ScalableBloomFilter initialization with defaults.

        Purpose:
            Verify default parameters are set correctly.

        Expected:
            Default initial_capacity=100, error_rate=0.001,
            mode=LARGE_SET_GROWTH, no initial filters.
        """
        sbf = ScalableBloomFilter()
        assert sbf.initial_capacity == 100
        assert sbf.error_rate == 0.001
        assert sbf.scale == ScalableBloomFilter.LARGE_SET_GROWTH
        assert len(sbf.filters) == 0

    def test_initialization_custom_parameters(self):
        """Test ScalableBloomFilter with custom parameters.

        Purpose:
            Verify custom parameters are stored correctly.

        Expected:
            Custom parameters should be used.
        """
        sbf = ScalableBloomFilter(
            initial_capacity=50,
            error_rate=0.05,
            mode=ScalableBloomFilter.SMALL_SET_GROWTH
        )
        assert sbf.initial_capacity == 50
        assert sbf.error_rate == 0.05
        assert sbf.scale == ScalableBloomFilter.SMALL_SET_GROWTH

    def test_invalid_error_rate(self):
        """Test that invalid error rate raises ValueError.

        Purpose:
            Validate input error rate must be positive.

        Expected:
            ValueError for zero or negative error_rate.
        """
        with pytest.raises(ValueError):
            ScalableBloomFilter(error_rate=0)

        with pytest.raises(ValueError):
            ScalableBloomFilter(error_rate=-0.1)

    def test_first_add_creates_filter(self):
        """Test that first addition creates initial filter.

        Purpose:
            Verify lazy initialization - filter created on first add.

        Expected:
            filters list should be empty initially, contain 1 after first add.
        """
        sbf = ScalableBloomFilter(initial_capacity=10)
        assert len(sbf.filters) == 0

        sbf.add("first_element")
        assert len(sbf.filters) == 1

    def test_scaling_creates_new_filters(self):
        """Test that exceeding capacity creates new filters.

        Purpose:
            Verify automatic scaling behavior when capacity is reached.

        Expected:
            New filter should be created when current filter fills up.
        """
        sbf = ScalableBloomFilter(initial_capacity=5)

        # Add beyond initial capacity
        for i in range(15):
            sbf.add(f"item_{i}")

        # Should have scaled to multiple filters
        assert len(sbf.filters) > 1, "Should create additional filters when scaling"

    def test_large_set_growth_mode(self):
        """Test LARGE_SET_GROWTH scaling factor.

        Purpose:
            Verify that filters grow by factor of 4 in LARGE_SET_GROWTH mode.

        Expected:
            Each new filter should be 4x larger than previous.
        """
        sbf = ScalableBloomFilter(
            initial_capacity=10,
            mode=ScalableBloomFilter.LARGE_SET_GROWTH
        )

        # Fill up first filter and trigger second
        for i in range(50):
            sbf.add(f"item_{i}")

        if len(sbf.filters) >= 2:
            assert sbf.filters[1].capacity == sbf.filters[0].capacity * 4

    def test_small_set_growth_mode(self):
        """Test SMALL_SET_GROWTH scaling factor.

        Purpose:
            Verify that filters grow by factor of 2 in SMALL_SET_GROWTH mode.

        Expected:
            Each new filter should be 2x larger than previous.
        """
        sbf = ScalableBloomFilter(
            initial_capacity=10,
            mode=ScalableBloomFilter.SMALL_SET_GROWTH
        )

        # Fill up first filter and trigger second
        for i in range(30):
            sbf.add(f"item_{i}")

        if len(sbf.filters) >= 2:
            assert sbf.filters[1].capacity == sbf.filters[0].capacity * 2

    def test_add_returns_true_for_duplicates(self):
        """Test that add returns True for duplicate elements.

        Purpose:
            Verify duplicate detection works across multiple internal filters.

        Expected:
            add() returns False first time, True for duplicates.
        """
        sbf = ScalableBloomFilter()

        result1 = sbf.add("test_element")
        assert result1 is False, "First add should return False"

        result2 = sbf.add("test_element")
        assert result2 is True, "Duplicate add should return True"

    def test_contains_checks_all_filters(self):
        """Test that membership check searches all internal filters.

        Purpose:
            Verify that 'in' operator checks all internal filters,
            not just the most recent one.

        Expected:
            Elements from early filters should still be found.
        """
        sbf = ScalableBloomFilter(initial_capacity=5)

        # Add element to first filter
        sbf.add("early_element")

        # Add many more to create additional filters
        for i in range(50):
            sbf.add(f"item_{i}")

        # Original element should still be found
        assert "early_element" in sbf, \
            "Should find elements from first filter"

    def test_len_sums_all_filters(self):
        """Test that len() returns total count across all filters.

        Purpose:
            Verify length calculation includes all internal filters.

        Expected:
            len() should equal sum of counts from all internal filters.
        """
        sbf = ScalableBloomFilter(initial_capacity=5)

        # Add elements to trigger multiple filters
        num_elements = 20
        for i in range(num_elements):
            sbf.add(f"item_{i}")

        assert len(sbf) == num_elements

    def test_capacity_property_sums_all_filters(self):
        """Test that capacity property returns total capacity.

        Purpose:
            Verify capacity calculation includes all internal filters.

        Expected:
            capacity should equal sum of capacities from all filters.
        """
        sbf = ScalableBloomFilter(initial_capacity=10)

        # Trigger scaling
        for i in range(50):
            sbf.add(f"item_{i}")

        # Capacity should be sum of all internal filter capacities
        expected_capacity = sum(f.capacity for f in sbf.filters)
        assert sbf.capacity == expected_capacity

    def test_union_scalable_filters(self):
        """Test union of two ScalableBloomFilters.

        Purpose:
            Verify that ScalableBloomFilter union combines all elements.

        Expected:
            Union should contain elements from both filters.
        """
        sbf1 = ScalableBloomFilter(
            initial_capacity=10,
            mode=ScalableBloomFilter.SMALL_SET_GROWTH
        )
        sbf2 = ScalableBloomFilter(
            initial_capacity=10,
            mode=ScalableBloomFilter.SMALL_SET_GROWTH
        )

        # Add different ranges to each filter
        for i in range(50):
            sbf1.add(i)
        for i in range(50, 100):
            sbf2.add(i)

        # Union should have all elements
        sbf_union = sbf1 | sbf2

        for i in range(100):
            assert i in sbf_union, f"Union should contain {i}"

    def test_union_requires_matching_parameters(self):
        """Test that union validates matching parameters.

        Purpose:
            Verify that union only works on compatible scalable filters.

        Expected:
            ValueError for mismatched parameters.
        """
        sbf1 = ScalableBloomFilter(initial_capacity=100, error_rate=0.001)
        sbf2 = ScalableBloomFilter(initial_capacity=200, error_rate=0.001)

        with pytest.raises(ValueError, match="same mode, initial capacity and error rate"):
            sbf1.union(sbf2)


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_filter_membership(self):
        """Test membership on empty filter always returns False.

        Purpose:
            Verify that empty filters correctly report no memberships.

        Expected:
            Any membership test on empty filter should return False.
        """
        bf = BloomFilter(capacity=100, error_rate=0.01)
        assert "anything" not in bf
        assert 42 not in bf
        assert b"bytes" not in bf

    def test_very_small_capacity(self):
        """Test filter with capacity of 1.

        Purpose:
            Verify filter works even with minimal capacity.

        Expected:
            Should function normally with just 1 capacity.
        """
        bf = BloomFilter(capacity=1, error_rate=0.1)
        bf.add("single_item")
        assert "single_item" in bf

    def test_very_high_error_rate(self):
        """Test filter with high error rate (0.5).

        Purpose:
            Verify filter works with higher error rates (less memory).

        Expected:
            Should create valid filter with fewer bits.
        """
        bf = BloomFilter(capacity=100, error_rate=0.5)
        bf.add("test")
        assert "test" in bf

    def test_very_low_error_rate(self):
        """Test filter with very low error rate (0.00001).

        Purpose:
            Verify filter works with very low error rates (more memory).

        Expected:
            Should create valid filter with more bits.
        """
        bf = BloomFilter(capacity=100, error_rate=0.00001)
        bf.add("test")
        assert "test" in bf

    def test_unicode_strings(self):
        """Test filter with unicode strings.

        Purpose:
            Verify UTF-8 encoding works correctly for unicode.

        Expected:
            Unicode strings should be properly encoded and hashed.
        """
        bf = BloomFilter(capacity=100, error_rate=0.01)

        unicode_strings = ["hello", "世界", "🎉", "Ñoño", "Кириллица"]
        for s in unicode_strings:
            bf.add(s)

        for s in unicode_strings:
            assert s in bf, f"Unicode string '{s}' should be in filter"

    def test_empty_string(self):
        """Test filter with empty string.

        Purpose:
            Verify empty string is handled correctly.

        Expected:
            Empty string should work like any other string.
        """
        bf = BloomFilter(capacity=100, error_rate=0.01)
        bf.add("")
        assert "" in bf

    def test_large_strings(self):
        """Test filter with very large strings.

        Purpose:
            Verify filter handles large string inputs efficiently.

        Expected:
            Large strings should be hashed and stored correctly.
        """
        bf = BloomFilter(capacity=100, error_rate=0.01)

        large_string = "x" * 10000
        bf.add(large_string)
        assert large_string in bf


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
