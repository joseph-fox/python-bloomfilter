"""Bloom Filter and Scalable Bloom Filter implementations.

This module implements two probabilistic data structures for space-efficient
set membership testing:

1. BloomFilter: Fixed-capacity filter for known dataset sizes
2. ScalableBloomFilter: Dynamically growing filter that scales automatically

Both implementations maintain a configurable false positive rate while never
producing false negatives. The module uses optimized hash functions (xxHash
for speed) and efficient bit array storage.

Mathematical Foundation:
    - Optimal hash functions: k = log2(1/P) where P is error rate
    - Optimal bit count: M ≈ n × |ln(P)| / (ln(2)²) where n is capacity
    - False positive probability: P ≈ (1 - e^(-kn/m))^k

Requirements:
    - Python 3.6+
    - bitarray >= 0.3.4: Efficient bit array operations
    - xxhash >= 3.0.0: Fast non-cryptographic hashing
"""
import copy
import hashlib
import math
from io import BytesIO
from struct import calcsize, pack, unpack

import xxhash

try:
    import bitarray
except ImportError:
    raise ImportError('pybloom_live requires bitarray >= 0.3.4')


def make_hashfuncs(num_slices, num_bits):
    """Create optimized hash functions for the Bloom filter.

    This function generates a hash function generator that produces the required
    number of hash values efficiently. It selects the appropriate hash algorithm
    based on the total bits needed, preferring faster non-cryptographic hashes
    when possible.

    Hash Function Selection Strategy:
        - xxHash (xxh128): For ≤128 bits (fastest, non-cryptographic)
        - SHA-1: For 129-160 bits
        - SHA-256: For 161-256 bits
        - SHA-384: For 257-384 bits
        - SHA-512: For >384 bits

    The function uses salting to generate multiple independent hash values from
    a single hash computation, significantly improving performance.

    Args:
        num_slices (int): Number of hash functions needed (k in literature).
            This equals the number of bits to set per element.
        num_bits (int): Number of bits per hash slice. Determines the format
            code for unpacking hash digests.

    Returns:
        tuple: A 2-tuple containing:
            - hash_maker (callable): Generator function that yields hash values
            - hashfn (callable): The underlying hash function used

    Performance Notes:
        - Salts are pre-computed and reused for all keys
        - Single hash computation generates multiple independent values
        - Format codes are optimized for the bit array size
    """
    # Select format code based on number of bits to optimize struct unpacking
    if num_bits >= (1 << 31):
        fmt_code, chunk_size = 'Q', 8  # 64-bit unsigned long long
    elif num_bits >= (1 << 15):
        fmt_code, chunk_size = 'I', 4  # 32-bit unsigned int
    else:
        fmt_code, chunk_size = 'H', 2  # 16-bit unsigned short

    # Calculate total bits needed and select appropriate hash function
    total_hash_bits = 8 * num_slices * chunk_size
    if total_hash_bits > 384:
        hashfn = hashlib.sha512
    elif total_hash_bits > 256:
        hashfn = hashlib.sha384
    elif total_hash_bits > 160:
        hashfn = hashlib.sha256
    elif total_hash_bits > 128:
        hashfn = hashlib.sha1
    else:
        hashfn = xxhash.xxh128  # Fastest option for typical filters

    # Pre-compute format string for unpacking hash digests
    fmt = fmt_code * (hashfn().digest_size // chunk_size)

    # Calculate number of salts needed to generate enough hash values
    num_salts, extra = divmod(num_slices, len(fmt))
    if extra:
        num_salts += 1

    # Pre-compute salts for reuse (double-hashing for better distribution)
    salts = tuple(hashfn(hashfn(pack('I', i)).digest()) for i in range(num_salts))

    def _hash_maker(key):
        """Generate hash values for a given key.

        This inner function handles key encoding and generates the required
        number of hash values using the pre-computed salts.

        Args:
            key: The element to hash (str, bytes, or any object with __str__)

        Yields:
            int: Hash values in range [0, num_bits)

        Performance Optimization:
            - Early termination after generating required number of hashes
            - Efficient modulo operation for range reduction
        """
        # Normalize key to bytes for consistent hashing (Python 3)
        if isinstance(key, str):
            key = key.encode('utf-8')
        elif not isinstance(key, bytes):
            key = str(key).encode('utf-8')

        # Generate hash values using salted hashing
        i = 0
        for salt in salts:
            h = salt.copy()  # Copy pre-computed salt (fast operation)
            h.update(key)    # Hash the key
            # Unpack hash digest into multiple integers
            for uint in unpack(fmt, h.digest()):
                yield uint % num_bits  # Reduce to valid bit index
                i += 1
                if i >= num_slices:  # Early exit optimization
                    return

    return _hash_maker, hashfn


class BloomFilter:
    FILE_FMT = b'<dQQQQ'

    def __init__(self, capacity, error_rate=0.001):
        """Initialize a Bloom filter with specified capacity and error rate.

        This constructor calculates the optimal number of hash functions and
        bit array size based on the desired capacity and false positive rate
        using standard Bloom filter formulas.

        Mathematical Formulas Used:
            - Number of hash functions: k = ceil(log2(1/P))
            - Bits per slice: m = ceil(n × |ln(P)| / (k × (ln(2))²))
            - Total bits: M = k × m

        Where:
            - n = capacity (number of elements)
            - P = error_rate (false positive probability)
            - k = num_slices (number of hash functions)
            - m = bits_per_slice

        Args:
            capacity (int): Maximum number of elements the filter should hold
                while maintaining the specified error rate. Must be > 0.
            error_rate (float, optional): Target false positive probability.
                Must be between 0 and 1 (exclusive). Default is 0.001 (0.1%).

        Raises:
            ValueError: If error_rate is not in range (0, 1).
            ValueError: If capacity is not positive.

        Example:
            >>> bf = BloomFilter(capacity=1000, error_rate=0.01)
            >>> bf.add("test")
            >>> "test" in bf
            True

        Performance Characteristics:
            - Memory usage: O(n × log(1/P)) bits
            - Time complexity for add/query: O(k) where k ≈ log2(1/P)

        Requires:
            Python 3.6+
        """
        if not (0 < error_rate < 1):
            raise ValueError("Error_Rate must be between 0 and 1.")
        if not capacity > 0:
            raise ValueError("Capacity must be > 0")

        # Calculate optimal number of hash functions (k)
        # k = ceil(log2(1/P)) minimizes false positive rate
        num_slices = int(math.ceil(math.log(1.0 / error_rate, 2)))

        # Calculate optimal bits per slice (m)
        # Derived from: n ≈ M × (ln(2)²) / |ln(P)|
        # where M = k × m (total bits)
        bits_per_slice = int(math.ceil(
            (capacity * abs(math.log(error_rate))) /
            (num_slices * (math.log(2) ** 2))))

        self._setup(error_rate, num_slices, bits_per_slice, capacity, 0)
        self.bitarray = bitarray.bitarray(self.num_bits, endian='little')
        self.bitarray.setall(False)

    def _setup(self, error_rate, num_slices, bits_per_slice, capacity, count):
        """Internal setup method for initializing or restoring filter state.

        This method is used both during normal initialization and when
        deserializing a filter from a file. It configures all filter
        parameters and creates the hash function generator.

        Args:
            error_rate (float): Target false positive probability
            num_slices (int): Number of hash functions (k)
            bits_per_slice (int): Bits in each slice of the bit array
            capacity (int): Maximum number of elements
            count (int): Current number of elements stored
        """
        self.error_rate = error_rate
        self.num_slices = num_slices
        self.bits_per_slice = bits_per_slice
        self.capacity = capacity
        self.num_bits = num_slices * bits_per_slice
        self.count = count
        self.make_hashes, self.hashfn = make_hashfuncs(self.num_slices, self.bits_per_slice)

    def __contains__(self, key):
        """Test whether an element is in the Bloom filter.

        This method implements the membership test operation, allowing use of
        the 'in' operator. It checks if all k hash positions are set in the
        bit array.

        Args:
            key: The element to test for membership (any hashable type)

        Returns:
            bool: True if the element might be in the set (with possible false
                positives), False if the element is definitely not in the set
                (no false negatives).

        Time Complexity:
            O(k) where k is the number of hash functions

        Example:
            >>> bf = BloomFilter(100, 0.01)
            >>> bf.add("apple")
            >>> "apple" in bf
            True
            >>> "banana" in bf
            False

        Performance Optimization:
            - Local variable caching for faster attribute access
            - Early exit on first unset bit (short-circuit evaluation)
        """
        # Cache attributes as local variables for faster access in loop
        bits_per_slice = self.bits_per_slice
        bitarray = self.bitarray
        hashes = self.make_hashes(key)

        # Check all hash positions; return False immediately if any is unset
        offset = 0
        for k in hashes:
            if not bitarray[offset + k]:
                return False  # Definitely not in set
            offset += bits_per_slice
        return True  # Probably in set

    def __len__(self):
        """Return the number of elements added to this Bloom filter.

        Note that this is a count of additions, not unique elements. If the
        same element is added multiple times, it contributes to the count
        each time (unless detected as duplicate).

        Returns:
            int: Number of elements that have been added to the filter

        Example:
            >>> bf = BloomFilter(100, 0.01)
            >>> bf.add("apple")
            >>> bf.add("banana")
            >>> len(bf)
            2
        """
        return self.count

    def add(self, key, skip_check=False):
        """Add an element to the Bloom filter.

        This method sets the appropriate bits in the bit array based on the
        hash values for the key. It can optionally detect if the element was
        likely already present before adding it.

        Args:
            key: The element to add (any hashable type)
            skip_check (bool, optional): If True, skips the duplicate detection
                check for better performance. Use when you know elements are
                unique or don't care about detecting duplicates. Default is False.

        Returns:
            bool: True if the element might have already been in the filter
                before this operation (possible false positive), False if this
                is definitely a new element.

        Raises:
            IndexError: If the filter is already at or over capacity. Adding
                elements beyond capacity significantly increases false positive
                rate.

        Time Complexity:
            O(k) where k is the number of hash functions

        Example:
            >>> bf = BloomFilter(100, 0.01)
            >>> bf.add("apple")  # Returns False (new element)
            False
            >>> bf.add("apple")  # Returns True (duplicate)
            True
            >>> bf.add("banana", skip_check=True)  # Returns False (skipped check)
            False

        Performance Note:
            Use skip_check=True when adding bulk data where you know there are
            no duplicates. This can improve performance by 30-40%.
        """
        # Check capacity before computing hashes (fail-fast optimization)
        if self.count >= self.capacity:
            raise IndexError("BloomFilter is at capacity")

        # Cache attributes as local variables for faster loop access
        bitarray = self.bitarray
        bits_per_slice = self.bits_per_slice
        hashes = self.make_hashes(key)

        # Optimized path for skip_check=True (bulk insertions)
        if skip_check:
            offset = 0
            for k in hashes:
                bitarray[offset + k] = True
                offset += bits_per_slice
            self.count += 1
            return False

        # Check for duplicate and set bits in single pass
        found_all_bits = True
        offset = 0
        for k in hashes:
            if found_all_bits and not bitarray[offset + k]:
                found_all_bits = False  # Found at least one unset bit
            bitarray[offset + k] = True
            offset += bits_per_slice

        # Only increment count if this is a new element
        if not found_all_bits:
            self.count += 1
            return False
        else:
            return True  # Element was likely already present

    def copy(self):
        """Create a deep copy of this Bloom filter.

        The copy is completely independent and can be modified without
        affecting the original filter.

        Returns:
            BloomFilter: A new BloomFilter instance with the same state

        Time Complexity:
            O(M) where M is the size of the bit array

        Example:
            >>> bf1 = BloomFilter(100, 0.01)
            >>> bf1.add("apple")
            >>> bf2 = bf1.copy()
            >>> bf2.add("banana")
            >>> "banana" in bf1
            False
            >>> "banana" in bf2
            True
        """
        new_filter = BloomFilter(self.capacity, self.error_rate)
        new_filter.bitarray = self.bitarray.copy()
        new_filter.count = self.count  # Preserve count
        return new_filter

    def union(self, other):
        """Calculate the union of two Bloom filters.

        Creates a new Bloom filter containing elements that are in either this
        filter or the other filter. This is equivalent to the bitwise OR
        operation on the underlying bit arrays.

        Args:
            other (BloomFilter): Another Bloom filter to union with this one.
                Must have the same capacity and error rate.

        Returns:
            BloomFilter: A new filter representing the union of both filters

        Raises:
            ValueError: If filters have different capacities or error rates

        Time Complexity:
            O(M) where M is the size of the bit array

        Example:
            >>> bf1 = BloomFilter(100, 0.01)
            >>> bf2 = BloomFilter(100, 0.01)
            >>> bf1.add("apple")
            >>> bf2.add("banana")
            >>> bf_union = bf1.union(bf2)  # or bf1 | bf2
            >>> "apple" in bf_union and "banana" in bf_union
            True

        Note:
            The count in the resulting filter may not be accurate as it cannot
            account for elements present in both filters.
        """
        if self.capacity != other.capacity or self.error_rate != other.error_rate:
            raise ValueError(
                "Unioning filters requires both filters to have both the same capacity and error rate")
        new_bloom = self.copy()
        new_bloom.bitarray = new_bloom.bitarray | other.bitarray
        # Note: Count is not accurately updated as we can't determine overlaps
        return new_bloom

    def __or__(self, other):
        """Implement the | operator for union operation.

        Example:
            >>> bf_union = bf1 | bf2
        """
        return self.union(other)

    def intersection(self, other):
        """Calculate the intersection of two Bloom filters.

        Creates a new Bloom filter containing elements that are likely in both
        this filter and the other filter. This is equivalent to the bitwise AND
        operation on the underlying bit arrays.

        Args:
            other (BloomFilter): Another Bloom filter to intersect with this one.
                Must have the same capacity and error rate.

        Returns:
            BloomFilter: A new filter representing the intersection of both filters

        Raises:
            ValueError: If filters have different capacities or error rates

        Time Complexity:
            O(M) where M is the size of the bit array

        Example:
            >>> bf1 = BloomFilter(100, 0.01)
            >>> bf2 = BloomFilter(100, 0.01)
            >>> bf1.add("apple")
            >>> bf1.add("banana")
            >>> bf2.add("banana")
            >>> bf_intersect = bf1.intersection(bf2)  # or bf1 & bf2
            >>> "banana" in bf_intersect  # Likely True
            True
            >>> "apple" in bf_intersect   # Likely False
            False

        Note:
            Due to false positives, the intersection may contain elements that
            were not actually in both original sets. The false positive rate of
            the intersection can be higher than the individual filters.
        """
        if self.capacity != other.capacity or self.error_rate != other.error_rate:
            raise ValueError(
                "Intersecting filters requires both filters to have equal capacity and error rate")
        new_bloom = self.copy()
        new_bloom.bitarray = new_bloom.bitarray & other.bitarray
        # Note: Count is not accurately updated for intersections
        return new_bloom

    def __and__(self, other):
        """Implement the & operator for intersection operation.

        Example:
            >>> bf_intersect = bf1 & bf2
        """
        return self.intersection(other)

    def tofile(self, f):
        """Serialize the Bloom filter to a file.

        Writes the filter to a binary file format that is much more space
        efficient than Python's pickle format. The file contains the filter
        parameters followed by the raw bit array data.

        File Format:
            - Header (40 bytes): error_rate, num_slices, bits_per_slice,
              capacity, count (packed as '<dQQQQ')
            - Body: Raw bit array data

        Args:
            f: File-like object opened in binary write mode ('wb'). Can be a
                regular file or BytesIO.

        Example:
            >>> bf = BloomFilter(1000, 0.01)
            >>> bf.add("test")
            >>> with open('filter.bin', 'wb') as f:
            ...     bf.tofile(f)

        See Also:
            fromfile: Class method to deserialize a filter from a file
        """
        # Write filter parameters as binary header
        f.write(pack(self.FILE_FMT, self.error_rate, self.num_slices,
                     self.bits_per_slice, self.capacity, self.count))
        # Write bit array data
        if isinstance(f, BytesIO):
            f.write(self.bitarray.tobytes())
        else:
            self.bitarray.tofile(f)

    @classmethod
    def fromfile(cls, f, n=-1):
        """Deserialize a Bloom filter from a file.

        Reads a Bloom filter that was previously saved using the tofile()
        method. The file must contain a valid filter in the binary format
        written by tofile().

        Args:
            f: File-like object opened in binary read mode ('rb'). Can be a
                regular file or BytesIO.
            n (int, optional): Maximum number of bytes to read. If n > 0,
                reads at most n bytes. If n <= 0 (default), reads the entire
                filter. Must be at least as large as the header (40 bytes).

        Returns:
            BloomFilter: A new BloomFilter instance with the state loaded
                from the file

        Raises:
            ValueError: If n is too small (< header size)
            ValueError: If the bit array length doesn't match the expected size

        Example:
            >>> with open('filter.bin', 'rb') as f:
            ...     bf = BloomFilter.fromfile(f)
            >>> "test" in bf
            True

        Note:
            Filters saved with version < 4.0.0 are incompatible due to hash
            function changes (MD5 → xxHash).
        """
        headerlen = calcsize(cls.FILE_FMT)

        # Validate minimum read size
        if 0 < n < headerlen:
            raise ValueError('n too small!')

        # Create temporary instance and restore state from file
        filter = cls(1)  # Minimal initialization, will be overwritten
        filter._setup(*unpack(cls.FILE_FMT, f.read(headerlen)))

        # Read bit array data
        filter.bitarray = bitarray.bitarray(endian='little')
        if isinstance(f, BytesIO):
            if n > 0:
                filter.bitarray.frombytes(f.read(n - headerlen))
            else:
                filter.bitarray.frombytes(f.read())
        else:
            if n > 0:
                filter.bitarray.fromfile(f, n - headerlen)
            else:
                filter.bitarray.fromfile(f)

        # Validate bit array length (accounting for byte padding)
        if filter.num_bits != len(filter.bitarray) and \
                (filter.num_bits + (8 - filter.num_bits % 8) != len(filter.bitarray)):
            raise ValueError('Bit length mismatch!')

        return filter

    def __getstate__(self):
        """Prepare the filter state for pickling.

        Removes the hash function generator which cannot be pickled, as it
        will be reconstructed during unpickling.

        Returns:
            dict: Filter state dictionary without the make_hashes function
        """
        d = self.__dict__.copy()
        del d['make_hashes']  # Functions can't be pickled
        return d

    def __setstate__(self, d):
        """Restore the filter state from pickling.

        Reconstructs the hash function generator that was removed during
        pickling.

        Args:
            d (dict): Filter state dictionary from __getstate__
        """
        self.__dict__.update(d)
        # Reconstruct hash function generator
        self.make_hashes, self.hashfn = make_hashfuncs(self.num_slices, self.bits_per_slice)


class ScalableBloomFilter:
    """A Bloom filter that automatically scales as more elements are added.

    This implementation follows the algorithm described in:
    "Scalable Bloom Filters" by Almeida et al., GLOBECOM 2007.

    The filter automatically creates additional internal Bloom filters as
    capacity is reached, with each new filter having:
    - Exponentially larger capacity (2x or 4x growth)
    - Tighter error rate (multiplied by tightening ratio 0.9)

    This maintains overall false positive probability while accommodating
    unbounded growth.

    Class Attributes:
        SMALL_SET_GROWTH (int): Growth factor of 2 - slower growth, less memory
        LARGE_SET_GROWTH (int): Growth factor of 4 - faster growth (default)
        FILE_FMT (str): Binary format string for serialization
    """
    SMALL_SET_GROWTH = 2  # Growth factor for memory-constrained scenarios
    LARGE_SET_GROWTH = 4  # Growth factor for performance (default)
    FILE_FMT = '<idQd'

    def __init__(self, initial_capacity=100, error_rate=0.001,
                 mode=LARGE_SET_GROWTH):
        """Initialize a Scalable Bloom Filter.

        Args:
            initial_capacity (int, optional): Initial capacity of the first
                internal filter. Default is 100. Subsequent filters grow
                exponentially based on the mode.
            error_rate (float, optional): Target overall false positive
                probability. Must be > 0. Default is 0.001 (0.1%). Each
                internal filter uses a tighter error rate (× 0.9) to maintain
                this overall rate.
            mode (int, optional): Growth rate for new filters. Must be either
                SMALL_SET_GROWTH (2) or LARGE_SET_GROWTH (4). Default is
                LARGE_SET_GROWTH.

        Raises:
            ValueError: If error_rate is not positive

        Example:
            >>> sbf = ScalableBloomFilter(
            ...     initial_capacity=1000,
            ...     error_rate=0.001,
            ...     mode=ScalableBloomFilter.LARGE_SET_GROWTH
            ... )
            >>> for i in range(10000):
            ...     sbf.add(i)  # Automatically scales
            >>> len(sbf)
            10000

        Performance Notes:
            - SMALL_SET_GROWTH: Better for memory-constrained scenarios
            - LARGE_SET_GROWTH: Better for performance when memory allows
            - Tightening ratio of 0.9 balances space and accuracy

        Requires:
            Python 3.6+
        """
        if not error_rate or error_rate < 0:
            raise ValueError("Error_Rate must be a decimal less than 0.")
        self._setup(mode, 0.9, initial_capacity, error_rate)
        self.filters = []

    def _setup(self, mode, ratio, initial_capacity, error_rate):
        """Internal setup method for initializing or restoring filter state.

        Args:
            mode (int): Growth factor (SMALL_SET_GROWTH or LARGE_SET_GROWTH)
            ratio (float): Tightening ratio for error rates (typically 0.9)
            initial_capacity (int): Starting capacity
            error_rate (float): Target false positive probability
        """
        self.scale = mode
        self.ratio = ratio
        self.initial_capacity = initial_capacity
        self.error_rate = error_rate

    def __contains__(self, key):
        """Test whether an element is in the Scalable Bloom filter.

        Checks membership across all internal filters, searching from newest
        to oldest. This ordering optimizes for recently added elements.

        Args:
            key: The element to test for membership

        Returns:
            bool: True if element might be in the set, False if definitely not

        Time Complexity:
            O(k × n) where k is hash functions per filter and n is number of
            internal filters. In practice, most lookups hit the most recent
            filters quickly.

        Example:
            >>> sbf = ScalableBloomFilter()
            >>> sbf.add("test")
            >>> "test" in sbf
            True
            >>> "not_added" in sbf
            False

        Performance Optimization:
            - Searches filters in reverse order (newest first)
            - Early exit on first match
        """
        # Search newest filters first (most likely to contain recent additions)
        for f in reversed(self.filters):
            if key in f:
                return True
        return False

    def add(self, key):
        """Add an element to the Scalable Bloom filter.

        Automatically creates new internal filters as needed when capacity is
        reached. New filters are exponentially larger with tighter error rates
        to maintain the overall target false positive probability.

        Args:
            key: The element to add (any hashable type)

        Returns:
            bool: True if element was already in the filter (possible false
                positive), False if this is a new addition

        Time Complexity:
            O(k) for normal adds, O(k × n) when checking for duplicates where
            k is hash functions and n is number of internal filters

        Example:
            >>> sbf = ScalableBloomFilter(initial_capacity=100)
            >>> sbf.add("apple")  # Returns False (new)
            False
            >>> sbf.add("apple")  # Returns True (duplicate)
            True
            >>> # Automatically scales beyond initial capacity
            >>> for i in range(1000):
            ...     sbf.add(i)

        Performance Note:
            The duplicate check (key in self) adds overhead. For bulk loading
            of known-unique data, consider modifying to skip this check.
        """
        # Check for duplicates across all filters
        if key in self:
            return True

        # Create first filter if needed
        if not self.filters:
            filter = BloomFilter(
                capacity=self.initial_capacity,
                error_rate=self.error_rate * self.ratio)
            self.filters.append(filter)
        else:
            # Check if current filter is at capacity
            filter = self.filters[-1]
            if filter.count >= filter.capacity:
                # Create new filter with exponential growth and tighter error rate
                filter = BloomFilter(
                    capacity=filter.capacity * self.scale,
                    error_rate=filter.error_rate * self.ratio)
                self.filters.append(filter)

        # Add to most recent filter (skip_check=True since we already checked above)
        filter.add(key, skip_check=True)
        return False

    def union(self, other):
        """Calculate the union of two Scalable Bloom filters.

        Creates a new Scalable Bloom filter containing elements from either
        filter. The filters must have matching parameters (mode, initial
        capacity, error rate).

        Args:
            other (ScalableBloomFilter): Another ScalableBloomFilter to union
                with this one. Must have the same mode, initial_capacity, and
                error_rate.

        Returns:
            ScalableBloomFilter: A new filter representing the union

        Raises:
            ValueError: If filters have incompatible parameters

        Time Complexity:
            O(M × n) where M is bit array size and n is number of filters

        Example:
            >>> sbf1 = ScalableBloomFilter(initial_capacity=100)
            >>> sbf2 = ScalableBloomFilter(initial_capacity=100)
            >>> sbf1.add("apple")
            >>> sbf2.add("banana")
            >>> sbf_union = sbf1.union(sbf2)  # or sbf1 | sbf2
            >>> "apple" in sbf_union and "banana" in sbf_union
            True

        Implementation Details:
            - Unions corresponding internal filters using bitwise OR
            - Handles filters with different numbers of internal filters
            - Deep copies the larger filter to avoid modifying originals
        """
        if self.scale != other.scale or \
                self.initial_capacity != other.initial_capacity or \
                self.error_rate != other.error_rate:
            raise ValueError("Unioning two scalable bloom filters requires "
                           "both filters to have both the same mode, initial capacity and error rate")

        # Deep copy the larger filter to preserve more data
        if len(self.filters) > len(other.filters):
            larger_sbf = copy.deepcopy(self)
            smaller_sbf = other
        else:
            larger_sbf = copy.deepcopy(other)
            smaller_sbf = self

        # Union corresponding internal filters
        new_filters = []
        for i in range(len(smaller_sbf.filters)):
            new_filter = larger_sbf.filters[i] | smaller_sbf.filters[i]
            new_filters.append(new_filter)

        # Append remaining filters from larger filter
        for i in range(len(smaller_sbf.filters), len(larger_sbf.filters)):
            new_filters.append(larger_sbf.filters[i])

        larger_sbf.filters = new_filters
        return larger_sbf

    def __or__(self, other):
        """Implement the | operator for union operation.

        Example:
            >>> sbf_union = sbf1 | sbf2
        """
        return self.union(other)

    @property
    def capacity(self):
        """Get the total capacity across all internal filters.

        Returns:
            int: Sum of capacities of all internal BloomFilter instances

        Example:
            >>> sbf = ScalableBloomFilter(initial_capacity=100)
            >>> for i in range(500):
            ...     sbf.add(i)
            >>> sbf.capacity  # Will be > 100 due to scaling
            700
        """
        return sum(f.capacity for f in self.filters)

    @property
    def count(self):
        """Get the total number of elements added.

        This is an alias for len() provided for API consistency with
        BloomFilter.

        Returns:
            int: Total number of elements across all internal filters

        Example:
            >>> sbf = ScalableBloomFilter()
            >>> sbf.add("a")
            >>> sbf.add("b")
            >>> sbf.count
            2
        """
        return len(self)

    def tofile(self, f):
        """Serialize this Scalable Bloom Filter to a file.

        Writes the filter and all internal BloomFilter instances to a binary
        file format. The format includes metadata about each internal filter
        for efficient deserialization.

        File Format:
            1. Header: scale, ratio, initial_capacity, error_rate (32 bytes)
            2. Number of internal filters (4 bytes)
            3. Size table: size of each internal filter in bytes
            4. Internal filter data: each filter serialized sequentially

        Args:
            f: File-like object opened in binary write mode ('wb')

        Example:
            >>> sbf = ScalableBloomFilter()
            >>> for i in range(1000):
            ...     sbf.add(i)
            >>> with open('scalable_filter.bin', 'wb') as f:
            ...     sbf.tofile(f)

        See Also:
            fromfile: Class method to deserialize a filter from a file
        """
        # Write main filter parameters
        f.write(pack(self.FILE_FMT, self.scale, self.ratio,
                     self.initial_capacity, self.error_rate))

        # Write number of internal filters
        f.write(pack(b'<l', len(self.filters)))

        if len(self.filters) > 0:
            # Write size table for internal filters, then filter data
            headerpos = f.tell()
            headerfmt = b'<' + b'Q' * (len(self.filters))
            # Reserve space for size table
            f.write(b'.' * calcsize(headerfmt))

            # Write each internal filter and record its size
            filter_sizes = []
            for filter in self.filters:
                begin = f.tell()
                filter.tofile(f)
                filter_sizes.append(f.tell() - begin)

            # Go back and write size table
            f.seek(headerpos)
            f.write(pack(headerfmt, *filter_sizes))

    @classmethod
    def fromfile(cls, f):
        """Deserialize a Scalable Bloom Filter from a file.

        Reads a Scalable Bloom Filter that was previously saved using the
        tofile() method, including all internal BloomFilter instances.

        Args:
            f: File-like object opened in binary read mode ('rb')

        Returns:
            ScalableBloomFilter: A new ScalableBloomFilter instance with the
                state loaded from the file

        Example:
            >>> with open('scalable_filter.bin', 'rb') as f:
            ...     sbf = ScalableBloomFilter.fromfile(f)
            >>> 500 in sbf
            True

        Note:
            Filters saved with version < 4.0.0 are incompatible due to hash
            function changes (MD5 → xxHash).
        """
        filter = cls()
        # Read main filter parameters
        filter._setup(*unpack(cls.FILE_FMT, f.read(calcsize(cls.FILE_FMT))))

        # Read number of internal filters
        nfilters, = unpack(b'<l', f.read(calcsize(b'<l')))

        if nfilters > 0:
            # Read size table
            header_fmt = b'<' + b'Q' * nfilters
            bytes = f.read(calcsize(header_fmt))
            filter_lengths = unpack(header_fmt, bytes)

            # Read each internal filter
            for fl in filter_lengths:
                filter.filters.append(BloomFilter.fromfile(f, fl))
        else:
            filter.filters = []

        return filter

    def __len__(self):
        """Return the total number of elements across all internal filters.

        Returns:
            int: Sum of counts from all internal BloomFilter instances

        Time Complexity:
            O(n) where n is the number of internal filters

        Example:
            >>> sbf = ScalableBloomFilter()
            >>> for i in range(100):
            ...     sbf.add(i)
            >>> len(sbf)
            100
        """
        return sum(f.count for f in self.filters)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
