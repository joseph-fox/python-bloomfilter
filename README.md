[![Build Status](https://travis-ci.org/joseph-fox/python-bloomfilter.svg?branch=master)](https://travis-ci.org/joseph-fox/python-bloomfilter)

# pybloom_live

A fast, scalable Python implementation of Bloom filters - space-efficient probabilistic data structures for set membership testing.

## What is a Bloom Filter?

A Bloom filter is a space-efficient probabilistic data structure that is used to test whether an element is a member of a set. It can tell you with certainty that an element is **not** in the set, or that it **might be** in the set (with a controllable false positive probability). Bloom filters never produce false negatives.

**Use cases:**
- Web crawlers: Track visited URLs to avoid re-crawling
- Caching systems: Quick membership checks before expensive lookups
- Database optimization: Pre-filter queries to avoid unnecessary disk reads
- Distributed systems: Efficient set reconciliation
- Network routers: Fast packet filtering
- Spell checkers: Quick dictionary lookups

## Features

- **Two implementations:**
  - `BloomFilter`: Fixed-capacity filter for known dataset sizes
  - `ScalableBloomFilter`: Automatically grows to accommodate more elements

- **High performance:** Uses xxHash for 128-bit hashing - one of the fastest non-cryptographic hash functions available

- **Set operations:** Union and intersection support for combining filters

- **Serialization:** Efficient binary file format for saving/loading filters

- **Python 3.6+:** Modern Python 3 implementation without legacy Python 2 baggage

- **Optimized space usage:** Tightening ratio of 0.9 provides excellent space efficiency across a wide range of growth patterns

## Installation

Install from PyPI:

```bash
pip install pybloom-live
```

Or install from source:

```bash
git clone https://github.com/joseph-fox/python-bloomfilter.git
cd python-bloomfilter
pip install .
```

Verify installation:

```bash
pip show pybloom-live
# Should show version 4.0.0 or higher
```

## Quick Start

### Basic BloomFilter

```python
import pybloom_live

# Create a Bloom Filter with capacity for 1000 elements
# and a 0.1% false positive rate
f = pybloom_live.BloomFilter(capacity=1000, error_rate=0.001)

# Add elements
f.add("apple")
f.add("banana")
f.add("orange")

# Check membership
print("apple" in f)      # True
print("grape" in f)      # False

# The add() method returns True if the element might already exist
already_exists = f.add("apple")  # Returns True
already_exists = f.add("mango")  # Returns False
```

### ScalableBloomFilter

The `ScalableBloomFilter` automatically grows as you add more elements, maintaining the target error rate:

```python
import pybloom_live

# Create a scalable filter with initial capacity of 100
sbf = pybloom_live.ScalableBloomFilter(
    initial_capacity=100,
    error_rate=0.001,
    mode=pybloom_live.ScalableBloomFilter.LARGE_SET_GROWTH
)

# Add many elements - it scales automatically
for i in range(10000):
    sbf.add(i)

print(f"Elements: {len(sbf)}")           # ~10000
print(f"Capacity: {sbf.capacity}")       # Automatically expanded
print(f"Filters: {len(sbf.filters)}")    # Multiple internal filters

# Check membership
print(5000 in sbf)   # True
print(10000 in sbf)  # False
```

**Growth modes:**
- `SMALL_SET_GROWTH = 2`: Slower growth, uses less memory
- `LARGE_SET_GROWTH = 4`: Faster growth (default), better for large datasets

## Examples

### Example 1: Web Crawler - Tracking Visited URLs

```python
import pybloom_live

# Track visited URLs to avoid re-crawling
visited = pybloom_live.ScalableBloomFilter(
    initial_capacity=10000,
    error_rate=0.001
)

urls_to_crawl = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page1",  # duplicate
    "https://example.com/page3",
]

for url in urls_to_crawl:
    if url in visited:
        print(f"SKIP (already crawled): {url}")
    else:
        visited.add(url)
        print(f"CRAWL: {url}")
        # crawl_page(url)
```

### Example 2: Set Operations

Combine filters using union and intersection:

```python
import pybloom_live

# Create two filters with same capacity and error rate
f1 = pybloom_live.BloomFilter(capacity=100, error_rate=0.01)
f2 = pybloom_live.BloomFilter(capacity=100, error_rate=0.01)

# Add elements
for i in range(20):
    f1.add(i)

for i in range(10, 30):
    f2.add(i)

# Union - elements in either filter
f_union = f1 | f2  # or f1.union(f2)
print(0 in f_union)   # True (in f1)
print(15 in f_union)  # True (in both)
print(25 in f_union)  # True (in f2)

# Intersection - elements likely in both filters
f_intersect = f1 & f2  # or f1.intersection(f2)
print(5 in f_intersect)   # False (only in f1)
print(15 in f_intersect)  # True (in both)
print(25 in f_intersect)  # False (only in f2)
```

### Example 3: Serialization

Save and load filters from disk:

```python
import pybloom_live

# Create and populate a filter
bf = pybloom_live.BloomFilter(capacity=10000, error_rate=0.001)
for i in range(1000):
    bf.add(f"user_{i}")

# Save to file
with open('bloom_filter.bin', 'wb') as f:
    bf.tofile(f)

# Load from file
with open('bloom_filter.bin', 'rb') as f:
    loaded_bf = pybloom_live.BloomFilter.fromfile(f)

# Verify
print("user_500" in loaded_bf)    # True
print("user_9999" in loaded_bf)   # False
```

### Example 4: Understanding Error Rates

```python
import pybloom_live

# Create a filter with 0.1% error rate
f = pybloom_live.BloomFilter(capacity=1000, error_rate=0.001)

# Add exactly capacity elements
for i in range(1000):
    f.add(i)

# Test false positive rate
false_positives = 0
tests = 10000
for i in range(1000, 1000 + tests):
    if i in f:  # These were never added
        false_positives += 1

actual_error_rate = false_positives / tests
print(f"Requested error rate: {f.error_rate}")
print(f"Actual error rate: {actual_error_rate:.4f}")
# Actual rate should be close to 0.001
```

## API Reference

### BloomFilter

**Constructor:**
```python
BloomFilter(capacity, error_rate=0.001)
```
- `capacity`: Maximum number of elements
- `error_rate`: False positive probability (0 < error_rate < 1)

**Methods:**
- `add(key, skip_check=False)`: Add element; returns `True` if element might already exist
- `__contains__(key)`: Check membership using `in` operator
- `__len__()`: Return count of elements added
- `copy()`: Create a copy of the filter
- `union(other)` / `__or__`: Combine with another filter (bitwise OR)
- `intersection(other)` / `__and__`: Intersect with another filter (bitwise AND)
- `tofile(f)`: Serialize to file object
- `fromfile(f, n=-1)`: Deserialize from file (class method)

**Properties:**
- `capacity`: Maximum capacity
- `error_rate`: Target false positive rate
- `count`: Number of elements added
- `num_bits`: Total bits in filter
- `num_slices`: Number of hash functions

### ScalableBloomFilter

**Constructor:**
```python
ScalableBloomFilter(initial_capacity=100, error_rate=0.001, mode=LARGE_SET_GROWTH)
```
- `initial_capacity`: Starting capacity
- `error_rate`: Target false positive probability
- `mode`: Growth mode (`SMALL_SET_GROWTH=2` or `LARGE_SET_GROWTH=4`)

**Methods:**
- `add(key)`: Add element to filter
- `__contains__(key)`: Check membership
- `__len__()`: Return total count across all internal filters
- `union(other)`: Combine with another scalable filter
- `tofile(f)` / `fromfile(f)`: Serialization support

**Properties:**
- `capacity`: Total capacity (sum of all internal filters)
- `count`: Total elements added
- `error_rate`: Target error rate
- `filters`: List of internal `BloomFilter` instances

## Hash Functions

The library automatically selects the optimal hash function based on the number of bits required:

| Bits Required | Hash Function | Type |
|--------------|---------------|------|
| ≤ 128 | xxh3_128 | Non-cryptographic (fastest) |
| 129-160 | SHA-1 | Cryptographic |
| 161-256 | SHA-256 | Cryptographic |
| 257-384 | SHA-384 | Cryptographic |
| > 384 | SHA-512 | Cryptographic |

## Algorithm Details

This implementation follows the Scalable Bloom Filter algorithm from:

> P. Almeida, C.Baquero, N. Preguiça, D. Hutchison, **Scalable Bloom Filters**, (GLOBECOM 2007), IEEE, 2007.

**Key optimizations:**
- **Tightening ratio of 0.9**: Provides better average space usage across a wide range of growth patterns
- **Exponential growth**: New filters are 2x or 4x larger than the previous one
- **Error rate tightening**: Each new filter has a tighter error probability to maintain overall target rate
- **Optimal bit calculation**: Uses `M ≈ n × |ln(P)| / (ln(2)²)` where n is capacity and P is error rate

## Breaking Changes in Version 4.x

Version 4.0.0 introduces significant performance improvements but is **not backward compatible** with earlier versions:

- **Hash function change**: Replaced MD5 with xxHash (xxh3_128) for 128-bit hashes
- **Performance**: Significantly faster hash computation (see [benchmarks](https://github.com/joseph-fox/python-bloomfilter/pull/38))
- **Incompatibility**: Files saved with versions < 4.0 cannot be loaded with version 4.x

**Migration:** If you have serialized filters from earlier versions, you must regenerate them using version 4.0+.

## Performance

Typical performance characteristics (results may vary by system):

- **Add operation**: ~500k-1M elements/second
- **Lookup operation**: ~700k-2M checks/second
- **Memory usage**: ~1.2 bits per element (with default 0.1% error rate)

## Python Version Support

- Python 3.6+
- Python 3.7+
- Python 3.8+
- Python 3.9+
- Python 3.10+
- Python 3.11+
- Python 3.12+

**Note:** Python 2 support has been dropped in version 4.0+. For Python 2 compatibility, use an earlier version.

## Development

We follow the [git branching model](http://nvie.com/posts/a-successful-git-branching-model/) described by Vincent Driessen.

**Running tests:**

```bash
pip install pytest
pytest pybloom_live/test_pybloom.py
```

## License

MIT License - see LICENSE.txt for details.

## Credits

- Original `pybloom` by Jay Baird
- Maintained and enhanced by the `pybloom_live` contributors
- Uses [xxHash](https://github.com/Cyan4973/xxHash) for high-performance hashing
