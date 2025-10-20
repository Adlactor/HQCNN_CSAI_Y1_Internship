from collections import OrderedDict

class LRUCache:
    """
    LRUCache: Least Recently Used (LRU) cache implementation using OrderedDict.
    Evicts the oldest accessed items when capacity is exceeded and tracks cache hits/misses.
    """
    def __init__(self, capacity=50000):
        """
        Initialize the LRUCache.

        Args:
            capacity (int): Maximum number of entries the cache can hold.
        """
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        """
        Retrieve a value from the cache.

        If the key exists, count as hit, move entry to the end (most recently used), and return its value.
        If the key does not exist, count as miss and return None.

        Args:
            key: Hashable key to look up.

        Returns:
            The stored value if present; otherwise, None.
        """
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        """
        Insert or update a cache entry.

        If key already exists, update its value and mark as recently used.
        If inserting a new key causes cache size to exceed capacity, evict the least recently used item.

        Args:
            key: Hashable key for the cache entry.
            value: Value to store in the cache.
        """
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def reset_counts(self):
        """
        Reset the hit and miss counters to zero.

        Useful for benchmarking or periodic measurement resets.
        """
        self.hits = 0
        self.misses = 0
