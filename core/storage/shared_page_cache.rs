use std::{
    fmt,
    hash::{Hash, Hasher},
    mem::size_of,
};

use lru::LruCache;
use rustc_hash::FxHasher;

use crate::sync::{atomic::AtomicU64, atomic::Ordering, Arc, Mutex};

const MAX_SHARDS: usize = 16;
const TARGET_BYTES_PER_SHARD: usize = 1024 * 1024;
const ENTRY_ALLOCATION_OVERHEAD: usize = 64;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum SharedPageVersion {
    Database {
        checkpoint_epoch: u32,
    },
    Wal {
        frame_id: u64,
        checkpoint_epoch: u32,
    },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SharedPageKey {
    namespace: u64,
    page_id: usize,
    page_size: u32,
    version: SharedPageVersion,
}

impl SharedPageKey {
    fn weight(self, value_len: usize) -> usize {
        value_len
            .saturating_add(size_of::<Self>())
            .saturating_add(size_of::<Arc<[u8]>>())
            .saturating_add(ENTRY_ALLOCATION_OVERHEAD)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SharedPageCacheStats {
    pub capacity_bytes: u64,
    pub resident_bytes: u64,
    pub entries: u64,
    pub hits: u64,
    pub misses: u64,
    pub insertions: u64,
    pub replacements: u64,
    pub evictions: u64,
    pub rejected: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SharedPageCacheLookup {
    Hit,
    Miss,
}

/// Optional low-cost observer for cache lookup outcomes.
///
/// Implementations should only update aggregate counters. This callback runs on the Pager read
/// path and must not perform blocking work, allocate high-cardinality labels or create spans.
pub trait SharedPageCacheObserver: Send + Sync {
    fn record_lookup(&self, outcome: SharedPageCacheLookup);
}

struct CacheShard {
    capacity_bytes: usize,
    resident_bytes: usize,
    entries: LruCache<SharedPageKey, Arc<[u8]>>,
    hits: u64,
    misses: u64,
    insertions: u64,
    replacements: u64,
    evictions: u64,
    rejected: u64,
}

impl CacheShard {
    fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            resident_bytes: 0,
            entries: LruCache::unbounded(),
            hits: 0,
            misses: 0,
            insertions: 0,
            replacements: 0,
            evictions: 0,
            rejected: 0,
        }
    }

    fn get(&mut self, key: &SharedPageKey) -> Option<Arc<[u8]>> {
        let value = self.entries.get(key).cloned();
        if value.is_some() {
            self.hits = self.hits.saturating_add(1);
        } else {
            self.misses = self.misses.saturating_add(1);
        }
        value
    }

    fn insert(&mut self, key: SharedPageKey, value: Arc<[u8]>) {
        let weight = key.weight(value.len());
        if weight > self.capacity_bytes {
            self.rejected = self.rejected.saturating_add(1);
            return;
        }

        if let Some(previous) = self.entries.pop(&key) {
            self.resident_bytes = self
                .resident_bytes
                .saturating_sub(key.weight(previous.len()));
            self.replacements = self.replacements.saturating_add(1);
        } else {
            self.insertions = self.insertions.saturating_add(1);
        }

        self.resident_bytes = self.resident_bytes.saturating_add(weight);
        self.entries.put(key, value);

        while self.resident_bytes > self.capacity_bytes {
            let Some((evicted_key, evicted_value)) = self.entries.pop_lru() else {
                break;
            };
            self.resident_bytes = self
                .resident_bytes
                .saturating_sub(evicted_key.weight(evicted_value.len()));
            self.evictions = self.evictions.saturating_add(1);
        }
    }

    fn add_stats(&self, stats: &mut SharedPageCacheStats) {
        stats.resident_bytes = stats
            .resident_bytes
            .saturating_add(self.resident_bytes as u64);
        stats.entries = stats.entries.saturating_add(self.entries.len() as u64);
        stats.hits = stats.hits.saturating_add(self.hits);
        stats.misses = stats.misses.saturating_add(self.misses);
        stats.insertions = stats.insertions.saturating_add(self.insertions);
        stats.replacements = stats.replacements.saturating_add(self.replacements);
        stats.evictions = stats.evictions.saturating_add(self.evictions);
        stats.rejected = stats.rejected.saturating_add(self.rejected);
    }
}

/// A byte-bounded process cache for immutable, clean database page images.
///
/// The cache deliberately stores bytes rather than [`super::pager::PageRef`] or
/// [`crate::Buffer`]. Each pager materializes a private buffer on hit, keeping
/// transaction-local dirty, spill, rollback and pin state isolated.
pub struct SharedPageCache {
    capacity_bytes: usize,
    shards: Box<[Mutex<CacheShard>]>,
    next_namespace: AtomicU64,
    observer: Option<Arc<dyn SharedPageCacheObserver>>,
}

impl fmt::Debug for SharedPageCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedPageCache")
            .field("stats", &self.stats())
            .finish()
    }
}

impl SharedPageCache {
    pub fn new(capacity_bytes: usize) -> Self {
        Self::with_optional_observer(capacity_bytes, None)
    }

    pub fn with_observer(
        capacity_bytes: usize,
        observer: Arc<dyn SharedPageCacheObserver>,
    ) -> Self {
        Self::with_optional_observer(capacity_bytes, Some(observer))
    }

    fn with_optional_observer(
        capacity_bytes: usize,
        observer: Option<Arc<dyn SharedPageCacheObserver>>,
    ) -> Self {
        let shard_count = if capacity_bytes == 0 {
            1
        } else {
            capacity_bytes
                .div_ceil(TARGET_BYTES_PER_SHARD)
                .clamp(1, MAX_SHARDS)
        };
        let base_capacity = capacity_bytes / shard_count;
        let remainder = capacity_bytes % shard_count;
        let shards = (0..shard_count)
            .map(|index| {
                let capacity = base_capacity + usize::from(index < remainder);
                Mutex::new(CacheShard::new(capacity))
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            capacity_bytes,
            shards,
            next_namespace: AtomicU64::new(1),
            observer,
        }
    }

    pub fn stats(&self) -> SharedPageCacheStats {
        let mut stats = SharedPageCacheStats {
            capacity_bytes: self.capacity_bytes as u64,
            ..SharedPageCacheStats::default()
        };
        for shard in &self.shards {
            shard.lock().add_stats(&mut stats);
        }
        stats
    }

    pub(crate) fn new_namespace(self: &Arc<Self>) -> SharedPageCacheNamespace {
        let id = self.next_namespace.fetch_add(1, Ordering::Relaxed);
        SharedPageCacheNamespace {
            cache: self.clone(),
            id,
        }
    }

    fn shard_index(&self, key: &SharedPageKey) -> usize {
        let mut hasher = FxHasher::default();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.shards.len()
    }

    fn get(&self, key: &SharedPageKey) -> Option<Arc<[u8]>> {
        let value = self.shards[self.shard_index(key)].lock().get(key);
        if let Some(observer) = self.observer.as_ref() {
            observer.record_lookup(if value.is_some() {
                SharedPageCacheLookup::Hit
            } else {
                SharedPageCacheLookup::Miss
            });
        }
        value
    }

    fn insert(&self, key: SharedPageKey, value: Arc<[u8]>) {
        self.shards[self.shard_index(&key)]
            .lock()
            .insert(key, value);
    }
}

#[derive(Clone)]
pub(crate) struct SharedPageCacheNamespace {
    cache: Arc<SharedPageCache>,
    id: u64,
}

impl fmt::Debug for SharedPageCacheNamespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedPageCacheNamespace")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl SharedPageCacheNamespace {
    pub(crate) fn get(
        &self,
        page_id: usize,
        page_size: u32,
        version: SharedPageVersion,
    ) -> Option<Arc<[u8]>> {
        self.cache.get(&SharedPageKey {
            namespace: self.id,
            page_id,
            page_size,
            version,
        })
    }

    pub(crate) fn publisher(
        &self,
        page_id: usize,
        page_size: u32,
        version: SharedPageVersion,
    ) -> SharedPageCachePublisher {
        SharedPageCachePublisher {
            namespace: self.clone(),
            key: SharedPageKey {
                namespace: self.id,
                page_id,
                page_size,
                version,
            },
        }
    }
}

#[derive(Clone)]
pub(crate) struct SharedPageCachePublisher {
    namespace: SharedPageCacheNamespace,
    key: SharedPageKey,
}

impl SharedPageCachePublisher {
    pub(crate) fn publish(&self, bytes: &[u8]) {
        if bytes.len() != self.key.page_size as usize {
            return;
        }
        self.namespace
            .cache
            .insert(self.key, Arc::<[u8]>::from(bytes));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicU64 as StdAtomicU64, Ordering as StdOrdering};

    #[cfg(feature = "fs")]
    use crate::{Connection, Database, PlatformIO};

    fn version(frame_id: u64) -> SharedPageVersion {
        SharedPageVersion::Wal {
            frame_id,
            checkpoint_epoch: 7,
        }
    }

    #[test]
    fn namespace_and_version_isolate_entries() {
        let cache = Arc::new(SharedPageCache::new(64 * 1024));
        let first = cache.new_namespace();
        let second = cache.new_namespace();
        first
            .publisher(3, 4096, version(11))
            .publish(&vec![1; 4096]);

        assert_eq!(first.get(3, 4096, version(11)).unwrap()[0], 1);
        assert!(first.get(3, 4096, version(12)).is_none());
        assert!(second.get(3, 4096, version(11)).is_none());
    }

    #[test]
    fn byte_capacity_evicts_and_never_exceeds_bound() {
        let per_entry = SharedPageKey {
            namespace: 1,
            page_id: 1,
            page_size: 4096,
            version: version(1),
        }
        .weight(4096);
        let cache = Arc::new(SharedPageCache::new(per_entry * 2));
        let namespace = cache.new_namespace();

        for page_id in 1..=3 {
            namespace
                .publisher(page_id, 4096, version(page_id as u64))
                .publish(&vec![page_id as u8; 4096]);
        }

        let stats = cache.stats();
        assert!(stats.resident_bytes <= stats.capacity_bytes);
        assert!(stats.entries <= 2);
        assert!(stats.evictions >= 1);
    }

    #[test]
    fn oversized_and_wrong_sized_values_are_not_cached() {
        let cache = Arc::new(SharedPageCache::new(1024));
        let namespace = cache.new_namespace();
        namespace
            .publisher(1, 4096, version(1))
            .publish(&vec![0; 4096]);
        namespace
            .publisher(2, 4096, version(1))
            .publish(&vec![0; 128]);

        assert!(namespace.get(1, 4096, version(1)).is_none());
        assert!(namespace.get(2, 4096, version(1)).is_none());
        assert_eq!(cache.stats().rejected, 1);
    }

    #[derive(Default)]
    struct LookupObserver {
        hits: StdAtomicU64,
        misses: StdAtomicU64,
    }

    impl SharedPageCacheObserver for LookupObserver {
        fn record_lookup(&self, outcome: SharedPageCacheLookup) {
            match outcome {
                SharedPageCacheLookup::Hit => {
                    self.hits.fetch_add(1, StdOrdering::Relaxed);
                }
                SharedPageCacheLookup::Miss => {
                    self.misses.fetch_add(1, StdOrdering::Relaxed);
                }
            }
        }
    }

    #[test]
    fn observer_receives_aggregate_lookup_outcomes() {
        let observer = Arc::new(LookupObserver::default());
        let cache = Arc::new(SharedPageCache::with_observer(64 * 1024, observer.clone()));
        let namespace = cache.new_namespace();

        assert!(namespace.get(1, 4096, version(1)).is_none());
        namespace
            .publisher(1, 4096, version(1))
            .publish(&vec![7; 4096]);
        assert!(namespace.get(1, 4096, version(1)).is_some());

        assert_eq!(observer.hits.load(StdOrdering::Relaxed), 1);
        assert_eq!(observer.misses.load(StdOrdering::Relaxed), 1);
    }

    #[cfg(feature = "fs")]
    fn database_with_cache() -> (tempfile::TempDir, Arc<Database>, Arc<SharedPageCache>) {
        database_with_cache_capacity(8 * 1024 * 1024)
    }

    #[cfg(feature = "fs")]
    fn database_with_cache_capacity(
        capacity_bytes: usize,
    ) -> (tempfile::TempDir, Arc<Database>, Arc<SharedPageCache>) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("shared-page-cache.db");
        let io = Arc::new(PlatformIO::new().unwrap());
        let database = Database::open_file(io, path.to_str().unwrap()).unwrap();
        let cache = Arc::new(SharedPageCache::new(capacity_bytes));
        database.set_shared_page_cache(Some(cache.clone()));
        (directory, database, cache)
    }

    #[cfg(feature = "fs")]
    fn seed_rows(connection: &Arc<Connection>, rows: usize) {
        connection
            .execute("CREATE TABLE test(id INTEGER PRIMARY KEY, value BLOB)")
            .unwrap();
        connection.execute("BEGIN").unwrap();
        for _ in 0..rows {
            connection
                .execute("INSERT INTO test(value) VALUES (zeroblob(1024))")
                .unwrap();
        }
        connection.execute("COMMIT").unwrap();
    }

    #[cfg(feature = "fs")]
    fn row_count(connection: &Arc<Connection>) -> i64 {
        let mut statement = connection.prepare("SELECT count(*) FROM test").unwrap();
        let mut count = None;
        statement
            .run_with_row_callback(|row| {
                count = Some(row.get(0).unwrap());
                Ok(())
            })
            .unwrap();
        count.unwrap()
    }

    #[cfg(feature = "fs")]
    #[test]
    fn fresh_connections_reuse_clean_pages_without_sharing_pager_state() {
        let (_directory, database, cache) = database_with_cache();
        let writer = database.connect().unwrap();
        seed_rows(&writer, 256);

        let first_reader = database.connect().unwrap();
        assert_eq!(row_count(&first_reader), 256);
        let after_first = cache.stats();

        let second_reader = database.connect().unwrap();
        assert_eq!(row_count(&second_reader), 256);
        let after_second = cache.stats();

        assert!(after_second.hits > after_first.hits);
        assert_eq!(after_second.misses, after_first.misses);
        assert!(after_second.resident_bytes <= after_second.capacity_bytes);
    }

    #[cfg(feature = "fs")]
    #[test]
    fn rollback_commit_and_old_snapshot_never_reuse_stale_pages() {
        let (_directory, database, _cache) = database_with_cache();
        let writer = database.connect().unwrap();
        seed_rows(&writer, 64);

        let old_snapshot = database.connect().unwrap();
        old_snapshot.execute("BEGIN").unwrap();
        assert_eq!(row_count(&old_snapshot), 64);

        writer.execute("BEGIN").unwrap();
        writer
            .execute("INSERT INTO test(value) VALUES (zeroblob(1024))")
            .unwrap();
        writer.execute("ROLLBACK").unwrap();
        assert_eq!(row_count(&database.connect().unwrap()), 64);

        writer
            .execute("INSERT INTO test(value) VALUES (zeroblob(1024))")
            .unwrap();
        assert_eq!(row_count(&old_snapshot), 64);
        assert_eq!(row_count(&database.connect().unwrap()), 65);
        old_snapshot.execute("COMMIT").unwrap();
    }

    #[cfg(feature = "fs")]
    #[test]
    fn checkpoint_epoch_prevents_pre_checkpoint_page_reuse() {
        let (_directory, database, cache) = database_with_cache();
        let writer = database.connect().unwrap();
        seed_rows(&writer, 128);

        assert_eq!(row_count(&database.connect().unwrap()), 128);
        assert_eq!(row_count(&database.connect().unwrap()), 128);
        let before_checkpoint = cache.stats();

        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").unwrap();
        assert_eq!(row_count(&database.connect().unwrap()), 128);
        let after_checkpoint = cache.stats();

        assert!(after_checkpoint.misses > before_checkpoint.misses);
        assert!(after_checkpoint.resident_bytes <= after_checkpoint.capacity_bytes);
    }

    #[cfg(feature = "fs")]
    #[test]
    fn contested_concurrent_insertions_are_safe_and_bounded() {
        let (_directory, database, cache) = database_with_cache();
        let writer = database.connect().unwrap();
        seed_rows(&writer, 256);

        let handles = (0..8)
            .map(|_| {
                let database = database.clone();
                std::thread::spawn(move || row_count(&database.connect().unwrap()))
            })
            .collect::<Vec<_>>();

        for handle in handles {
            assert_eq!(handle.join().unwrap(), 256);
        }
        let stats = cache.stats();
        assert!(stats.resident_bytes <= stats.capacity_bytes);
        assert!(stats.entries > 0);
    }

    #[cfg(feature = "fs")]
    #[test]
    fn eviction_falls_back_and_repopulates_without_changing_results() {
        let page_weight = SharedPageKey {
            namespace: 1,
            page_id: 1,
            page_size: 4096,
            version: version(1),
        }
        .weight(4096);
        let (_directory, database, cache) = database_with_cache_capacity(page_weight * 2);
        let writer = database.connect().unwrap();
        seed_rows(&writer, 128);

        assert_eq!(row_count(&database.connect().unwrap()), 128);
        let after_first = cache.stats();
        assert!(after_first.evictions > 0);

        assert_eq!(row_count(&database.connect().unwrap()), 128);
        let after_second = cache.stats();
        assert!(after_second.misses > after_first.misses);
        assert!(after_second.insertions > after_first.insertions);
        assert!(after_second.resident_bytes <= after_second.capacity_bytes);
    }
}
