use std::{
    array,
    cmp::max,
    fmt,
    hash::BuildHasherDefault,
    hash::{Hash, Hasher},
    mem::{align_of, size_of},
    num::NonZeroUsize,
    ops::Deref,
};

use lru::LruCache;
use rustc_hash::FxHasher;

use crate::sync::{
    atomic::{AtomicU64, AtomicU8, Ordering},
    Arc, Mutex,
};

const MAX_SHARDS: usize = 16;
const TARGET_BYTES_PER_SHARD: usize = 1024 * 1024;
const ENTRY_ALLOCATION_OVERHEAD: usize = 64;

const RESERVATION_STATE_RESERVED: u8 = 0;
const RESERVATION_STATE_RESIDENT: u8 = 1;
const RESERVATION_STATE_RETIRED: u8 = 2;

/// The category of a managed shared-page cache charge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SharedPageCacheChargeKind {
    /// Fixed cache, shard, LRU node-capacity and index growth-peak space.
    MetadataCapacity,
    /// One immutable page payload and its ownership envelope.
    Page,
}

/// A conservative managed-memory charge, not an allocator or RSS measurement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SharedPageCacheCharge {
    pub kind: SharedPageCacheChargeKind,
    pub bytes: usize,
}

/// The current lifetime classification of a managed charge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SharedPageCacheChargeState {
    Reserved,
    Resident,
    Retired,
}

/// Result of an opt-in shared-page reservation attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SharedPageCacheReservation {
    /// The callback charged the requested allocation.
    Reserved,
    /// The callback has insufficient capacity; bounded local reclaim may make progress.
    CapacityDenied,
    /// The callback cannot decide now; insertion is skipped without reclaiming entries.
    Deferred,
}

/// Admission and lifetime callbacks for opt-in shared-page accounting.
///
/// Implementations must be short, synchronous, non-allocating and must not panic. They must not
/// perform I/O, wait, or re-enter an engine cache. The accounting object must be an independent
/// ledger handle that neither owns nor calls a [`SharedPageCache`], preventing an ownership cycle
/// or reverse lock order. Calls may occur while a cache shard is locked, and a final release may
/// run on any thread that drops the last page handle.
pub trait SharedPageCacheAccounting: Send + Sync {
    fn try_reserve(&self, charge: SharedPageCacheCharge) -> bool;

    /// Attempts a reservation while distinguishing capacity from transient suppression.
    ///
    /// For compatibility, the default maps a `false` boolean reservation to capacity pressure.
    /// Implementations that can be transiently unable to reserve should override this method and
    /// return [`SharedPageCacheReservation::Deferred`]. A deferred page is bypassed without cache
    /// mutation; a later publisher call may retry, but the cache does not schedule retries itself.
    fn try_reserve_classified(&self, charge: SharedPageCacheCharge) -> SharedPageCacheReservation {
        if self.try_reserve(charge) {
            SharedPageCacheReservation::Reserved
        } else {
            SharedPageCacheReservation::CapacityDenied
        }
    }

    fn transition(
        &self,
        _charge: SharedPageCacheCharge,
        _from: SharedPageCacheChargeState,
        _to: SharedPageCacheChargeState,
    ) {
    }

    fn release(&self, charge: SharedPageCacheCharge, state: SharedPageCacheChargeState);
}

/// Failure to create an accounted shared-page cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SharedPageCacheCreateError {
    ChargeOverflow,
    AdmissionDenied,
}

struct Reservation {
    accounting: Arc<dyn SharedPageCacheAccounting>,
    charge: SharedPageCacheCharge,
    state: AtomicU8,
}

enum ReservationAttempt {
    Reserved(Reservation),
    CapacityDenied,
    Deferred,
}

impl Reservation {
    fn try_new(
        accounting: Arc<dyn SharedPageCacheAccounting>,
        charge: SharedPageCacheCharge,
    ) -> Option<Self> {
        accounting.try_reserve(charge).then(|| Self {
            accounting,
            charge,
            state: AtomicU8::new(RESERVATION_STATE_RESERVED),
        })
    }

    fn try_new_classified(
        accounting: Arc<dyn SharedPageCacheAccounting>,
        charge: SharedPageCacheCharge,
    ) -> ReservationAttempt {
        match accounting.try_reserve_classified(charge) {
            SharedPageCacheReservation::Reserved => ReservationAttempt::Reserved(Self {
                accounting,
                charge,
                state: AtomicU8::new(RESERVATION_STATE_RESERVED),
            }),
            SharedPageCacheReservation::CapacityDenied => ReservationAttempt::CapacityDenied,
            SharedPageCacheReservation::Deferred => ReservationAttempt::Deferred,
        }
    }

    fn transition(&self, from: SharedPageCacheChargeState, to: SharedPageCacheChargeState) {
        let previous = self.state.compare_exchange(
            reservation_state_code(from),
            reservation_state_code(to),
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        assert!(
            previous.is_ok(),
            "shared-page charge must follow its single-owner state transition"
        );
        self.accounting.transition(self.charge, from, to);
    }

    fn make_resident(&self) {
        self.transition(
            SharedPageCacheChargeState::Reserved,
            SharedPageCacheChargeState::Resident,
        );
    }

    fn retire(&self) {
        self.transition(
            SharedPageCacheChargeState::Resident,
            SharedPageCacheChargeState::Retired,
        );
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.accounting.release(
            self.charge,
            reservation_state_from_code(self.state.load(Ordering::Acquire)),
        );
    }
}

fn reservation_state_code(state: SharedPageCacheChargeState) -> u8 {
    match state {
        SharedPageCacheChargeState::Reserved => RESERVATION_STATE_RESERVED,
        SharedPageCacheChargeState::Resident => RESERVATION_STATE_RESIDENT,
        SharedPageCacheChargeState::Retired => RESERVATION_STATE_RETIRED,
    }
}

fn reservation_state_from_code(state: u8) -> SharedPageCacheChargeState {
    match state {
        RESERVATION_STATE_RESERVED => SharedPageCacheChargeState::Reserved,
        RESERVATION_STATE_RESIDENT => SharedPageCacheChargeState::Resident,
        RESERVATION_STATE_RETIRED => SharedPageCacheChargeState::Retired,
        _ => unreachable!("reservation state is only written by Reservation::transition"),
    }
}

struct SharedAllocation {
    // Field order is intentional: page bytes are destroyed before their reservation is released.
    bytes: Box<[u8]>,
    reservation: Reservation,
}

#[derive(Clone)]
pub(crate) struct SharedPageBytes(SharedPageBytesInner);

#[derive(Clone)]
enum SharedPageBytesInner {
    Legacy(Arc<[u8]>),
    Accounted(Arc<SharedAllocation>),
}

impl SharedPageBytes {
    fn legacy(bytes: Arc<[u8]>) -> Self {
        Self(SharedPageBytesInner::Legacy(bytes))
    }

    fn accounted(allocation: Arc<SharedAllocation>) -> Self {
        Self(SharedPageBytesInner::Accounted(allocation))
    }
}

impl Deref for SharedPageBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match &self.0 {
            SharedPageBytesInner::Legacy(bytes) => bytes,
            SharedPageBytesInner::Accounted(allocation) => &allocation.bytes,
        }
    }
}

impl AsRef<[u8]> for SharedPageBytes {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

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

    fn checked_weight(self, value_len: usize) -> Option<usize> {
        value_len
            .checked_add(size_of::<Self>())?
            .checked_add(size_of::<Arc<[u8]>>())?
            .checked_add(ENTRY_ALLOCATION_OVERHEAD)
    }
}

fn checked_round_up(value: usize, alignment: usize) -> Option<usize> {
    debug_assert!(alignment.is_power_of_two());
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
}

fn initial_bucket_count(max_entries: usize) -> Option<usize> {
    if max_entries < 4 {
        Some(4)
    } else if max_entries < 8 {
        Some(8)
    } else if max_entries < 15 {
        Some(16)
    } else {
        max_entries
            .checked_mul(8)?
            .checked_div(7)?
            .checked_next_power_of_two()
    }
}

fn table_layout_envelope(bucket_count: usize) -> Option<usize> {
    let pointer_size = size_of::<usize>();
    let entries = bucket_count.checked_mul(pointer_size.checked_mul(2)?)?;
    checked_round_up(entries, 16)?
        .checked_add(bucket_count)?
        .checked_add(16)
}

fn node_layout_envelope() -> Option<usize> {
    let pointer_size = size_of::<usize>();
    let alignment = max(
        max(align_of::<SharedPageKey>(), align_of::<SharedPageBytes>()),
        pointer_size,
    );
    checked_round_up(size_of::<SharedPageKey>(), alignment)?
        .checked_add(checked_round_up(size_of::<SharedPageBytes>(), alignment)?)?
        .checked_add(checked_round_up(pointer_size, alignment)?.checked_mul(2)?)
}

fn metadata_charge_bytes(
    capacity_bytes: usize,
    max_entries_per_shard: NonZeroUsize,
) -> Option<usize> {
    let pointer_size = size_of::<usize>();
    if !matches!(pointer_size, 4 | 8) {
        return None;
    }

    let shard_count = shard_count(capacity_bytes);
    let initial_buckets = initial_bucket_count(max_entries_per_shard.get())?;
    let grown_buckets = initial_buckets.checked_mul(2)?;
    let per_shard = table_layout_envelope(initial_buckets)?
        .checked_add(table_layout_envelope(grown_buckets)?)?
        .checked_add(
            max_entries_per_shard
                .get()
                .checked_add(2)?
                .checked_mul(node_layout_envelope()?)?,
        )?;
    let index_and_nodes = per_shard.checked_mul(shard_count)?;

    let fixed_shards = checked_round_up(
        size_of::<[AccountedShardSlot; MAX_SHARDS]>(),
        align_of::<AccountedShardSlot>(),
    )?;
    let fixed_shards_source_and_destination = fixed_shards.checked_mul(2)?;
    let cache_alignment = max(align_of::<SharedPageCache>(), pointer_size);
    let cache_and_arc = checked_round_up(
        size_of::<SharedPageCache>().checked_add(pointer_size.checked_mul(4)?)?,
        cache_alignment,
    )?;

    index_and_nodes
        .checked_add(fixed_shards_source_and_destination)?
        .checked_add(cache_and_arc)
}

fn page_charge_bytes(page_len: usize) -> Option<usize> {
    let pointer_size = size_of::<usize>();
    if !matches!(pointer_size, 4 | 8) {
        return None;
    }
    let bytes = checked_round_up(page_len, pointer_size)?;
    let allocation_alignment = max(align_of::<SharedAllocation>(), pointer_size);
    let owner = checked_round_up(
        size_of::<SharedAllocation>().checked_add(pointer_size.checked_mul(4)?)?,
        allocation_alignment,
    )?;
    bytes.checked_add(owner)
}

fn shard_count(capacity_bytes: usize) -> usize {
    if capacity_bytes == 0 {
        1
    } else {
        capacity_bytes
            .div_ceil(TARGET_BYTES_PER_SHARD)
            .clamp(1, MAX_SHARDS)
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum OwnedAllocationKind {
    ShardIndex,
    ShardBacking,
    PageBytes,
    PageOwner,
    IndexNode,
}

#[cfg(test)]
impl OwnedAllocationKind {
    fn index(self) -> usize {
        match self {
            Self::ShardIndex => 0,
            Self::ShardBacking => 1,
            Self::PageBytes => 2,
            Self::PageOwner => 3,
            Self::IndexNode => 4,
        }
    }
}

#[cfg(test)]
std::thread_local! {
    static OWNED_ALLOCATION_COUNTS: std::cell::Cell<Option<[usize; 5]>> = const {
        std::cell::Cell::new(None)
    };
}

#[cfg(test)]
fn record_owned_allocation(kind: OwnedAllocationKind) {
    OWNED_ALLOCATION_COUNTS.with(|counts| {
        let Some(mut current) = counts.get() else {
            return;
        };
        current[kind.index()] += 1;
        counts.set(Some(current));
    });
}

fn allocate_page_bytes(bytes: &[u8]) -> Box<[u8]> {
    #[cfg(test)]
    record_owned_allocation(OwnedAllocationKind::PageBytes);
    Box::<[u8]>::from(bytes)
}

fn allocate_page_owner(bytes: Box<[u8]>, reservation: Reservation) -> Arc<SharedAllocation> {
    #[cfg(test)]
    record_owned_allocation(OwnedAllocationKind::PageOwner);
    Arc::new(SharedAllocation { bytes, reservation })
}

fn allocate_shard_backing(
    shards: [AccountedShardSlot; MAX_SHARDS],
) -> Box<[AccountedShardSlot; MAX_SHARDS]> {
    #[cfg(test)]
    record_owned_allocation(OwnedAllocationKind::ShardBacking);
    Box::new(shards)
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

type AccountedEntries =
    LruCache<SharedPageKey, Arc<SharedAllocation>, BuildHasherDefault<FxHasher>>;

struct AccountedCacheShard {
    capacity_bytes: usize,
    resident_bytes: usize,
    max_entries: usize,
    entries: AccountedEntries,
    hits: u64,
    misses: u64,
    insertions: u64,
    replacements: u64,
    evictions: u64,
    rejected: u64,
}

impl AccountedCacheShard {
    fn new(capacity_bytes: usize, max_entries: NonZeroUsize) -> Self {
        #[cfg(test)]
        record_owned_allocation(OwnedAllocationKind::ShardIndex);
        Self {
            capacity_bytes,
            resident_bytes: 0,
            max_entries: max_entries.get(),
            entries: LruCache::with_hasher(max_entries, BuildHasherDefault::default()),
            hits: 0,
            misses: 0,
            insertions: 0,
            replacements: 0,
            evictions: 0,
            rejected: 0,
        }
    }

    fn get(&mut self, key: &SharedPageKey) -> Option<Arc<SharedAllocation>> {
        let value = self.entries.get(key).cloned();
        if value.is_some() {
            self.hits = self.hits.saturating_add(1);
        } else {
            self.misses = self.misses.saturating_add(1);
        }
        value
    }

    fn retire_entry(&mut self, key: SharedPageKey, value: Arc<SharedAllocation>, eviction: bool) {
        self.resident_bytes = self
            .resident_bytes
            .checked_sub(
                key.checked_weight(value.bytes.len())
                    .expect("cached page weight was checked before insertion"),
            )
            .expect("resident accounting includes every cached entry");
        value.reservation.retire();
        if eviction {
            self.evictions = self.evictions.saturating_add(1);
        }
    }

    fn pop_lru_and_retire(&mut self) -> bool {
        let Some((key, value)) = self.entries.pop_lru() else {
            return false;
        };
        self.retire_entry(key, value, true);
        true
    }

    fn reclaim_unreferenced(&mut self, target_bytes: usize) -> usize {
        let entries_to_scan = self.entries.len();
        let mut reclaimed_bytes = 0_usize;
        for _ in 0..entries_to_scan {
            if reclaimed_bytes >= target_bytes {
                break;
            }
            let Some((key, value)) = self.entries.peek_lru() else {
                break;
            };
            let key = *key;
            if Arc::strong_count(value) == 1 {
                let Some((removed_key, removed_value)) = self.entries.pop_lru() else {
                    break;
                };
                let charge_bytes = removed_value.reservation.charge.bytes;
                self.retire_entry(removed_key, removed_value, true);
                reclaimed_bytes = reclaimed_bytes.saturating_add(charge_bytes);
            } else {
                // Advance past an externally referenced allocation without allocating an
                // inventory snapshot. A complete scan restores the relative order of survivors.
                self.entries.promote(&key);
            }
        }
        reclaimed_bytes
    }

    fn insert(
        &mut self,
        key: SharedPageKey,
        bytes: &[u8],
        accounting: &Arc<dyn SharedPageCacheAccounting>,
    ) {
        let Some(resident_weight) = key.checked_weight(bytes.len()) else {
            self.rejected = self.rejected.saturating_add(1);
            return;
        };
        let Some(charge_bytes) = page_charge_bytes(bytes.len()) else {
            self.rejected = self.rejected.saturating_add(1);
            return;
        };
        if resident_weight > self.capacity_bytes {
            self.rejected = self.rejected.saturating_add(1);
            return;
        }

        let charge = SharedPageCacheCharge {
            kind: SharedPageCacheChargeKind::Page,
            bytes: charge_bytes,
        };
        let mut reservation = match Reservation::try_new_classified(accounting.clone(), charge) {
            ReservationAttempt::Reserved(reservation) => Some(reservation),
            ReservationAttempt::CapacityDenied => None,
            ReservationAttempt::Deferred => {
                self.rejected = self.rejected.saturating_add(1);
                return;
            }
        };
        let mut replacing = false;

        if reservation.is_none() {
            let mut reclaim_attempts = self.entries.len().min(self.max_entries);
            loop {
                if reclaim_attempts == 0 {
                    self.rejected = self.rejected.saturating_add(1);
                    return;
                }
                let removed = if let Some(previous) = self.entries.pop(&key) {
                    self.retire_entry(key, previous, false);
                    self.replacements = self.replacements.saturating_add(1);
                    replacing = true;
                    true
                } else {
                    self.pop_lru_and_retire()
                };
                if !removed {
                    self.rejected = self.rejected.saturating_add(1);
                    return;
                }
                reclaim_attempts -= 1;
                match Reservation::try_new_classified(accounting.clone(), charge) {
                    ReservationAttempt::Reserved(admitted) => {
                        reservation = Some(admitted);
                        break;
                    }
                    ReservationAttempt::CapacityDenied => {}
                    ReservationAttempt::Deferred => {
                        self.rejected = self.rejected.saturating_add(1);
                        return;
                    }
                }
            }
        }

        if let Some(previous) = self.entries.pop(&key) {
            self.retire_entry(key, previous, false);
            self.replacements = self.replacements.saturating_add(1);
            replacing = true;
        }
        let resident_limit_before_insert = self.capacity_bytes - resident_weight;
        while self.resident_bytes > resident_limit_before_insert
            || self.entries.len() >= self.max_entries
        {
            let removed = self.pop_lru_and_retire();
            assert!(
                removed,
                "a non-empty accounted shard must provide an LRU entry"
            );
        }
        let reservation = reservation.expect("accounted insertion has a charged reservation");

        let owned_bytes = allocate_page_bytes(bytes);
        let allocation = allocate_page_owner(owned_bytes, reservation);

        let previous = self.put_new_entry(key, allocation);
        assert!(
            previous.is_none(),
            "replacement and capacity eviction happen before accounted insertion"
        );
        let inserted = self
            .entries
            .peek(&key)
            .expect("the inserted accounted page remains in the bounded LRU");
        inserted.reservation.make_resident();
        self.resident_bytes = self
            .resident_bytes
            .checked_add(resident_weight)
            .expect("resident weight was checked against shard capacity");
        if !replacing {
            self.insertions = self.insertions.saturating_add(1);
        }
    }

    fn put_new_entry(
        &mut self,
        key: SharedPageKey,
        allocation: Arc<SharedAllocation>,
    ) -> Option<Arc<SharedAllocation>> {
        #[cfg(test)]
        record_owned_allocation(OwnedAllocationKind::IndexNode);
        self.entries.put(key, allocation)
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

impl Drop for AccountedCacheShard {
    fn drop(&mut self) {
        while let Some((key, value)) = self.entries.pop_lru() {
            self.retire_entry(key, value, false);
        }
    }
}

enum AccountedShardSlot {
    Inactive,
    Active(Mutex<AccountedCacheShard>),
}

enum CacheShards {
    Legacy(Box<[Mutex<CacheShard>]>),
    Accounted(Box<[AccountedShardSlot; MAX_SHARDS]>),
}

/// A byte-bounded process cache for immutable, clean database page images.
///
/// The cache deliberately stores bytes rather than [`super::pager::PageRef`] or
/// [`crate::Buffer`]. Each pager materializes a private buffer on hit, keeping
/// transaction-local dirty, spill, rollback and pin state isolated.
pub struct SharedPageCache {
    capacity_bytes: usize,
    // Accounted shard backing drops before the metadata reservation, so cached pages retire and
    // the bounded index is destroyed before its fixed-capacity charge is released.
    shards: CacheShards,
    metadata_reservation: Option<Reservation>,
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
        let shard_count = shard_count(capacity_bytes);
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
            shards: CacheShards::Legacy(shards),
            metadata_reservation: None,
            next_namespace: AtomicU64::new(1),
            observer,
        }
    }

    /// Creates an opt-in cache with pre-allocation admission and exact page-owner lifetime hooks.
    ///
    /// `capacity_bytes` remains the resident-byte limit used by [`SharedPageCacheStats`].
    /// `max_entries_per_shard` independently bounds each shard's retained LRU/index population.
    /// Accounted page admission occurs before replacement or capacity eviction, so a successful
    /// replacement may transiently charge both old and new allocations until the old page retires.
    /// The legacy unaccounted constructor and insertion order are unchanged.
    pub fn try_with_accounting(
        capacity_bytes: usize,
        max_entries_per_shard: NonZeroUsize,
        accounting: Arc<dyn SharedPageCacheAccounting>,
        observer: Option<Arc<dyn SharedPageCacheObserver>>,
    ) -> Result<Self, SharedPageCacheCreateError> {
        let metadata_bytes = metadata_charge_bytes(capacity_bytes, max_entries_per_shard)
            .ok_or(SharedPageCacheCreateError::ChargeOverflow)?;
        let metadata_reservation = Reservation::try_new(
            accounting,
            SharedPageCacheCharge {
                kind: SharedPageCacheChargeKind::MetadataCapacity,
                bytes: metadata_bytes,
            },
        )
        .ok_or(SharedPageCacheCreateError::AdmissionDenied)?;

        let active_shards = shard_count(capacity_bytes);
        let base_capacity = capacity_bytes / active_shards;
        let remainder = capacity_bytes % active_shards;
        let shards = array::from_fn(|index| {
            if index < active_shards {
                let capacity = base_capacity + usize::from(index < remainder);
                AccountedShardSlot::Active(Mutex::new(AccountedCacheShard::new(
                    capacity,
                    max_entries_per_shard,
                )))
            } else {
                AccountedShardSlot::Inactive
            }
        });
        let shards = allocate_shard_backing(shards);

        Ok(Self {
            capacity_bytes,
            shards: CacheShards::Accounted(shards),
            metadata_reservation: Some(metadata_reservation),
            next_namespace: AtomicU64::new(1),
            observer,
        })
    }

    pub fn stats(&self) -> SharedPageCacheStats {
        let mut stats = SharedPageCacheStats {
            capacity_bytes: self.capacity_bytes as u64,
            ..SharedPageCacheStats::default()
        };
        match &self.shards {
            CacheShards::Legacy(shards) => {
                for shard in shards {
                    shard.lock().add_stats(&mut stats);
                }
            }
            CacheShards::Accounted(shards) => {
                for slot in shards.iter().take(self.active_shard_count()) {
                    let AccountedShardSlot::Active(shard) = slot else {
                        unreachable!("active accounted shard range contains only active slots");
                    };
                    shard.lock().add_stats(&mut stats);
                }
            }
        }
        stats
    }

    /// Retires accounted clean-page allocations that have no external reader.
    ///
    /// At most the entries present at the start of each bounded shard scan are inspected. The
    /// method performs no I/O or payload allocation, and legacy caches are unchanged. Returned
    /// bytes are the page allocation charges whose final cache-owned references were released;
    /// callers must still retry their own admission instead of treating this value as credit.
    pub fn reclaim_unreferenced(&self, target_bytes: usize) -> usize {
        if target_bytes == 0 {
            return 0;
        }
        let CacheShards::Accounted(shards) = &self.shards else {
            return 0;
        };
        let mut reclaimed_bytes = 0_usize;
        for slot in shards.iter().take(self.active_shard_count()) {
            if reclaimed_bytes >= target_bytes {
                break;
            }
            let AccountedShardSlot::Active(shard) = slot else {
                unreachable!("active accounted shard range contains only active slots");
            };
            let remaining = target_bytes.saturating_sub(reclaimed_bytes);
            reclaimed_bytes =
                reclaimed_bytes.saturating_add(shard.lock().reclaim_unreferenced(remaining));
        }
        reclaimed_bytes
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
        (hasher.finish() as usize) % self.active_shard_count()
    }

    fn active_shard_count(&self) -> usize {
        match &self.shards {
            CacheShards::Legacy(shards) => shards.len(),
            CacheShards::Accounted(_) => shard_count(self.capacity_bytes),
        }
    }

    fn get(&self, key: &SharedPageKey) -> Option<SharedPageBytes> {
        let shard_index = self.shard_index(key);
        let value = match &self.shards {
            CacheShards::Legacy(shards) => shards[shard_index]
                .lock()
                .get(key)
                .map(SharedPageBytes::legacy),
            CacheShards::Accounted(shards) => {
                let AccountedShardSlot::Active(shard) = &shards[shard_index] else {
                    unreachable!("hashed shard index is always active");
                };
                shard.lock().get(key).map(SharedPageBytes::accounted)
            }
        };
        if let Some(observer) = self.observer.as_ref() {
            observer.record_lookup(if value.is_some() {
                SharedPageCacheLookup::Hit
            } else {
                SharedPageCacheLookup::Miss
            });
        }
        value
    }

    fn publish(&self, key: SharedPageKey, bytes: &[u8]) {
        let shard_index = self.shard_index(&key);
        match &self.shards {
            CacheShards::Legacy(shards) => {
                shards[shard_index]
                    .lock()
                    .insert(key, Arc::<[u8]>::from(bytes));
            }
            CacheShards::Accounted(shards) => {
                let AccountedShardSlot::Active(shard) = &shards[shard_index] else {
                    unreachable!("hashed shard index is always active");
                };
                let reservation = self
                    .metadata_reservation
                    .as_ref()
                    .expect("accounted shards retain their metadata reservation");
                shard.lock().insert(key, bytes, &reservation.accounting);
            }
        }
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
    ) -> Option<SharedPageBytes> {
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
        self.namespace.cache.publish(self.key, bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::{
        atomic::{
            AtomicBool as StdAtomicBool, AtomicU64 as StdAtomicU64, AtomicUsize as StdAtomicUsize,
            Ordering as StdOrdering,
        },
        mpsc, Barrier,
    };

    #[cfg(feature = "fs")]
    use crate::{Connection, Database, PlatformIO};

    fn version(frame_id: u64) -> SharedPageVersion {
        SharedPageVersion::Wal {
            frame_id,
            checkpoint_epoch: 7,
        }
    }

    #[derive(Default)]
    struct TestAccounting {
        limit: StdAtomicUsize,
        used: StdAtomicUsize,
        peak: StdAtomicUsize,
        reserved: StdAtomicUsize,
        resident: StdAtomicUsize,
        retired: StdAtomicUsize,
        reserve_calls: StdAtomicUsize,
        transition_calls: StdAtomicUsize,
        release_calls: StdAtomicUsize,
        metadata_release_calls: StdAtomicUsize,
        page_release_calls: StdAtomicUsize,
        deny_metadata: StdAtomicBool,
        deny_pages: StdAtomicBool,
        defer_pages: StdAtomicBool,
        capacity_then_defer_pages: StdAtomicUsize,
    }

    impl TestAccounting {
        fn with_limit(limit: usize) -> Self {
            Self {
                limit: StdAtomicUsize::new(limit),
                ..Self::default()
            }
        }

        fn state_bytes(&self, state: SharedPageCacheChargeState) -> &StdAtomicUsize {
            match state {
                SharedPageCacheChargeState::Reserved => &self.reserved,
                SharedPageCacheChargeState::Resident => &self.resident,
                SharedPageCacheChargeState::Retired => &self.retired,
            }
        }

        fn update_peak(&self, candidate: usize) {
            let mut peak = self.peak.load(StdOrdering::Relaxed);
            while candidate > peak {
                match self.peak.compare_exchange_weak(
                    peak,
                    candidate,
                    StdOrdering::Relaxed,
                    StdOrdering::Relaxed,
                ) {
                    Ok(_) => return,
                    Err(observed) => peak = observed,
                }
            }
        }
    }

    impl SharedPageCacheAccounting for TestAccounting {
        fn try_reserve(&self, charge: SharedPageCacheCharge) -> bool {
            self.reserve_calls.fetch_add(1, StdOrdering::Relaxed);
            let denied = match charge.kind {
                SharedPageCacheChargeKind::MetadataCapacity => {
                    self.deny_metadata.load(StdOrdering::Relaxed)
                }
                SharedPageCacheChargeKind::Page => self.deny_pages.load(StdOrdering::Relaxed),
            };
            if denied {
                return false;
            }

            let mut used = self.used.load(StdOrdering::Relaxed);
            loop {
                let Some(candidate) = used.checked_add(charge.bytes) else {
                    return false;
                };
                if candidate > self.limit.load(StdOrdering::Relaxed) {
                    return false;
                }
                match self.used.compare_exchange_weak(
                    used,
                    candidate,
                    StdOrdering::AcqRel,
                    StdOrdering::Relaxed,
                ) {
                    Ok(_) => {
                        self.reserved.fetch_add(charge.bytes, StdOrdering::Relaxed);
                        self.update_peak(candidate);
                        return true;
                    }
                    Err(observed) => used = observed,
                }
            }
        }

        fn try_reserve_classified(
            &self,
            charge: SharedPageCacheCharge,
        ) -> SharedPageCacheReservation {
            if charge.kind == SharedPageCacheChargeKind::Page {
                match self.capacity_then_defer_pages.load(StdOrdering::Relaxed) {
                    1 => {
                        self.capacity_then_defer_pages
                            .store(2, StdOrdering::Relaxed);
                        return SharedPageCacheReservation::CapacityDenied;
                    }
                    2 => return SharedPageCacheReservation::Deferred,
                    _ if self.defer_pages.load(StdOrdering::Relaxed) => {
                        return SharedPageCacheReservation::Deferred;
                    }
                    _ => {}
                }
            }
            if self.try_reserve(charge) {
                SharedPageCacheReservation::Reserved
            } else {
                SharedPageCacheReservation::CapacityDenied
            }
        }

        fn transition(
            &self,
            charge: SharedPageCacheCharge,
            from: SharedPageCacheChargeState,
            to: SharedPageCacheChargeState,
        ) {
            self.transition_calls.fetch_add(1, StdOrdering::Relaxed);
            self.state_bytes(from)
                .fetch_sub(charge.bytes, StdOrdering::Relaxed);
            self.state_bytes(to)
                .fetch_add(charge.bytes, StdOrdering::Relaxed);
        }

        fn release(&self, charge: SharedPageCacheCharge, state: SharedPageCacheChargeState) {
            self.release_calls.fetch_add(1, StdOrdering::Relaxed);
            match charge.kind {
                SharedPageCacheChargeKind::MetadataCapacity => {
                    self.metadata_release_calls
                        .fetch_add(1, StdOrdering::Relaxed);
                }
                SharedPageCacheChargeKind::Page => {
                    self.page_release_calls.fetch_add(1, StdOrdering::Relaxed);
                }
            }
            self.state_bytes(state)
                .fetch_sub(charge.bytes, StdOrdering::Relaxed);
            self.used.fetch_sub(charge.bytes, StdOrdering::Relaxed);
        }
    }

    fn begin_owned_allocation_observation() {
        OWNED_ALLOCATION_COUNTS.with(|counts| counts.set(Some([0; 5])));
    }

    fn end_owned_allocation_observation() -> [usize; 5] {
        OWNED_ALLOCATION_COUNTS.with(|counts| {
            counts
                .replace(None)
                .expect("owned allocation observation must be active")
        })
    }

    fn one_shard_capacity(page_size: usize, entries: usize) -> usize {
        SharedPageKey {
            namespace: 1,
            page_id: 1,
            page_size: page_size as u32,
            version: version(1),
        }
        .checked_weight(page_size)
        .expect("test page weight must fit")
            * entries
    }

    #[test]
    fn getter_keeps_charge_after_eviction_and_cache_drop() {
        let page_size = 256;
        let max_entries = NonZeroUsize::new(1).expect("one is non-zero");
        let capacity = one_shard_capacity(page_size, 1);
        let metadata = metadata_charge_bytes(capacity, max_entries).expect("metadata charge");
        let page_charge = page_charge_bytes(page_size).expect("page charge");
        let accounting = Arc::new(TestAccounting::with_limit(metadata + page_charge * 2));
        let cache = Arc::new(
            SharedPageCache::try_with_accounting(capacity, max_entries, accounting.clone(), None)
                .expect("metadata admission succeeds"),
        );
        let namespace = cache.new_namespace();

        namespace
            .publisher(1, page_size as u32, version(1))
            .publish(&vec![17; page_size]);
        let first_handle = namespace
            .get(1, page_size as u32, version(1))
            .expect("first page is cached");
        let held_handle = first_handle.clone();
        drop(first_handle);
        namespace
            .publisher(2, page_size as u32, version(2))
            .publish(&vec![29; page_size]);

        assert_eq!(held_handle.as_ref(), &[17; 256]);
        assert_eq!(accounting.retired.load(StdOrdering::Relaxed), page_charge);
        drop(namespace);
        drop(cache);

        assert_eq!(accounting.reserved.load(StdOrdering::Relaxed), 0);
        assert_eq!(accounting.resident.load(StdOrdering::Relaxed), 0);
        assert_eq!(accounting.retired.load(StdOrdering::Relaxed), page_charge);
        assert_eq!(accounting.used.load(StdOrdering::Relaxed), page_charge);
        assert_eq!(
            accounting.metadata_release_calls.load(StdOrdering::Relaxed),
            1
        );
        assert_eq!(accounting.page_release_calls.load(StdOrdering::Relaxed), 1);

        drop(held_handle);
        assert_eq!(accounting.used.load(StdOrdering::Relaxed), 0);
        assert_eq!(accounting.retired.load(StdOrdering::Relaxed), 0);
        assert_eq!(accounting.page_release_calls.load(StdOrdering::Relaxed), 2);
    }

    #[test]
    fn reclaim_releases_only_cache_owned_accounted_pages() {
        let page_size = 256;
        let max_entries = NonZeroUsize::new(3).expect("three is non-zero");
        let capacity = one_shard_capacity(page_size, max_entries.get());
        let metadata = metadata_charge_bytes(capacity, max_entries).expect("metadata charge");
        let page_charge = page_charge_bytes(page_size).expect("page charge");
        let accounting = Arc::new(TestAccounting::with_limit(metadata + page_charge * 3));
        let cache = Arc::new(
            SharedPageCache::try_with_accounting(capacity, max_entries, accounting.clone(), None)
                .expect("metadata admission succeeds"),
        );
        let namespace = cache.new_namespace();
        for page_id in 1..=3 {
            namespace
                .publisher(page_id, page_size as u32, version(page_id as u64))
                .publish(&vec![page_id as u8; page_size]);
        }
        let held = namespace
            .get(1, page_size as u32, version(1))
            .expect("oldest page remains readable");

        assert_eq!(cache.reclaim_unreferenced(page_charge * 2), page_charge * 2);
        assert_eq!(cache.stats().entries, 1);
        assert_eq!(
            accounting.used.load(StdOrdering::Relaxed),
            metadata + page_charge
        );
        assert_eq!(held.as_ref(), &[1; 256]);
        assert_eq!(cache.reclaim_unreferenced(page_charge), 0);

        drop(held);
        assert_eq!(cache.reclaim_unreferenced(page_charge), page_charge);
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(accounting.used.load(StdOrdering::Relaxed), metadata);
    }

    #[test]
    fn reclaim_is_opt_in_for_legacy_cache() {
        let cache = Arc::new(SharedPageCache::new(1024));
        let namespace = cache.new_namespace();
        namespace.publisher(1, 256, version(1)).publish(&[7; 256]);

        assert_eq!(cache.reclaim_unreferenced(usize::MAX), 0);
        assert_eq!(cache.stats().entries, 1);
    }

    #[test]
    fn deferred_admission_preserves_full_shard_until_retry() {
        let page_size = 256;
        let max_entries = NonZeroUsize::new(2).expect("two is non-zero");
        let capacity = one_shard_capacity(page_size, max_entries.get());
        let metadata = metadata_charge_bytes(capacity, max_entries).expect("metadata charge");
        let page_charge = page_charge_bytes(page_size).expect("page charge");
        let accounting = Arc::new(TestAccounting::with_limit(metadata + page_charge * 3));
        let cache = Arc::new(
            SharedPageCache::try_with_accounting(capacity, max_entries, accounting.clone(), None)
                .expect("metadata admission succeeds"),
        );
        let namespace = cache.new_namespace();
        for page_id in 1..=2 {
            namespace
                .publisher(page_id, page_size as u32, version(page_id as u64))
                .publish(&vec![page_id as u8; page_size]);
        }
        let before = cache.stats();

        accounting.defer_pages.store(true, StdOrdering::Relaxed);
        namespace
            .publisher(3, page_size as u32, version(3))
            .publish(&vec![3; page_size]);

        let deferred = cache.stats();
        assert_eq!(deferred.entries, before.entries);
        assert_eq!(deferred.evictions, before.evictions);
        assert_eq!(deferred.insertions, before.insertions);
        assert_eq!(
            namespace
                .get(1, page_size as u32, version(1))
                .expect("oldest page survives deferral")
                .as_ref(),
            &[1; 256]
        );
        assert_eq!(
            namespace
                .get(2, page_size as u32, version(2))
                .expect("newest page survives deferral")
                .as_ref(),
            &[2; 256]
        );

        accounting.defer_pages.store(false, StdOrdering::Relaxed);
        namespace
            .publisher(3, page_size as u32, version(3))
            .publish(&vec![3; page_size]);
        let retried = cache.stats();
        assert_eq!(retried.entries, before.entries);
        assert_eq!(retried.evictions, before.evictions + 1);
        assert_eq!(retried.insertions, before.insertions + 1);
        assert_eq!(
            namespace
                .get(3, page_size as u32, version(3))
                .expect("retry inserts the deferred page")
                .as_ref(),
            &[3; 256]
        );
    }

    #[test]
    fn deferred_admission_preserves_same_key_value_until_retry() {
        let page_size = 256;
        let max_entries = NonZeroUsize::new(1).expect("one is non-zero");
        let capacity = one_shard_capacity(page_size, 1);
        let metadata = metadata_charge_bytes(capacity, max_entries).expect("metadata charge");
        let page_charge = page_charge_bytes(page_size).expect("page charge");
        let accounting = Arc::new(TestAccounting::with_limit(metadata + page_charge * 2));
        let cache = Arc::new(
            SharedPageCache::try_with_accounting(capacity, max_entries, accounting.clone(), None)
                .expect("metadata admission succeeds"),
        );
        let namespace = cache.new_namespace();
        let publisher = namespace.publisher(1, page_size as u32, version(1));
        publisher.publish(&vec![17; page_size]);
        let before = cache.stats();

        accounting.defer_pages.store(true, StdOrdering::Relaxed);
        publisher.publish(&vec![29; page_size]);

        let deferred = cache.stats();
        assert_eq!(deferred.entries, before.entries);
        assert_eq!(deferred.replacements, before.replacements);
        assert_eq!(deferred.evictions, before.evictions);
        assert_eq!(
            namespace
                .get(1, page_size as u32, version(1))
                .expect("old replacement value survives deferral")
                .as_ref(),
            &[17; 256]
        );

        accounting.defer_pages.store(false, StdOrdering::Relaxed);
        publisher.publish(&vec![29; page_size]);
        let retried = cache.stats();
        assert_eq!(retried.entries, before.entries);
        assert_eq!(retried.replacements, before.replacements + 1);
        assert_eq!(retried.evictions, before.evictions);
        assert_eq!(
            namespace
                .get(1, page_size as u32, version(1))
                .expect("retry installs the replacement value")
                .as_ref(),
            &[29; 256]
        );
    }

    #[test]
    fn capacity_retry_that_defers_stops_after_one_victim() {
        let page_size = 256;
        let max_entries = NonZeroUsize::new(2).expect("two is non-zero");
        let capacity = one_shard_capacity(page_size, max_entries.get());
        let metadata = metadata_charge_bytes(capacity, max_entries).expect("metadata charge");
        let page_charge = page_charge_bytes(page_size).expect("page charge");
        let accounting = Arc::new(TestAccounting::with_limit(metadata + page_charge * 3));
        let cache = Arc::new(
            SharedPageCache::try_with_accounting(capacity, max_entries, accounting.clone(), None)
                .expect("metadata admission succeeds"),
        );
        let namespace = cache.new_namespace();
        for page_id in 1..=2 {
            namespace
                .publisher(page_id, page_size as u32, version(page_id as u64))
                .publish(&vec![page_id as u8; page_size]);
        }
        let before = cache.stats();

        accounting
            .capacity_then_defer_pages
            .store(1, StdOrdering::Relaxed);
        namespace
            .publisher(3, page_size as u32, version(3))
            .publish(&vec![3; page_size]);

        let deferred_retry = cache.stats();
        assert_eq!(deferred_retry.entries, before.entries - 1);
        assert_eq!(deferred_retry.evictions, before.evictions + 1);
        assert_eq!(deferred_retry.insertions, before.insertions);
        assert!(namespace.get(3, page_size as u32, version(3)).is_none());

        accounting
            .capacity_then_defer_pages
            .store(0, StdOrdering::Relaxed);
        namespace
            .publisher(3, page_size as u32, version(3))
            .publish(&vec![3; page_size]);
        let retried = cache.stats();
        assert_eq!(retried.entries, before.entries);
        assert_eq!(retried.evictions, before.evictions + 1);
        assert_eq!(retried.insertions, before.insertions + 1);
    }

    #[test]
    fn admission_denied_before_owned_allocation() {
        let page_size = 256;
        let max_entries = NonZeroUsize::new(1).expect("one is non-zero");
        let capacity = one_shard_capacity(page_size, 1);

        let base_denied = Arc::new(TestAccounting::with_limit(0));
        begin_owned_allocation_observation();
        let result =
            SharedPageCache::try_with_accounting(capacity, max_entries, base_denied.clone(), None);
        let base_allocations = end_owned_allocation_observation();
        assert!(matches!(
            result,
            Err(SharedPageCacheCreateError::AdmissionDenied)
        ));
        assert_eq!(base_allocations, [0; 5]);
        assert_eq!(base_denied.used.load(StdOrdering::Relaxed), 0);

        let metadata = metadata_charge_bytes(capacity, max_entries).expect("metadata charge");
        let page_denied = Arc::new(TestAccounting::with_limit(metadata));
        begin_owned_allocation_observation();
        let cache = Arc::new(
            SharedPageCache::try_with_accounting(capacity, max_entries, page_denied.clone(), None)
                .expect("metadata admission succeeds"),
        );
        let admitted_base_allocations = end_owned_allocation_observation();
        assert_eq!(
            admitted_base_allocations[OwnedAllocationKind::ShardIndex.index()],
            1
        );
        assert_eq!(
            admitted_base_allocations[OwnedAllocationKind::ShardBacking.index()],
            1
        );
        let namespace = cache.new_namespace();
        let fixture = vec![3; page_size];
        begin_owned_allocation_observation();
        namespace
            .publisher(1, page_size as u32, version(1))
            .publish(&fixture);
        let page_allocations = end_owned_allocation_observation();
        assert_eq!(page_allocations[OwnedAllocationKind::PageBytes.index()], 0);
        assert_eq!(page_allocations[OwnedAllocationKind::PageOwner.index()], 0);
        assert_eq!(page_allocations[OwnedAllocationKind::IndexNode.index()], 0);
        assert_eq!(cache.stats().rejected, 1);

        let early_return_charge = SharedPageCacheCharge {
            kind: SharedPageCacheChargeKind::Page,
            bytes: 1,
        };
        page_denied.limit.store(metadata + 1, StdOrdering::Relaxed);
        let reservation = Reservation::try_new(page_denied.clone(), early_return_charge)
            .expect("the early-return reservation is admitted");
        assert_eq!(page_denied.used.load(StdOrdering::Relaxed), metadata + 1);
        drop(reservation);
        assert_eq!(page_denied.used.load(StdOrdering::Relaxed), metadata);
        assert_eq!(page_denied.reserved.load(StdOrdering::Relaxed), metadata);

        page_denied.limit.store(
            metadata + page_charge_bytes(page_size).expect("page charge"),
            StdOrdering::Relaxed,
        );
        begin_owned_allocation_observation();
        namespace
            .publisher(2, page_size as u32, version(2))
            .publish(&fixture);
        let admitted_page_allocations = end_owned_allocation_observation();
        assert_eq!(
            admitted_page_allocations[OwnedAllocationKind::PageBytes.index()],
            1
        );
        assert_eq!(
            admitted_page_allocations[OwnedAllocationKind::PageOwner.index()],
            1
        );
        assert_eq!(
            admitted_page_allocations[OwnedAllocationKind::IndexNode.index()],
            1
        );
    }

    #[test]
    fn bounded_index_churn_retains_metadata_reservation() {
        let boundaries = [(1, 4), (3, 4), (4, 8), (7, 8), (8, 16), (14, 16), (15, 32)];
        for (entries, expected_buckets) in boundaries {
            assert_eq!(initial_bucket_count(entries), Some(expected_buckets));
            let entries = NonZeroUsize::new(entries).expect("boundary is non-zero");
            assert!(metadata_charge_bytes(0, entries).is_some());
        }
        assert_eq!(initial_bucket_count(usize::MAX), None);
        assert_eq!(
            metadata_charge_bytes(
                usize::MAX,
                NonZeroUsize::new(usize::MAX).expect("usize max is non-zero")
            ),
            None
        );

        let page_size = 128;
        let max_entries = NonZeroUsize::new(3).expect("three is non-zero");
        let capacity = one_shard_capacity(page_size, max_entries.get());
        let metadata = metadata_charge_bytes(capacity, max_entries).expect("metadata charge");
        let page_charge = page_charge_bytes(page_size).expect("page charge");
        let accounting = Arc::new(TestAccounting::with_limit(
            metadata * 2 + page_charge * max_entries.get(),
        ));
        let cache = Arc::new(
            SharedPageCache::try_with_accounting(capacity, max_entries, accounting.clone(), None)
                .expect("first metadata reservation succeeds"),
        );
        let namespace = cache.new_namespace();
        for page_id in 1..=64 {
            namespace
                .publisher(page_id, page_size as u32, version(page_id as u64))
                .publish(&vec![page_id as u8; page_size]);
        }
        assert_eq!(cache.stats().entries, max_entries.get() as u64);
        assert_eq!(accounting.reserved.load(StdOrdering::Relaxed), metadata);

        accounting.deny_pages.store(true, StdOrdering::Relaxed);
        namespace
            .publisher(100, page_size as u32, version(100))
            .publish(&vec![100; page_size]);
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(accounting.used.load(StdOrdering::Relaxed), metadata);
        assert_eq!(accounting.reserved.load(StdOrdering::Relaxed), metadata);

        accounting.limit.store(metadata * 2, StdOrdering::Relaxed);
        let second =
            SharedPageCache::try_with_accounting(capacity, max_entries, accounting.clone(), None)
                .expect("remaining metadata capacity admits one more cache");
        let third =
            SharedPageCache::try_with_accounting(capacity, max_entries, accounting.clone(), None);
        assert!(matches!(
            third,
            Err(SharedPageCacheCreateError::AdmissionDenied)
        ));
        drop(second);
        assert_eq!(accounting.used.load(StdOrdering::Relaxed), metadata);
    }

    #[test]
    fn concurrent_publication_retires_once() {
        let page_size = 256;
        let max_entries = NonZeroUsize::new(1).expect("one is non-zero");
        let capacity = one_shard_capacity(page_size, 1);
        let metadata = metadata_charge_bytes(capacity, max_entries).expect("metadata charge");
        let page_charge = page_charge_bytes(page_size).expect("page charge");
        let accounting = Arc::new(TestAccounting::with_limit(metadata + page_charge * 3));
        let cache = Arc::new(
            SharedPageCache::try_with_accounting(capacity, max_entries, accounting.clone(), None)
                .expect("metadata admission succeeds"),
        );
        let namespace = cache.new_namespace();
        namespace
            .publisher(1, page_size as u32, version(1))
            .publish(&vec![1; page_size]);

        let (held_sender, held_receiver) = mpsc::channel();
        let (release_sender, release_receiver) = mpsc::channel();
        let getter_namespace = namespace.clone();
        let getter = std::thread::spawn(move || {
            let held = getter_namespace
                .get(1, page_size as u32, version(1))
                .expect("getter races with no removal until after its signal");
            drop(getter_namespace);
            held_sender.send(()).expect("main receives held signal");
            release_receiver.recv().expect("main releases held handle");
            assert_eq!(held.as_ref(), &[1; 256]);
            drop(held);
        });
        held_receiver
            .recv()
            .expect("getter holds the first allocation");

        let start = Arc::new(Barrier::new(3));
        let publishers = [2, 3].map(|page_id| {
            let publisher_namespace = namespace.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                start.wait();
                publisher_namespace
                    .publisher(page_id, page_size as u32, version(page_id as u64))
                    .publish(&vec![page_id as u8; page_size]);
            })
        });
        start.wait();
        for publisher in publishers {
            publisher.join().expect("publisher thread completes");
        }

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert!(stats.resident_bytes <= stats.capacity_bytes);
        assert!(accounting.peak.load(StdOrdering::Relaxed) <= metadata + page_charge * 3);
        assert_eq!(accounting.retired.load(StdOrdering::Relaxed), page_charge);
        assert_eq!(accounting.page_release_calls.load(StdOrdering::Relaxed), 1);

        drop(namespace);
        drop(cache);
        assert_eq!(accounting.retired.load(StdOrdering::Relaxed), page_charge);
        assert_eq!(accounting.transition_calls.load(StdOrdering::Relaxed), 6);
        release_sender.send(()).expect("getter is released");
        getter.join().expect("getter thread completes");
        assert_eq!(accounting.used.load(StdOrdering::Relaxed), 0);
        assert_eq!(accounting.page_release_calls.load(StdOrdering::Relaxed), 3);
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
    fn default_constructor_preserves_legacy_value_and_allocation_path() {
        let cache = Arc::new(SharedPageCache::new(64 * 1024));
        assert!(matches!(&cache.shards, CacheShards::Legacy(_)));
        let namespace = cache.new_namespace();
        let fixture = vec![41; 4096];

        begin_owned_allocation_observation();
        namespace.publisher(7, 4096, version(9)).publish(&fixture);
        let managed_allocations = end_owned_allocation_observation();

        assert_eq!(managed_allocations, [0; 5]);
        assert_eq!(
            namespace
                .get(7, 4096, version(9))
                .expect("legacy page remains available")
                .as_ref(),
            fixture
        );
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

    #[test]
    fn accounted_cache_preserves_observer_contract() {
        let max_entries = NonZeroUsize::new(2).expect("two is non-zero");
        let capacity = 64 * 1024;
        let metadata = metadata_charge_bytes(capacity, max_entries).expect("metadata charge");
        let page_charge = page_charge_bytes(4096).expect("page charge");
        let accounting = Arc::new(TestAccounting::with_limit(metadata + page_charge));
        let observer = Arc::new(LookupObserver::default());
        let cache = Arc::new(
            SharedPageCache::try_with_accounting(
                capacity,
                max_entries,
                accounting,
                Some(observer.clone()),
            )
            .expect("accounted cache is admitted"),
        );
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
    fn separate_databases_keep_independent_namespaces_in_one_cache() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let io = Arc::new(PlatformIO::new().expect("platform I/O"));
        let first_path = directory.path().join("first.db");
        let second_path = directory.path().join("second.db");
        let first_database = Database::open_file(
            io.clone(),
            first_path.to_str().expect("UTF-8 first database path"),
        )
        .expect("first database opens");
        let second_database = Database::open_file(
            io,
            second_path.to_str().expect("UTF-8 second database path"),
        )
        .expect("second database opens");
        seed_rows(
            &first_database.connect().expect("first seed connection"),
            32,
        );
        seed_rows(
            &second_database.connect().expect("second seed connection"),
            32,
        );

        let cache = Arc::new(SharedPageCache::new(8 * 1024 * 1024));
        first_database.set_shared_page_cache(Some(cache.clone()));
        second_database.set_shared_page_cache(Some(cache.clone()));

        assert_eq!(
            row_count(&first_database.connect().expect("first reader")),
            32
        );
        let after_first = cache.stats();
        assert_eq!(
            row_count(&second_database.connect().expect("second reader")),
            32
        );
        let after_second = cache.stats();

        assert!(after_second.misses > after_first.misses);
        assert!(after_second.entries > after_first.entries);
    }

    #[cfg(feature = "fs")]
    #[test]
    fn cache_setter_only_affects_future_connections() {
        let (_directory, database, cache) = database_with_cache_capacity(8 * 1024 * 1024);
        database.set_shared_page_cache(None);
        let writer = database.connect().expect("writer connection");
        seed_rows(&writer, 32);
        let connection_before_setter = database.connect().expect("connection before setter");

        database.set_shared_page_cache(Some(cache.clone()));
        assert_eq!(row_count(&connection_before_setter), 32);
        assert_eq!(cache.stats().entries, 0);

        let connection_after_setter = database.connect().expect("connection after setter");
        assert_eq!(row_count(&connection_after_setter), 32);
        assert!(cache.stats().entries > 0);
    }

    #[cfg(feature = "fs")]
    #[test]
    fn accounted_publication_refusal_is_a_cache_bypass() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("accounting-refusal.db");
        let io = Arc::new(PlatformIO::new().expect("platform I/O"));
        let database =
            Database::open_file(io, path.to_str().expect("UTF-8 accounting-refusal path"))
                .expect("database opens");
        seed_rows(&database.connect().expect("seed connection"), 32);

        let capacity = 8 * 1024 * 1024;
        let max_entries = NonZeroUsize::new(64).expect("64 is non-zero");
        let metadata = metadata_charge_bytes(capacity, max_entries).expect("metadata charge");
        let accounting = Arc::new(TestAccounting::with_limit(metadata));
        let cache = Arc::new(
            SharedPageCache::try_with_accounting(capacity, max_entries, accounting, None)
                .expect("metadata admission succeeds"),
        );
        database.set_shared_page_cache(Some(cache.clone()));

        assert_eq!(
            row_count(&database.connect().expect("reader connection")),
            32
        );
        let stats = cache.stats();
        assert_eq!(stats.entries, 0);
        assert!(stats.rejected > 0);
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
