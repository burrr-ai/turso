use crate::{turso_assert, turso_assert_eq};
use core::fmt::{self, Debug};
use std::{
    future::Future,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, OnceLock,
    },
    task::{Poll, Waker},
};

use crate::sync::Mutex;

use crate::{Buffer, CompletionError};

/// Callback for read completions. Returns `Some(error)` if the callback detects an error
/// (e.g., short read), which will be stored in the completion and propagated to VDBE.
pub type ReadComplete =
    dyn Fn(Result<(Arc<Buffer>, i32), CompletionError>) -> Option<CompletionError> + Send + Sync;
pub type WriteComplete = dyn Fn(Result<i32, CompletionError>) + Send + Sync;
pub type SyncComplete = dyn Fn(Result<i32, CompletionError>) + Send + Sync;
pub type TruncateComplete = dyn Fn(Result<i32, CompletionError>) + Send + Sync;

#[must_use]
#[derive(Debug, Clone)]
pub struct Completion {
    /// Optional completion state. If None, it means we are Yield in order to not allocate anything
    pub(super) inner: Option<Arc<CompletionInner>>,
}

impl Future for Completion {
    type Output = Result<(), crate::LimboError>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        self.set_waker(cx.waker());
        if self.finished() {
            self.wake();
            let res = self
                .get_error()
                .map_or(Ok(()), |err| Err(crate::LimboError::CompletionError(err)));
            return Poll::Ready(res);
        }
        Poll::Pending
    }
}

#[derive(Debug, Default)]
struct ContextInner {
    waker: Option<Waker>,
    // TODO: add abort signal
}

#[derive(Debug, Clone)]
pub struct Context {
    inner: Arc<Mutex<ContextInner>>,
}

impl ContextInner {
    pub fn new() -> Self {
        Self { waker: None }
    }

    pub fn wake(&mut self) {
        if let Some(waker) = self.waker.take() {
            waker.wake();
        }
    }

    pub fn set_waker(&mut self, waker: &Waker) {
        if let Some(curr_waker) = self.waker.as_mut() {
            // only call and change waker if it would awake a different task
            if !curr_waker.will_wake(waker) {
                let prev_waker = std::mem::replace(curr_waker, waker.clone());
                prev_waker.wake();
            }
        } else {
            self.waker = Some(waker.clone());
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(ContextInner::new())),
        }
    }

    pub fn wake(&self) {
        self.inner.lock().wake();
    }

    pub fn set_waker(&self, waker: &Waker) {
        self.inner.lock().set_waker(waker);
    }
}

pub(super) struct CompletionInner {
    completion_type: CompletionType,
    /// None means we completed successfully
    // Thread safe with OnceLock
    pub(super) result: crate::sync::OnceLock<Option<CompletionError>>,
    context: Context,
    /// The group this completion belongs to and whether the group was told
    /// about its result.
    link: Mutex<ParentLink>,
    /// Keeps the write buffer alive for async I/O backends (io_uring, VFS)
    /// where pwrite returns before the kernel has consumed the buffer.
    write_buffer: OnceLock<Arc<Buffer>>,
}

impl fmt::Debug for CompletionInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompletionInner")
            .field("completion_type", &self.completion_type)
            .field("parent", &self.link.lock().parent.is_some())
            .finish()
    }
}

/// A completion's membership in a [`CompletionGroup`]. Registration with the
/// group and publication of the completion's result can happen in either
/// order on different threads; whichever comes second sees both under this
/// lock and claims the one notification to the group. The group is told
/// outside the lock, so no callback, waker or other completion's lock ever
/// runs under it.
#[derive(Default)]
struct ParentLink {
    parent: Option<Arc<GroupCompletionInner>>,
    /// Set by the call that published the completion's `result`, after it did.
    published: bool,
    notified: bool,
}

impl ParentLink {
    /// Called once after registration and once after result publication;
    /// only the second of the two finds both and claims the notification.
    fn claim(&mut self) -> Option<Arc<GroupCompletionInner>> {
        let (Some(parent), true) = (&self.parent, self.published) else {
            return None;
        };
        turso_assert!(
            !self.notified,
            "a completion's group is told about its result only once"
        );
        self.notified = true;
        Some(parent.clone())
    }
}

pub struct CompletionGroup {
    completions: Vec<Completion>,
    callback: Box<dyn Fn(Result<i32, CompletionError>) + Send + Sync>,
}

impl CompletionGroup {
    pub fn new<F>(callback: F) -> Self
    where
        F: Fn(Result<i32, CompletionError>) + Send + Sync + 'static,
    {
        Self {
            completions: Vec::new(),
            callback: Box::new(callback),
        }
    }

    /// A non-yield completion belongs to one group only: adding it to a second
    /// group, even after it finished, panics in `build`. An explicit yield
    /// completion (an empty group's result, for one) has nothing to link and
    /// counts as already succeeded in any group.
    pub fn add(&mut self, completion: &Completion) {
        self.completions.push(completion.clone());
    }

    /// The children added so far. Used by error paths that need to
    /// wait on the kernel side via `IO::drain_completions` after
    /// cancelling the group.
    pub fn completions(&self) -> &[Completion] {
        &self.completions
    }

    pub fn cancel(&self) {
        for c in &self.completions {
            c.abort();
        }
    }

    /// Finishes once every child finished, with the first error it reconciled;
    /// panics if a non-yield child is already in another group.
    pub fn build(self) -> Completion {
        let total = self.completions.len();
        if total == 0 {
            (self.callback)(Ok(0));
            return Completion::new_yield();
        }
        // One count more than the children: build holds it until every child
        // is registered, so children finishing meanwhile cannot finish the
        // group before it is built.
        let group_completion = GroupCompletion::new(self.callback, total + 1);
        let group = Completion::new(CompletionType::Group(group_completion));
        let group_inner = match &group.get_inner().completion_type {
            CompletionType::Group(g) => g.inner.clone(),
            _ => unreachable!(),
        };
        // Installed before any child is registered: a registered child may
        // be the one that finishes the group.
        group_inner
            .self_completion
            .set(group.clone())
            .expect("a new group has no completion yet");

        // Every child is registered, finished or not, even after one failed:
        // the group finishes only once every child's IO has finished, with
        // the first error.
        for c in self.completions {
            #[cfg(any(test, feature = "completion_test_hooks"))]
            registration_hooks::before_child_register(&c);
            c.link_internal(&group_inner);
            #[cfg(any(test, feature = "completion_test_hooks"))]
            registration_hooks::after_child_register();
        }
        group_inner.release();
        group
    }
}

/// Test-only pause points in the bookkeeping between a [`CompletionGroup`]
/// and its children. Every point is outside the bookkeeping lock, so a paused
/// thread blocks no other thread's completion. Hooks are scoped to the
/// calling thread, so builds and completions on other threads are unaffected.
#[cfg(any(test, feature = "completion_test_hooks"))]
pub mod registration_hooks {
    use std::cell::RefCell;

    use super::Completion;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum HookPoint {
        /// In [`super::CompletionGroup::build`], after a child was observed
        /// unfinished and before it is registered with the group.
        BeforeChildRegister,
        /// In [`super::CompletionGroup::build`], after a child was registered
        /// and before the next one (or the build's own count) is handled.
        AfterChildRegister,
        /// In a completion's callback, after its result was published and
        /// before its group, if any, is told.
        AfterResultPublished,
    }

    type Hook = Box<dyn FnMut(HookPoint)>;

    thread_local! {
        static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
    }

    /// Runs `f` with `hook` called at every [`HookPoint`] this thread reaches.
    pub fn with_hook<T>(hook: impl FnMut(HookPoint) + 'static, f: impl FnOnce() -> T) -> T {
        struct Restore(Option<Hook>);
        impl Drop for Restore {
            fn drop(&mut self) {
                let previous = self.0.take();
                let _ = HOOK.try_with(|slot| *slot.borrow_mut() = previous);
            }
        }
        let previous = HOOK.with(|slot| slot.borrow_mut().replace(Box::new(hook)));
        let _restore = Restore(previous);
        f()
    }

    /// Runs `f` with `hook` called each time a `CompletionGroup::build` on
    /// this thread is about to register a child it observed unfinished.
    pub fn with_before_child_register<T>(
        mut hook: impl FnMut() + 'static,
        f: impl FnOnce() -> T,
    ) -> T {
        with_hook(
            move |point| {
                if point == HookPoint::BeforeChildRegister {
                    hook();
                }
            },
            f,
        )
    }

    pub(super) fn before_child_register(child: &Completion) {
        if !child.finished() {
            hit(HookPoint::BeforeChildRegister);
        }
    }

    pub(super) fn after_child_register() {
        hit(HookPoint::AfterChildRegister);
    }

    pub(super) fn after_result_published() {
        hit(HookPoint::AfterResultPublished);
    }

    fn hit(point: HookPoint) {
        // Taken out while it runs, so the hook itself may build a group or
        // complete a completion.
        let Some(mut hook) = HOOK.with(|slot| slot.borrow_mut().take()) else {
            return;
        };
        hook(point);
        HOOK.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.is_none() {
                *slot = Some(hook);
            }
        });
    }
}

pub struct GroupCompletion {
    inner: Arc<GroupCompletionInner>,
}

impl fmt::Debug for GroupCompletion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GroupCompletion")
            .field(
                "outstanding",
                &self.inner.outstanding.load(Ordering::SeqCst),
            )
            .finish()
    }
}

struct GroupCompletionInner {
    /// Children not yet accounted for, plus the one count `build` holds
    /// until every child is registered.
    outstanding: AtomicUsize,
    /// Callback to invoke when all completions finish
    complete: Box<dyn Fn(Result<i32, CompletionError>) + Send + Sync>,
    /// The first child error this group observed while reconciling its
    /// children, not necessarily the earliest IO to fail; the group finishes
    /// with it.
    first_error: OnceLock<CompletionError>,
    /// Reference to the group's own Completion for notifying parents
    self_completion: OnceLock<Completion>,
}

impl GroupCompletionInner {
    /// Accounts for one child's result. [`ParentLink::claim`] makes sure
    /// this is called exactly once per child.
    fn child_finished(&self, error: Option<CompletionError>) {
        if let Some(err) = error {
            // Only the first error is kept.
            let _ = self.first_error.set(err);
        }
        self.release();
    }

    /// Drops one count. The call that takes it to zero finishes the group,
    /// exactly once, through the group completion's own callback, so the
    /// group's result is published like any other completion's.
    fn release(&self) {
        let prev = self.outstanding.fetch_sub(1, Ordering::SeqCst);
        turso_assert!(
            prev > 0,
            "completion group released more counts than it holds"
        );
        let group = self
            .self_completion
            .get()
            .expect("build installs the group completion before registering children");
        if prev == 1 {
            group.callback(self.first_error.get().map_or(Ok(0), |err| Err(*err)));
        } else {
            // progress wake so the waiter keeps driving io.step
            group.wake();
        }
    }
}

impl GroupCompletion {
    pub fn new<F>(complete: F, outstanding: usize) -> Self
    where
        F: Fn(Result<i32, CompletionError>) + Send + Sync + 'static,
    {
        Self {
            inner: Arc::new(GroupCompletionInner {
                outstanding: AtomicUsize::new(outstanding),
                complete: Box::new(complete),
                first_error: OnceLock::new(),
                self_completion: OnceLock::new(),
            }),
        }
    }

    pub fn callback(&self, result: Result<i32, CompletionError>) {
        turso_assert_eq!(
            self.inner.outstanding.load(Ordering::SeqCst),
            0,
            "callback called before all completions finished"
        );
        (self.inner.complete)(result);
    }
}

impl Debug for CompletionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(..) => f.debug_tuple("Read").finish(),
            Self::Write(..) => f.debug_tuple("Write").finish(),
            Self::Sync(..) => f.debug_tuple("Sync").finish(),
            Self::Truncate(..) => f.debug_tuple("Truncate").finish(),
            Self::Group(..) => f.debug_tuple("Group").finish(),
            Self::Yield => f.debug_tuple("Yield").finish(),
        }
    }
}

pub enum CompletionType {
    Read(ReadCompletion),
    Write(WriteCompletion),
    Sync(SyncCompletion),
    Truncate(TruncateCompletion),
    Group(GroupCompletion),
    Yield,
}

impl CompletionInner {
    fn new(completion_type: CompletionType) -> Self {
        Self {
            completion_type,
            result: OnceLock::new(),
            context: Context::new(),
            link: Mutex::new(ParentLink::default()),
            write_buffer: OnceLock::new(),
        }
    }
}

impl Completion {
    pub fn new(completion_type: CompletionType) -> Self {
        Self {
            inner: Some(Arc::new(CompletionInner::new(completion_type))),
        }
    }

    pub(super) fn get_inner(&self) -> &Arc<CompletionInner> {
        self.inner
            .as_ref()
            .expect("completion inner should be initialized")
    }

    /// Stores a write buffer reference in the completion to keep it alive
    /// until the I/O completes. Required for async backends (io_uring, VFS)
    /// where pwrite returns before the kernel has consumed the buffer.
    pub fn keep_write_buffer_alive(&self, buf: Arc<Buffer>) {
        self.get_inner()
            .write_buffer
            .set(buf)
            .expect("write buffer should only be set once");
    }

    pub fn new_write<F>(complete: F) -> Self
    where
        F: Fn(Result<i32, CompletionError>) + Send + Sync + 'static,
    {
        Self::new(CompletionType::Write(WriteCompletion::new(Box::new(
            complete,
        ))))
    }

    pub fn new_read<F>(buf: Arc<Buffer>, complete: F) -> Self
    where
        F: Fn(Result<(Arc<Buffer>, i32), CompletionError>) -> Option<CompletionError>
            + Send
            + Sync
            + 'static,
    {
        Self::new(CompletionType::Read(ReadCompletion::new(
            buf,
            Box::new(complete),
        )))
    }
    pub fn new_sync<F>(complete: F) -> Self
    where
        F: Fn(Result<i32, CompletionError>) + Send + Sync + 'static,
    {
        Self::new(CompletionType::Sync(SyncCompletion::new(Box::new(
            complete,
        ))))
    }

    pub fn new_trunc<F>(complete: F) -> Self
    where
        F: Fn(Result<i32, CompletionError>) + Send + Sync + 'static,
    {
        Self::new(CompletionType::Truncate(TruncateCompletion::new(Box::new(
            complete,
        ))))
    }

    /// Create a yield completion. These are completed by default allowing to yield control without
    /// allocating memory.
    pub fn new_yield() -> Self {
        Self { inner: None }
    }

    pub fn wake(&self) {
        if let Some(inner) = &self.inner {
            inner.context.wake();
        }
    }

    /// Fire a "progress wake" without consuming the completion's result —
    /// wakes the waker on this completion *and* on its parent group, if any.
    /// Used by IO backends to nudge the future when an operation made progress
    /// but isn't fully done (e.g. an io_uring writev that completed only the
    /// first chunk and was resubmitted internally). Without this, a poll
    /// whose drained CQEs are all intermediate-chunk completions returns
    /// without waking anything, and since `step()` is the only thing that
    /// drains the CQ, the resubmitted chunks pile up and the task deadlocks.
    pub fn wake_progress(&self) {
        if let Some(inner) = &self.inner {
            let parent = inner.link.lock().parent.clone();
            if let Some(group) = parent {
                if let Some(group_completion) = group.self_completion.get() {
                    group_completion.wake();
                }
            }
            inner.context.wake();
        }
    }

    pub fn set_waker(&self, waker: &Waker) {
        if self.finished() || self.inner.is_none() {
            waker.wake_by_ref();
        } else {
            self.get_inner().context.set_waker(waker);
        }
    }

    // A group's result is published like any other completion's, once,
    // after its callback returned, so these read the same field for groups.
    pub fn succeeded(&self) -> bool {
        match &self.inner {
            Some(inner) => inner.result.get().is_some_and(|e| e.is_none()),
            None => true,
        }
    }

    pub fn failed(&self) -> bool {
        match &self.inner {
            Some(inner) => inner.result.get().is_some_and(|val| val.is_some()),
            None => false,
        }
    }

    pub fn get_error(&self) -> Option<CompletionError> {
        match &self.inner {
            Some(inner) => inner.result.get().and_then(|res| *res),
            None => None,
        }
    }

    /// Checks if the Completion completed or errored
    pub fn finished(&self) -> bool {
        match &self.inner {
            Some(inner) => inner.result.get().is_some(),
            None => true,
        }
    }

    /// Returns true if this completion is an explicit yield — a signal to
    /// return control to the cooperative scheduler so other connections can make
    /// progress. Unlike real I/O completions that happen to be finished,
    /// yield completions must not be treated as "ready to continue immediately"
    /// because the yielding operation is waiting on external state (e.g. a lock
    /// held by another fiber) that can only change when other fibers are stepped.
    pub fn is_explicit_yield(&self) -> bool {
        self.inner.is_none()
    }

    pub fn complete(&self, result: i32) {
        let result = Ok(result);
        self.callback(result);
    }

    pub fn error(&self, err: CompletionError) {
        let result = Err(err);
        self.callback(result);
    }

    pub fn abort(&self) {
        self.error(CompletionError::Aborted);
    }

    fn callback(&self, result: Result<i32, CompletionError>) {
        let inner = self.get_inner();
        let mut published = None;
        inner.result.get_or_init(|| {
            // Run the type-specific callback. For ReadCompletion, this returns
            // an optional error detected by the callback (e.g., short read).
            let callback_error = match &inner.completion_type {
                CompletionType::Read(r) => r.callback(result),
                CompletionType::Write(w) => {
                    w.callback(result);
                    None
                }
                CompletionType::Sync(s) => {
                    s.callback(result);
                    None
                }
                CompletionType::Truncate(t) => {
                    t.callback(result);
                    None
                }
                CompletionType::Group(g) => {
                    g.callback(result);
                    None
                }
                CompletionType::Yield => None,
            };

            // Use callback error if present, otherwise use the original IO error
            let final_error = callback_error.or_else(|| result.err());
            published = Some(final_error);
            final_error
        });
        // Only the call that published the result tells the group, after the
        // result is visible and outside this completion's callback.
        if let Some(final_error) = published {
            #[cfg(any(test, feature = "completion_test_hooks"))]
            registration_hooks::after_result_published();
            let notify = {
                let mut link = inner.link.lock();
                link.published = true;
                link.claim()
            };
            if let Some(group) = notify {
                group.child_finished(final_error);
            }
        }
        // call the waker regardless
        inner.context.wake();
    }

    /// only call this method if you are sure that the completion is
    /// a ReadCompletion, panics otherwise
    pub fn as_read(&self) -> &ReadCompletion {
        let inner = self.get_inner();
        match inner.completion_type {
            CompletionType::Read(ref r) => r,
            _ => unreachable!(),
        }
    }

    /// Registers this completion as a child of `group` (internal use only).
    /// If its result was already published, the group is told here.
    fn link_internal(&self, group: &Arc<GroupCompletionInner>) {
        let Some(inner) = &self.inner else {
            // A yield completion is always finished, successfully.
            group.child_finished(None);
            return;
        };
        let notify = {
            let mut link = inner.link.lock();
            turso_assert!(link.parent.is_none(), "completion can only be linked once");
            link.parent = Some(group.clone());
            link.claim()
        };
        if let Some(group) = notify {
            let result = inner
                .result
                .get()
                .expect("a published completion has its result set");
            group.child_finished(*result);
        }
    }
}

pub struct ReadCompletion {
    pub buf: Arc<Buffer>,
    pub complete: Box<ReadComplete>,
}

impl ReadCompletion {
    pub fn new(buf: Arc<Buffer>, complete: Box<ReadComplete>) -> Self {
        Self { buf, complete }
    }

    pub fn buf(&self) -> &Buffer {
        &self.buf
    }

    pub fn callback(&self, bytes_read: Result<i32, CompletionError>) -> Option<CompletionError> {
        (self.complete)(bytes_read.map(|b| (self.buf.clone(), b)))
    }

    pub fn buf_arc(&self) -> Arc<Buffer> {
        self.buf.clone()
    }
}

pub struct WriteCompletion {
    pub complete: Box<WriteComplete>,
}

impl WriteCompletion {
    pub fn new(complete: Box<WriteComplete>) -> Self {
        Self { complete }
    }

    pub fn callback(&self, bytes_written: Result<i32, CompletionError>) {
        (self.complete)(bytes_written);
    }
}

pub struct SyncCompletion {
    pub complete: Box<SyncComplete>,
}

impl SyncCompletion {
    pub fn new(complete: Box<SyncComplete>) -> Self {
        Self { complete }
    }

    pub fn callback(&self, res: Result<i32, CompletionError>) {
        (self.complete)(res);
    }
}

pub struct TruncateCompletion {
    pub complete: Box<TruncateComplete>,
}

impl TruncateCompletion {
    pub fn new(complete: Box<TruncateComplete>) -> Self {
        Self { complete }
    }

    pub fn callback(&self, res: Result<i32, CompletionError>) {
        (self.complete)(res);
    }
}

#[cfg(test)]
mod tests {
    use crate::CompletionError;

    use super::*;

    #[test]
    fn test_completion_group_empty() {
        use crate::sync::atomic::{AtomicBool, Ordering};

        let callback_called = Arc::new(AtomicBool::new(false));
        let callback_called_clone = callback_called.clone();

        let group = CompletionGroup::new(move |_| {
            callback_called_clone.store(true, Ordering::SeqCst);
        });
        let group = group.build();
        assert!(group.finished());
        assert!(group.succeeded());
        assert!(group.get_error().is_none());

        // Verify the callback was actually called
        assert!(
            callback_called.load(Ordering::SeqCst),
            "callback should be called for empty group"
        );
    }

    #[test]
    fn test_completion_group_single_completion() {
        let mut group = CompletionGroup::new(|_| {});
        let c = Completion::new_write(|_| {});
        group.add(&c);
        let group = group.build();

        assert!(!group.finished());
        assert!(!group.succeeded());

        c.complete(0);

        assert!(group.finished());
        assert!(group.succeeded());
        assert!(group.get_error().is_none());
    }

    #[test]
    fn test_completion_group_multiple_completions() {
        let mut group = CompletionGroup::new(|_| {});
        let c1 = Completion::new_write(|_| {});
        let c2 = Completion::new_write(|_| {});
        let c3 = Completion::new_write(|_| {});
        group.add(&c1);
        group.add(&c2);
        group.add(&c3);
        let group = group.build();

        assert!(!group.succeeded());
        assert!(!group.finished());

        c1.complete(0);
        assert!(!group.succeeded());
        assert!(!group.finished());

        c2.complete(0);
        assert!(!group.succeeded());
        assert!(!group.finished());

        c3.complete(0);
        assert!(group.succeeded());
        assert!(group.finished());
    }

    #[test]
    fn test_completion_group_with_error() {
        let mut group = CompletionGroup::new(|_| {});
        let c1 = Completion::new_write(|_| {});
        let c2 = Completion::new_write(|_| {});
        group.add(&c1);
        group.add(&c2);
        let group = group.build();

        c1.complete(0);
        c2.error(CompletionError::Aborted);

        assert!(group.finished());
        assert!(!group.succeeded());
        assert_eq!(group.get_error(), Some(CompletionError::Aborted));
    }

    #[test]
    fn test_completion_group_callback() {
        use crate::sync::atomic::{AtomicBool, Ordering};
        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();

        let mut group = CompletionGroup::new(move |_| {
            called_clone.store(true, Ordering::SeqCst);
        });

        let c1 = Completion::new_write(|_| {});
        let c2 = Completion::new_write(|_| {});
        group.add(&c1);
        group.add(&c2);
        let group = group.build();

        assert!(!called.load(Ordering::SeqCst));

        c1.complete(0);
        assert!(!called.load(Ordering::SeqCst));

        c2.complete(0);
        assert!(called.load(Ordering::SeqCst));
        assert!(group.finished());
        assert!(group.succeeded());
    }

    #[test]
    fn test_completion_group_some_already_completed() {
        // Test some completions added to group, then finish before build()
        let mut group = CompletionGroup::new(|_| {});
        let c1 = Completion::new_write(|_| {});
        let c2 = Completion::new_write(|_| {});
        let c3 = Completion::new_write(|_| {});

        // Add all to group while pending
        group.add(&c1);
        group.add(&c2);
        group.add(&c3);

        // Complete c1 and c2 AFTER adding but BEFORE build()
        c1.complete(0);
        c2.complete(0);

        let group = group.build();

        // c1 and c2 finished before build(), so outstanding should account for them
        // Only c3 should be pending
        assert!(!group.finished());
        assert!(!group.succeeded());

        // Complete c3
        c3.complete(0);

        // Now the group should be finished
        assert!(group.finished());
        assert!(group.succeeded());
        assert!(group.get_error().is_none());
    }

    #[test]
    fn test_completion_group_all_already_completed() {
        // Test when all completions are already finished before build()
        let mut group = CompletionGroup::new(|_| {});
        let c1 = Completion::new_write(|_| {});
        let c2 = Completion::new_write(|_| {});

        // Complete both before adding to group
        c1.complete(0);
        c2.complete(0);

        group.add(&c1);
        group.add(&c2);

        let group = group.build();

        // All completions were already complete, so group should be finished immediately
        assert!(group.finished());
        assert!(group.succeeded());
        assert!(group.get_error().is_none());
    }

    #[test]
    fn test_completion_group_mixed_finished_and_pending() {
        use crate::sync::atomic::{AtomicBool, Ordering};
        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();

        let mut group = CompletionGroup::new(move |_| {
            called_clone.store(true, Ordering::SeqCst);
        });

        let c1 = Completion::new_write(|_| {});
        let c2 = Completion::new_write(|_| {});
        let c3 = Completion::new_write(|_| {});
        let c4 = Completion::new_write(|_| {});

        // Complete c1 and c3 before adding to group
        c1.complete(0);
        c3.complete(0);

        group.add(&c1);
        group.add(&c2);
        group.add(&c3);
        group.add(&c4);

        let group = group.build();

        // Only c2 and c4 should be pending
        assert!(!group.finished());
        assert!(!called.load(Ordering::SeqCst));

        c2.complete(0);
        assert!(!group.finished());
        assert!(!called.load(Ordering::SeqCst));

        c4.complete(0);
        assert!(group.finished());
        assert!(group.succeeded());
        assert!(called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_completion_group_already_completed_with_error() {
        // Test when a completion finishes with error before build()
        let mut group = CompletionGroup::new(|_| {});
        let c1 = Completion::new_write(|_| {});
        let c2 = Completion::new_write(|_| {});

        // Complete c1 with error before adding to group
        c1.error(CompletionError::Aborted);

        group.add(&c1);
        group.add(&c2);

        let group = group.build();

        // c2's IO is still running; ending now would let a caller free its buffers under it.
        assert!(!group.finished());
        c2.complete(0);
        assert!(group.finished());
        assert!(!group.succeeded());
        assert_eq!(group.get_error(), Some(CompletionError::Aborted));
    }

    #[test]
    fn test_completion_group_tracks_all_completions() {
        // This test verifies the fix for the bug where CompletionGroup::add()
        // would skip successfully-finished completions. This caused problems
        // when code used drain() to move completions into a group, because
        // finished completions would be removed from the source but not tracked
        // by the group, effectively losing them.
        use crate::sync::atomic::{AtomicUsize, Ordering};

        let callback_count = Arc::new(AtomicUsize::new(0));
        let callback_count_clone = callback_count.clone();

        // Simulate the pattern: create multiple completions, complete some,
        // then add ALL of them to a group (like drain() would do)
        let mut completions = Vec::new();

        // Create 4 completions
        for _ in 0..4 {
            completions.push(Completion::new_write(|_| {}));
        }

        // Complete 2 of them before adding to group (simulate async completion)
        completions[0].complete(0);
        completions[2].complete(0);

        // Now create a group and add ALL completions (like drain() would do)
        let mut group = CompletionGroup::new(move |_| {
            callback_count_clone.fetch_add(1, Ordering::SeqCst);
        });

        // Add all completions to the group
        for c in &completions {
            group.add(c);
        }

        let group = group.build();

        // The group should track all 4 completions:
        // - c[0] and c[2] are already finished
        // - c[1] and c[3] are still pending
        // So the group should not be finished yet
        assert!(!group.finished());
        assert_eq!(callback_count.load(Ordering::SeqCst), 0);

        // Complete the first pending completion
        completions[1].complete(0);
        assert!(!group.finished());
        assert_eq!(callback_count.load(Ordering::SeqCst), 0);

        // Complete the last pending completion - now group should finish
        completions[3].complete(0);
        assert!(group.finished());
        assert!(group.succeeded());
        assert_eq!(callback_count.load(Ordering::SeqCst), 1);

        // Verify no errors
        assert!(group.get_error().is_none());
    }

    #[test]
    fn test_completion_group_with_all_finished_successfully() {
        // Edge case: all completions are already successfully finished
        // when added to the group. The group should complete immediately.
        use crate::sync::atomic::{AtomicBool, Ordering};

        let callback_called = Arc::new(AtomicBool::new(false));
        let callback_called_clone = callback_called.clone();

        let mut completions = Vec::new();

        // Create and immediately complete 3 completions
        for _ in 0..3 {
            let c = Completion::new_write(|_| {});
            c.complete(0);
            completions.push(c);
        }

        // Add all already-completed completions to group
        let mut group = CompletionGroup::new(move |_| {
            callback_called_clone.store(true, Ordering::SeqCst);
        });

        for c in &completions {
            group.add(c);
        }

        let group = group.build();

        // Group should be immediately finished since all completions were done
        assert!(group.finished());
        assert!(group.succeeded());
        assert!(callback_called.load(Ordering::SeqCst));
        assert!(group.get_error().is_none());
    }

    #[test]
    fn test_completion_group_nested() {
        use crate::sync::atomic::{AtomicUsize, Ordering};

        // Track callbacks at different levels
        let parent_called = Arc::new(AtomicUsize::new(0));
        let child1_called = Arc::new(AtomicUsize::new(0));
        let child2_called = Arc::new(AtomicUsize::new(0));

        // Create child group 1 with 2 completions
        let child1_called_clone = child1_called.clone();
        let mut child_group1 = CompletionGroup::new(move |_| {
            child1_called_clone.fetch_add(1, Ordering::SeqCst);
        });
        let c1 = Completion::new_write(|_| {});
        let c2 = Completion::new_write(|_| {});
        child_group1.add(&c1);
        child_group1.add(&c2);
        let child_group1 = child_group1.build();

        // Create child group 2 with 2 completions
        let child2_called_clone = child2_called.clone();
        let mut child_group2 = CompletionGroup::new(move |_| {
            child2_called_clone.fetch_add(1, Ordering::SeqCst);
        });
        let c3 = Completion::new_write(|_| {});
        let c4 = Completion::new_write(|_| {});
        child_group2.add(&c3);
        child_group2.add(&c4);
        let child_group2 = child_group2.build();

        // Create parent group containing both child groups
        let parent_called_clone = parent_called.clone();
        let mut parent_group = CompletionGroup::new(move |_| {
            parent_called_clone.fetch_add(1, Ordering::SeqCst);
        });
        parent_group.add(&child_group1);
        parent_group.add(&child_group2);
        let parent_group = parent_group.build();

        // Initially nothing should be finished
        assert!(!parent_group.finished());
        assert!(!child_group1.finished());
        assert!(!child_group2.finished());
        assert_eq!(parent_called.load(Ordering::SeqCst), 0);
        assert_eq!(child1_called.load(Ordering::SeqCst), 0);
        assert_eq!(child2_called.load(Ordering::SeqCst), 0);

        // Complete first completion in child group 1
        c1.complete(0);
        assert!(!child_group1.finished());
        assert!(!parent_group.finished());
        assert_eq!(child1_called.load(Ordering::SeqCst), 0);
        assert_eq!(parent_called.load(Ordering::SeqCst), 0);

        // Complete second completion in child group 1 - should finish child group 1
        c2.complete(0);
        assert!(child_group1.finished());
        assert!(child_group1.succeeded());
        assert_eq!(child1_called.load(Ordering::SeqCst), 1);

        // Parent should not be finished yet because child group 2 is still pending
        assert!(!parent_group.finished());
        assert_eq!(parent_called.load(Ordering::SeqCst), 0);

        // Complete first completion in child group 2
        c3.complete(0);
        assert!(!child_group2.finished());
        assert!(!parent_group.finished());
        assert_eq!(child2_called.load(Ordering::SeqCst), 0);
        assert_eq!(parent_called.load(Ordering::SeqCst), 0);

        // Complete second completion in child group 2 - should finish everything
        c4.complete(0);
        assert!(child_group2.finished());
        assert!(child_group2.succeeded());
        assert_eq!(child2_called.load(Ordering::SeqCst), 1);

        // Parent should now be finished
        assert!(parent_group.finished());
        assert!(parent_group.succeeded());
        assert_eq!(parent_called.load(Ordering::SeqCst), 1);
        assert!(parent_group.get_error().is_none());
    }

    #[test]
    fn test_completion_group_nested_with_error() {
        use crate::sync::atomic::{AtomicBool, Ordering};

        let parent_called = Arc::new(AtomicBool::new(false));
        let child_called = Arc::new(AtomicBool::new(false));

        // Create child group with 2 completions
        let child_called_clone = child_called.clone();
        let mut child_group = CompletionGroup::new(move |_| {
            child_called_clone.store(true, Ordering::SeqCst);
        });
        let c1 = Completion::new_write(|_| {});
        let c2 = Completion::new_write(|_| {});
        child_group.add(&c1);
        child_group.add(&c2);
        let child_group = child_group.build();

        // Create parent group containing child group and another completion
        let parent_called_clone = parent_called.clone();
        let mut parent_group = CompletionGroup::new(move |_| {
            parent_called_clone.store(true, Ordering::SeqCst);
        });
        let c3 = Completion::new_write(|_| {});
        parent_group.add(&child_group);
        parent_group.add(&c3);
        let parent_group = parent_group.build();

        // Complete child group with success
        c1.complete(0);
        c2.complete(0);
        assert!(child_group.finished());
        assert!(child_group.succeeded());
        assert!(child_called.load(Ordering::SeqCst));

        // Parent still pending
        assert!(!parent_group.finished());
        assert!(!parent_called.load(Ordering::SeqCst));

        // Complete c3 with error
        c3.error(CompletionError::Aborted);

        // Parent should finish with error
        assert!(parent_group.finished());
        assert!(!parent_group.succeeded());
        assert_eq!(parent_group.get_error(), Some(CompletionError::Aborted));
        assert!(parent_called.load(Ordering::SeqCst));
    }

    // Tests for individual completion success/failure status

    #[test]
    fn test_write_completion_pending_status() {
        let c = Completion::new_write(|_| {});

        // Pending completion should not be finished, succeeded, or failed
        assert!(!c.finished());
        assert!(!c.succeeded());
        assert!(!c.failed());
        assert!(c.get_error().is_none());
    }

    #[test]
    fn test_write_completion_success() {
        let c = Completion::new_write(|_| {});

        c.complete(42);

        assert!(c.finished());
        assert!(c.succeeded());
        assert!(!c.failed());
        assert!(c.get_error().is_none());
    }

    #[test]
    fn test_write_completion_failure() {
        let c = Completion::new_write(|_| {});

        c.error(CompletionError::Aborted);

        assert!(c.finished());
        assert!(!c.succeeded());
        assert!(c.failed());
        assert_eq!(c.get_error(), Some(CompletionError::Aborted));
    }

    #[test]
    fn test_read_completion_pending_status() {
        let buf = Arc::new(crate::Buffer::new_temporary(4096));
        let c = Completion::new_read(buf, |_| None);

        assert!(!c.finished());
        assert!(!c.succeeded());
        assert!(!c.failed());
        assert!(c.get_error().is_none());
    }

    #[test]
    fn test_read_completion_success() {
        let buf = Arc::new(crate::Buffer::new_temporary(4096));
        let c = Completion::new_read(buf, |_| None);

        c.complete(1024);

        assert!(c.finished());
        assert!(c.succeeded());
        assert!(!c.failed());
        assert!(c.get_error().is_none());
    }

    #[test]
    fn test_read_completion_failure() {
        let buf = Arc::new(crate::Buffer::new_temporary(4096));
        let c = Completion::new_read(buf, |_| None);

        c.error(CompletionError::Aborted);

        assert!(c.finished());
        assert!(!c.succeeded());
        assert!(c.failed());
        assert_eq!(c.get_error(), Some(CompletionError::Aborted));
    }

    #[test]
    fn test_sync_completion_pending_status() {
        let c = Completion::new_sync(|_| {});

        assert!(!c.finished());
        assert!(!c.succeeded());
        assert!(!c.failed());
        assert!(c.get_error().is_none());
    }

    #[test]
    fn test_sync_completion_success() {
        let c = Completion::new_sync(|_| {});

        c.complete(0);

        assert!(c.finished());
        assert!(c.succeeded());
        assert!(!c.failed());
        assert!(c.get_error().is_none());
    }

    #[test]
    fn test_sync_completion_failure() {
        let c = Completion::new_sync(|_| {});

        c.error(CompletionError::Aborted);

        assert!(c.finished());
        assert!(!c.succeeded());
        assert!(c.failed());
        assert_eq!(c.get_error(), Some(CompletionError::Aborted));
    }

    #[test]
    fn test_truncate_completion_pending_status() {
        let c = Completion::new_trunc(|_| {});

        assert!(!c.finished());
        assert!(!c.succeeded());
        assert!(!c.failed());
        assert!(c.get_error().is_none());
    }

    #[test]
    fn test_truncate_completion_success() {
        let c = Completion::new_trunc(|_| {});

        c.complete(0);

        assert!(c.finished());
        assert!(c.succeeded());
        assert!(!c.failed());
        assert!(c.get_error().is_none());
    }

    #[test]
    fn test_truncate_completion_failure() {
        let c = Completion::new_trunc(|_| {});

        c.error(CompletionError::Aborted);

        assert!(c.finished());
        assert!(!c.succeeded());
        assert!(c.failed());
        assert_eq!(c.get_error(), Some(CompletionError::Aborted));
    }

    #[test]
    fn test_yield_completion_status() {
        let c = Completion::new_yield();

        // Yield completions are always considered finished and succeeded
        assert!(c.finished());
        assert!(c.succeeded());
        assert!(!c.failed());
        assert!(c.get_error().is_none());
    }

    #[test]
    fn test_completion_abort() {
        let c = Completion::new_write(|_| {});

        c.abort();

        assert!(c.finished());
        assert!(!c.succeeded());
        assert!(c.failed());
        assert_eq!(c.get_error(), Some(CompletionError::Aborted));
    }

    #[test]
    fn test_completion_callback_receives_success_result() {
        use crate::sync::atomic::{AtomicI32, Ordering};

        let result_value = Arc::new(AtomicI32::new(-1));
        let result_value_clone = result_value.clone();

        let c = Completion::new_write(move |res| {
            if let Ok(val) = res {
                result_value_clone.store(val, Ordering::SeqCst);
            }
        });

        c.complete(42);

        assert_eq!(result_value.load(Ordering::SeqCst), 42);
        assert!(c.succeeded());
    }

    #[test]
    fn test_completion_callback_receives_error_result() {
        use crate::sync::atomic::{AtomicBool, Ordering};

        let got_error = Arc::new(AtomicBool::new(false));
        let got_error_clone = got_error.clone();

        let c = Completion::new_write(move |res| {
            if res.is_err() {
                got_error_clone.store(true, Ordering::SeqCst);
            }
        });

        c.error(CompletionError::Aborted);

        assert!(got_error.load(Ordering::SeqCst));
        assert!(c.failed());
    }

    #[test]
    fn test_completion_idempotent_complete() {
        // Completing a completion multiple times should only trigger the callback once
        use crate::sync::atomic::{AtomicUsize, Ordering};

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let c = Completion::new_write(move |_| {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
        });

        c.complete(1);
        c.complete(2);
        c.complete(3);

        // Callback should only be called once
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
        assert!(c.succeeded());
    }

    #[test]
    fn test_completion_idempotent_error() {
        // Erroring a completion multiple times should only trigger the callback once
        use crate::sync::atomic::{AtomicUsize, Ordering};

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let c = Completion::new_write(move |_| {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
        });

        c.error(CompletionError::Aborted);
        c.error(CompletionError::Aborted);
        c.complete(0); // Try completing after error

        // Callback should only be called once
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
        assert!(c.failed());
    }

    /// Re-runs the calling test alone in a child process bounded by `budget`,
    /// so a wait that never returns ends the child, not this test thread.
    fn run_current_test_in_child(child_env: &str, budget: std::time::Duration) {
        use std::io::Read;
        use std::process::{Command, Stdio};
        use std::time::Instant;

        let name = std::thread::current()
            .name()
            .expect("libtest names its test threads")
            .to_string();
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([name.as_str(), "--exact", "--nocapture", "--test-threads=1"])
            .env(child_env, "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        // Drain both pipes while polling, so a chatty child never blocks on a
        // full pipe and looks like a timeout.
        let drain = |mut pipe: Box<dyn Read + Send>| {
            std::thread::spawn(move || {
                let mut text = String::new();
                let _ = pipe.read_to_string(&mut text);
                text
            })
        };
        let stdout = drain(Box::new(child.stdout.take().unwrap()));
        let stderr = drain(Box::new(child.stderr.take().unwrap()));
        let deadline = Instant::now() + budget;
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break Some(status);
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        };
        // The child has exited or was killed, so both pipes are closed.
        let stdout = stdout.join().unwrap();
        let stderr = stderr.join().unwrap();
        let Some(status) = status else {
            panic!("{name}: child exceeded its {budget:?} budget\n{stdout}{stderr}");
        };
        assert!(
            status.success()
                && stdout.contains("running 1 test")
                && stdout.contains("test result: ok. 1 passed; 0 failed; 0 ignored"),
            "{name}: child failed ({status})\n{stdout}{stderr}"
        );
    }

    /// A peer that completes a child after `build` observed it unfinished but
    /// before `build` registered it with the group must still be counted;
    /// otherwise the group never finishes and every wait on it steps forever.
    #[test]
    fn completion_group_peer_completion_during_registration_is_not_lost() {
        use crate::IO;
        use std::sync::mpsc;
        use std::time::Duration;

        const CHILD: &str = "TURSO_COMPLETION_REGISTRATION_CHILD";
        if std::env::var_os(CHILD).is_none() {
            run_current_test_in_child(CHILD, Duration::from_secs(20));
            return;
        }

        let child_calls = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
        let child = |index: usize| {
            let calls = child_calls.clone();
            Completion::new_sync(move |_| {
                calls[index].fetch_add(1, Ordering::SeqCst);
            })
        };
        let (c0, c1) = (child(0), child(1));
        let group_calls = Arc::new(AtomicUsize::new(0));
        let mut group = CompletionGroup::new({
            let calls = group_calls.clone();
            move |_| {
                calls.fetch_add(1, Ordering::SeqCst);
            }
        });
        group.add(&c0);
        group.add(&c1);

        // The peer completes both children while `build` is paused between
        // observing c1 unfinished and registering it. Channels fix the order.
        let (paused_tx, paused_rx) = mpsc::channel::<()>();
        let (resume_tx, resume_rx) = mpsc::channel::<()>();
        let peer = {
            let (c0, c1) = (c0.clone(), c1.clone());
            std::thread::spawn(move || {
                if paused_rx.recv_timeout(Duration::from_secs(10)).is_err() {
                    return None;
                }
                c0.complete(0);
                c1.complete(0);
                // Witness after both completions returned, not from a callback.
                let finished = (c0.finished(), c1.finished());
                resume_tx.send(()).unwrap();
                Some(finished)
            })
        };
        let mut registrations = 0;
        let group = registration_hooks::with_before_child_register(
            move || {
                registrations += 1;
                if registrations == 2 {
                    paused_tx.send(()).unwrap();
                    if resume_rx.recv_timeout(Duration::from_secs(10)).is_err() {
                        panic!("completion_group_registration_rendezvous: the peer never resumed");
                    }
                }
            },
            || group.build(),
        );
        let finished = peer.join().unwrap();
        assert_eq!(
            finished,
            Some((true, true)),
            "completion_group_registration_rendezvous: build never paused before c1"
        );

        let outstanding = match &group.get_inner().completion_type {
            CompletionType::Group(g) => g.inner.outstanding.load(Ordering::SeqCst),
            _ => unreachable!(),
        };
        let (waited_tx, waited_rx) = mpsc::channel();
        {
            let group = group.clone();
            // Never joined: on a lost notification this thread steps forever.
            std::thread::spawn(move || {
                let _ = waited_tx.send(crate::MemoryIO::new().wait_for_completion(group));
            });
        }
        let waited = waited_rx.recv_timeout(Duration::from_secs(5));
        let calls = || {
            (
                child_calls[0].load(Ordering::SeqCst),
                child_calls[1].load(Ordering::SeqCst),
                group_calls.load(Ordering::SeqCst),
            )
        };
        let Ok(waited) = waited else {
            panic!(
                "completion_group_registration_deadline: wait_for_completion did not return \
                 within 5s; outstanding={outstanding} children_finished={finished:?} \
                 (child0, child1, group) callbacks={:?}",
                calls()
            );
        };
        waited.unwrap();
        assert!(group.finished());
        assert!(group.succeeded());
        assert_eq!(calls(), (1, 1, 1));
    }

    fn group_outstanding(group: &Completion) -> usize {
        match &group.get_inner().completion_type {
            CompletionType::Group(g) => g.inner.outstanding.load(Ordering::SeqCst),
            _ => unreachable!("not a group completion"),
        }
    }

    /// Counts the calls of every callback it hands out.
    #[derive(Clone, Default)]
    struct Calls(Arc<AtomicUsize>);

    impl Calls {
        fn write(&self) -> Completion {
            let calls = self.0.clone();
            Completion::new_write(move |_| {
                calls.fetch_add(1, Ordering::SeqCst);
            })
        }

        fn group(&self) -> CompletionGroup {
            let calls = self.0.clone();
            CompletionGroup::new(move |_| {
                calls.fetch_add(1, Ordering::SeqCst);
            })
        }

        fn get(&self) -> usize {
            self.0.load(Ordering::SeqCst)
        }
    }

    const RENDEZVOUS: std::time::Duration = std::time::Duration::from_secs(10);
    const CHILD: &str = "TURSO_COMPLETION_REGISTRATION_CHILD";

    /// One thread pauses until the driving thread resumes it. Both sides give
    /// up after `RENDEZVOUS` with the rendezvous name, so a broken order fails
    /// instead of hanging.
    fn rendezvous(name: &'static str) -> (Paused, Driver) {
        let (paused_tx, paused_rx) = std::sync::mpsc::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        (
            Paused {
                name,
                paused: paused_tx,
                resume: resume_rx,
            },
            Driver {
                name,
                paused: paused_rx,
                resume: resume_tx,
            },
        )
    }

    struct Paused {
        name: &'static str,
        paused: std::sync::mpsc::Sender<()>,
        resume: std::sync::mpsc::Receiver<()>,
    }

    impl Paused {
        fn pause(&self) {
            self.paused
                .send(())
                .expect("the driver outlives the rendezvous");
            if self.resume.recv_timeout(RENDEZVOUS).is_err() {
                panic!("{}: never resumed", self.name);
            }
        }
    }

    struct Driver {
        name: &'static str,
        paused: std::sync::mpsc::Receiver<()>,
        resume: std::sync::mpsc::Sender<()>,
    }

    impl Driver {
        fn wait_paused(&self) {
            if self.paused.recv_timeout(RENDEZVOUS).is_err() {
                panic!("{}: never paused", self.name);
            }
        }

        fn resume(&self) {
            self.resume
                .send(())
                .expect("the paused thread waits for resume");
        }
    }

    /// The other order of the registration race: the child's result is
    /// already visible when build registers it, but the completing thread has
    /// not reconciled with the group yet. Build must not count the child, and
    /// the completing thread must count it once when it resumes.
    #[test]
    fn completion_group_registration_before_reconcile_counts_the_child_once() {
        if std::env::var_os(CHILD).is_none() {
            run_current_test_in_child(CHILD, std::time::Duration::from_secs(20));
            return;
        }
        let (child_calls, group_calls) = (Calls::default(), Calls::default());
        let child = child_calls.write();
        let mut group = group_calls.group();
        group.add(&child);
        let (paused, driver) = rendezvous("completion_group_reconcile_rendezvous");
        let peer = {
            let child = child.clone();
            std::thread::spawn(move || {
                let mut published = 0;
                registration_hooks::with_hook(
                    move |point| {
                        if point == registration_hooks::HookPoint::AfterResultPublished {
                            published += 1;
                            if published == 1 {
                                paused.pause();
                            }
                        }
                    },
                    || child.complete(0),
                );
            })
        };
        driver.wait_paused();
        assert!(child.finished(), "the child's result is published");
        let group = group.build();
        assert!(
            !group.finished(),
            "build counted a child whose completer had not reconciled"
        );
        assert_eq!(group_outstanding(&group), 1);
        driver.resume();
        peer.join().expect("the completing thread finished");
        assert!(group.finished());
        assert!(group.succeeded());
        assert_eq!(group_outstanding(&group), 0);
        assert_eq!((child_calls.get(), group_calls.get()), (1, 1));
    }

    /// Every child finishes after its registration but before build gives up
    /// its own count. The group must not finish until build is done, and then
    /// exactly once.
    #[test]
    fn completion_group_children_finishing_before_build_ends_finish_it_once() {
        if std::env::var_os(CHILD).is_none() {
            run_current_test_in_child(CHILD, std::time::Duration::from_secs(20));
            return;
        }
        let (child_calls, group_calls) = (Calls::default(), Calls::default());
        let (c0, c1) = (child_calls.write(), child_calls.write());
        let mut group = group_calls.group();
        group.add(&c0);
        group.add(&c1);
        let (paused, driver) = rendezvous("completion_group_build_end_rendezvous");
        let peer = {
            let (c0, c1, group_calls) = (c0.clone(), c1.clone(), group_calls.clone());
            std::thread::spawn(move || {
                driver.wait_paused();
                c0.complete(0);
                c1.complete(0);
                let group_calls_while_building = group_calls.get();
                driver.resume();
                group_calls_while_building
            })
        };
        let mut registered = 0;
        let group = registration_hooks::with_hook(
            move |point| {
                if point == registration_hooks::HookPoint::AfterChildRegister {
                    registered += 1;
                    if registered == 2 {
                        paused.pause();
                    }
                }
            },
            || group.build(),
        );
        let group_calls_while_building = peer.join().expect("the peer finished");
        assert_eq!(
            group_calls_while_building, 0,
            "the group finished before build ended"
        );
        assert!(group.finished());
        assert!(group.succeeded());
        assert_eq!(group_outstanding(&group), 0);
        assert_eq!((child_calls.get(), group_calls.get()), (2, 1));
    }

    /// A child that failed before build no longer ends the group early: the
    /// others are still registered, the group waits for their IO and then
    /// finishes once with the first error, including a read callback's.
    #[test]
    fn completion_group_failed_child_does_not_skip_pending_children() {
        let group_calls = Calls::default();
        let short_read = CompletionError::ShortRead {
            page_idx: 1,
            expected: 4096,
            actual: 0,
        };
        let failed =
            Completion::new_read(Arc::new(crate::Buffer::new_temporary(4096)), move |_| {
                Some(short_read)
            });
        failed.complete(0);
        let pending = Completion::new_write(|_| {});
        let mut group = group_calls.group();
        group.add(&failed);
        group.add(&pending);
        let group = group.build();
        assert!(
            !group.finished(),
            "the group finished while a child's IO was pending"
        );
        assert_eq!(group_outstanding(&group), 1);
        assert_eq!(group_calls.get(), 0);
        pending.error(CompletionError::ShortWrite);
        assert!(group.finished());
        assert_eq!(group.get_error(), Some(short_read), "the first error wins");
        assert_eq!(group_outstanding(&group), 0);
        assert_eq!(group_calls.get(), 1);
    }

    /// Finishing a child again (complete, error or abort) before or after its
    /// registration changes nothing: its group counts it once, with its first
    /// result, and a finished group ignores being finished again.
    #[test]
    fn completion_group_counts_a_repeatedly_finished_child_once() {
        let (child_calls, group_calls) = (Calls::default(), Calls::default());
        let (before, during, after) = (
            child_calls.write(),
            child_calls.write(),
            child_calls.write(),
        );
        before.complete(0);
        before.abort();
        let mut group = group_calls.group();
        group.add(&before);
        group.add(&during);
        group.add(&after);
        let group = group.build();
        before.abort();
        during.abort();
        during.abort();
        during.complete(0);
        assert!(!group.finished());
        assert_eq!(group_outstanding(&group), 1);
        after.complete(0);
        after.error(CompletionError::ShortWrite);
        group.complete(0);
        assert!(group.finished());
        assert_eq!(group.get_error(), Some(CompletionError::Aborted));
        assert_eq!(group_outstanding(&group), 0);
        assert_eq!((child_calls.get(), group_calls.get()), (3, 1));
    }

    /// A group's finished, succeeded and error state come from one published
    /// result: inside its callback the group is not finished yet, and once
    /// the callback returned it reports the callback's result.
    #[test]
    fn completion_group_is_not_finished_until_its_callback_returned() {
        type Seen = (
            Result<i32, CompletionError>,
            bool,
            bool,
            Option<CompletionError>,
        );
        let slot: Arc<OnceLock<Completion>> = Arc::new(OnceLock::new());
        let seen: Arc<Mutex<Option<Seen>>> = Arc::new(Mutex::new(None));
        let mut group = CompletionGroup::new({
            let (slot, seen) = (slot.clone(), seen.clone());
            move |result| {
                let group = slot.get().expect("the test stores the group after build");
                *seen.lock() = Some((
                    result,
                    group.finished(),
                    group.succeeded(),
                    group.get_error(),
                ));
            }
        });
        let child = Completion::new_write(|_| {});
        group.add(&child);
        let group = group.build();
        slot.set(group.clone()).expect("stored once");
        child.error(CompletionError::ShortWrite);
        assert_eq!(
            *seen.lock(),
            Some((Err(CompletionError::ShortWrite), false, false, None))
        );
        assert!(group.finished());
        assert!(!group.succeeded());
        assert_eq!(group.get_error(), Some(CompletionError::ShortWrite));
    }

    /// A completion belongs to one group only, even after it finished: build
    /// registers finished children too.
    #[test]
    #[should_panic(expected = "completion can only be linked once")]
    fn completion_group_rejects_a_child_already_in_another_group() {
        let child = Completion::new_write(|_| {});
        child.complete(0);
        let mut first = CompletionGroup::new(|_| {});
        first.add(&child);
        assert!(first.build().finished());
        let mut second = CompletionGroup::new(|_| {});
        second.add(&child);
        let _ = second.build();
    }

    /// A nested group that finishes while its parent's build is about to
    /// register it is counted once, like any other child.
    #[test]
    fn completion_group_nested_group_finishing_during_registration_is_counted() {
        if std::env::var_os(CHILD).is_none() {
            run_current_test_in_child(CHILD, std::time::Duration::from_secs(20));
            return;
        }
        let (inner_calls, outer_calls) = (Calls::default(), Calls::default());
        let leaf = Completion::new_write(|_| {});
        let mut inner = inner_calls.group();
        inner.add(&leaf);
        let inner = inner.build();
        let sibling = Completion::new_write(|_| {});
        let mut outer = outer_calls.group();
        outer.add(&inner);
        outer.add(&sibling);
        let (paused, driver) = rendezvous("completion_group_nested_rendezvous");
        let peer = {
            let (leaf, inner) = (leaf.clone(), inner.clone());
            std::thread::spawn(move || {
                driver.wait_paused();
                leaf.complete(0);
                let inner_finished = inner.finished();
                driver.resume();
                inner_finished
            })
        };
        let mut seen = 0;
        let outer = registration_hooks::with_before_child_register(
            move || {
                seen += 1;
                if seen == 1 {
                    paused.pause();
                }
            },
            || outer.build(),
        );
        assert!(
            peer.join().expect("the peer finished"),
            "completion_group_nested_rendezvous: the nested group did not finish while paused"
        );
        assert!(!outer.finished());
        assert_eq!(group_outstanding(&outer), 1);
        sibling.complete(0);
        assert!(outer.finished());
        assert!(outer.succeeded());
        assert_eq!(group_outstanding(&outer), 0);
        assert_eq!((inner_calls.get(), outer_calls.get()), (1, 1));
    }

    /// Cancelling a group's children while build is registering them, or
    /// before build, finishes the group once with the abort after every
    /// child is counted.
    #[test]
    fn completion_group_children_cancelled_during_build_finish_it_once() {
        if std::env::var_os(CHILD).is_none() {
            run_current_test_in_child(CHILD, std::time::Duration::from_secs(20));
            return;
        }
        let (child_calls, group_calls) = (Calls::default(), Calls::default());
        let (c0, c1) = (child_calls.write(), child_calls.write());
        let mut group = group_calls.group();
        group.add(&c0);
        group.add(&c1);
        let (paused, driver) = rendezvous("completion_group_cancel_rendezvous");
        let peer = {
            let children = [c0.clone(), c1.clone()];
            std::thread::spawn(move || {
                driver.wait_paused();
                for child in &children {
                    child.abort();
                }
                driver.resume();
            })
        };
        let mut seen = 0;
        let group = registration_hooks::with_before_child_register(
            move || {
                seen += 1;
                if seen == 2 {
                    paused.pause();
                }
            },
            || group.build(),
        );
        peer.join().expect("the peer finished");
        assert!(group.finished());
        assert_eq!(group.get_error(), Some(CompletionError::Aborted));
        assert_eq!(group_outstanding(&group), 0);
        assert_eq!((child_calls.get(), group_calls.get()), (2, 1));

        let cancelled_calls = Calls::default();
        let mut cancelled = cancelled_calls.group();
        cancelled.add(&Completion::new_write(|_| {}));
        cancelled.add(&Completion::new_write(|_| {}));
        cancelled.cancel();
        let cancelled = cancelled.build();
        assert!(cancelled.finished());
        assert_eq!(cancelled.get_error(), Some(CompletionError::Aborted));
        assert_eq!(group_outstanding(&cancelled), 0);
        assert_eq!(cancelled_calls.get(), 1);
    }
}
