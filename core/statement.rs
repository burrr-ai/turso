use std::{
    borrow::Cow,
    num::NonZero,
    ops::Deref,
    sync::{atomic::Ordering, Arc},
    task::Waker,
    time::Duration,
};

use tracing::{instrument, Level};
use turso_parser::{
    ast::{fmt::ToTokens, Cmd},
    parser::Parser,
};

use crate::{
    busy::BusyHandlerState,
    parameters,
    schema::Trigger,
    stats::refresh_analyze_stats,
    storage::pager::CommitFinality,
    translate::{self, display::PlanContext, emitter::TransactionMode, plan::BitSet},
    vdbe::{
        self,
        explain::{EXPLAIN_COLUMNS_TYPE, EXPLAIN_QUERY_PLAN_COLUMNS_TYPE},
    },
    LimboError, MvStore, Pager, QueryMode, Result, TransactionState, Value, EXPLAIN_COLUMNS,
    EXPLAIN_QUERY_PLAN_COLUMNS,
};

type ProgramExecutionState = vdbe::ProgramExecutionState;
type Row = vdbe::Row;
type StepResult = vdbe::StepResult;

/// Classifies how a [`Statement`] participates in connection-level lifecycle
/// and active-statement accounting.
///
/// Use [`StatementOrigin::Root`] for ordinary top-level statements prepared on
/// behalf of the user. Root statements are the only statements that count
/// toward `Connection::n_active_root_statements` once execution begins, which
/// is the SQLite-compatible notion of "another SQL statement in progress" used
/// by operations like `VACUUM`.
///
/// Use [`StatementOrigin::InternalHelper`] when the engine prepares and runs a
/// separate helper statement on the same connection, for example helper SQL in
/// schema parsing or CDC setup. This is separately prepared SQL with its own
/// `prepare`/`step`/`reset`/`drop` lifecycle, but it is owned by a parent root
/// statement, so it stays nested and does not count as another root statement.
///
/// Use [`StatementOrigin::Subprogram`] only for bytecode subprograms that are
/// already compiled into a parent statement and entered through `OP_Program`,
/// such as trigger or foreign-key actions. This is not separately prepared SQL;
/// it is embedded child bytecode execution inside the parent statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StatementOrigin {
    Root,
    InternalHelper,
    Subprogram,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatementStatusCounter {
    FullscanStep,
    Sort,
    VmStep,
    Reprepare,
    RowsRead,
    RowsWritten,
}

impl StatementOrigin {
    pub(crate) const fn needs_nested_guard(self) -> bool {
        matches!(self, Self::InternalHelper)
    }
}

/// Identifies one execution of a statement and the connection transaction
/// it ran in. The counters are opaque: only equality is meaningful.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TxnIdentity {
    /// The connection, unique within the process.
    pub connection_generation: u64,
    /// The connection transaction this execution committed, rolled back or
    /// ran in.
    pub transaction_generation: u64,
    /// This execution of the statement; 0 before it starts.
    pub root_generation: u64,
}

/// How far the commit of a statement's execution got.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitPhase {
    /// The statement has not been stepped.
    NotStarted,
    /// No commit driven by this execution was published.
    NotPublished,
    /// The commit this execution drove is published: durable and visible.
    Published {
        /// The WAL transaction count it reached, when the WAL tracks one.
        transaction_count: Option<u64>,
        max_frame: u64,
    },
}

/// How a statement's execution ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootTerminal {
    /// It ran to completion.
    Done,
    /// It ended before any commit append was submitted; its writes were
    /// rolled back.
    RolledBack,
    /// Its commit append was submitted and then failed: the WAL may or may
    /// not hold the commit.
    Unknown,
    /// It ended with an error after its commit was published.
    Failed,
}

/// Where an execution first observed a cancellation (interrupt, query
/// deadline or progress handler).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelObserved {
    /// Before its commit append was submitted: the execution was aborted.
    BeforeIrreversibleCommit,
    /// After the commit append was submitted, before it was published: the
    /// commit ran on.
    AfterSubmission,
    /// After the commit was published: its auto-checkpoint was given up.
    AfterPublication,
}

/// Evidence about one execution of a statement, from
/// [`Statement::outcome_snapshot`] or [`Statement::reset_with_outcome`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatementOutcome {
    pub identity: TxnIdentity,
    pub phase: CommitPhase,
    /// None while the execution can still continue.
    pub terminal: Option<RootTerminal>,
    pub cancel_observed: Option<CancelObserved>,
    /// The published commit's auto-checkpoint failed and was given up; the
    /// commit stands.
    pub checkpoint_failure: Option<String>,
}

pub struct Statement {
    pub(crate) program: vdbe::Program,
    state: vdbe::ProgramState,
    pager: Arc<Pager>,
    /// indicates if the statement is a NORMAL/EXPLAIN/EXPLAIN QUERY PLAN
    query_mode: QueryMode,
    /// Flag to show if the statement was busy
    busy: bool,
    /// Busy handler state for tracking invocations and timeouts
    busy_handler_state: Option<BusyHandlerState>,
    /// Per-execution timeout override for this statement.
    /// - `None`: use connection default
    /// - `Some(Some(duration))`: override with a query-specific timeout
    /// - `Some(None)`: disable timeout for this execution
    query_timeout_override: Option<Option<Duration>>,
    /// True once step() has returned Row for a write statement (INSERT/UPDATE/DELETE
    /// with RETURNING). With ephemeral-buffered RETURNING, the first Row proves all
    /// DML completed — only the scan-back remains. Used by reset_internal to decide
    /// commit vs rollback when a statement is abandoned.
    has_returned_row: bool,
    /// Byte offset in the original SQL string where this statement ends.
    /// Used by sqlite3_prepare_v2 to set the *pzTail output parameter.
    tail_offset: usize,
    origin: StatementOrigin,
    /// True once this root statement has started executing and incremented
    /// `Connection::n_active_root_statements`.
    counted_as_active_root: bool,
    /// True if this statement called `Connection::start_nested()` during
    /// construction and therefore must call `end_nested()` on drop.
    nested_guard_active: bool,
    /// This execution's identity on the connection; 0 until it starts.
    root_generation: u64,
    /// The statement was stepped in this execution.
    stepped: bool,
    /// How this execution ended, once it did.
    terminal: Option<RootTerminal>,
}

crate::assert::assert_send_sync!(Statement);

impl std::fmt::Debug for Statement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Statement").finish()
    }
}

impl Statement {
    pub fn new(
        program: vdbe::Program,
        pager: Arc<Pager>,
        query_mode: QueryMode,
        tail_offset: usize,
    ) -> Self {
        Self::new_with_origin(
            program,
            pager,
            query_mode,
            tail_offset,
            StatementOrigin::Root,
            false,
        )
    }

    #[turso_macros::trace_stack]
    pub(crate) fn new_with_origin(
        program: vdbe::Program,
        pager: Arc<Pager>,
        query_mode: QueryMode,
        tail_offset: usize,
        origin: StatementOrigin,
        nested_guard_active: bool,
    ) -> Self {
        let (max_registers, cursor_count) = match query_mode {
            QueryMode::Normal => (program.max_registers, program.cursor_ref.len()),
            QueryMode::Explain => (EXPLAIN_COLUMNS.len(), 0),
            QueryMode::ExplainQueryPlan => (EXPLAIN_QUERY_PLAN_COLUMNS.len(), 0),
        };
        let state = vdbe::ProgramState::new(max_registers, cursor_count);
        Self {
            program,
            state,
            pager,
            query_mode,
            busy: false,
            busy_handler_state: None,
            query_timeout_override: None,
            has_returned_row: false,
            tail_offset,
            origin,
            counted_as_active_root: false,
            nested_guard_active,
            root_generation: 0,
            stepped: false,
            terminal: None,
        }
    }

    pub fn tail_offset(&self) -> usize {
        self.tail_offset
    }

    pub fn get_trigger(&self) -> Option<Arc<Trigger>> {
        self.program.trigger.clone()
    }

    pub fn get_query_mode(&self) -> QueryMode {
        self.query_mode
    }

    pub fn get_program(&self) -> &vdbe::Program {
        &self.program
    }

    pub fn get_pager(&self) -> &Arc<Pager> {
        &self.pager
    }

    pub fn n_change(&self) -> i64 {
        self.state
            .n_change
            .load(crate::sync::atomic::Ordering::SeqCst)
    }

    pub fn set_mv_tx(&mut self, mv_tx: Option<(u64, TransactionMode)>) {
        self.program.connection.set_mv_tx(mv_tx);
    }

    pub fn interrupt(&mut self) {
        self.state.interrupt();
    }

    /// Sets a per-execution timeout override for this statement.
    ///
    /// - `None`: use connection default
    /// - `Some(Some(duration))`: use query-specific timeout
    /// - `Some(None)`: disable timeout for this execution
    pub fn set_query_timeout_override(&mut self, timeout: Option<Option<Duration>>) {
        self.query_timeout_override = timeout;
    }

    pub fn execution_state(&self) -> ProgramExecutionState {
        self.state.execution_state
    }

    /// Statement metrics accumulated across executions of this prepared
    /// statement. Includes subprogram work.
    pub fn metrics(&self) -> vdbe::metrics::StatementMetrics {
        self.state.metrics()
    }

    pub fn reset_metrics(&mut self) {
        self.state.reset_metrics();
    }

    pub fn stmt_status(&self, counter: StatementStatusCounter) -> u64 {
        let metrics = self.metrics();
        match counter {
            StatementStatusCounter::FullscanStep => metrics.fullscan_steps,
            StatementStatusCounter::Sort => metrics.sort_operations,
            StatementStatusCounter::VmStep => metrics.insn_executed,
            StatementStatusCounter::Reprepare => metrics.reprepares,
            StatementStatusCounter::RowsRead => metrics.rows_read,
            StatementStatusCounter::RowsWritten => metrics.rows_written,
        }
    }

    pub fn reset_stmt_status(&mut self, counter: StatementStatusCounter) {
        self.state.reset_stmt_status(counter);
    }

    pub fn mv_store(&self) -> impl Deref<Target = Option<Arc<MvStore>>> {
        self.program.connection.mv_store()
    }

    /// Take the pending IO completions from this statement.
    /// Returns None if no IO is pending.
    /// This is used by async state machines that need to yield the completions.
    pub fn take_io_completions(&mut self) -> Option<crate::types::IOCompletions> {
        self.state.io_completions.take()
    }

    fn arm_query_timeout_if_needed(&mut self) {
        if !matches!(self.state.execution_state, ProgramExecutionState::Init)
            || self.state.query_deadline.is_some()
        {
            return;
        }
        let timeout = match self.query_timeout_override {
            Some(timeout_override) => timeout_override,
            None => {
                let connection_timeout = self.program.connection.get_query_timeout();
                if connection_timeout.is_zero() {
                    None
                } else {
                    Some(connection_timeout)
                }
            }
        };
        let Some(timeout) = timeout else {
            return;
        };
        self.state.query_deadline = Some(self.pager.io.current_time_monotonic() + timeout);
    }

    fn release_active_root_if_counted(&mut self) {
        if self.counted_as_active_root {
            let previous = self
                .program
                .connection
                .n_active_root_statements
                .fetch_sub(1, Ordering::SeqCst);
            if previous == 1 {
                self.program.connection.clear_interrupt_if_idle();
            }
            self.counted_as_active_root = false;
        }
    }

    /// Starts this execution before its first step: gives it its identity
    /// and, for a root statement, counts it as active on the connection.
    ///
    /// `Connection::interrupt` ignores a connection with no active root
    /// statement, so an interrupt delivered between preparing a statement and
    /// its first step is lost. A caller that may interrupt calls this first;
    /// the first step then observes the interrupt and rolls the execution
    /// back. A step does the same on its own; calling this again before a
    /// reset does nothing.
    pub fn activate_root(&mut self) {
        if self.root_generation == 0 {
            self.root_generation = self.program.connection.next_root_generation();
        }
        if !self.counted_as_active_root && matches!(self.origin, StatementOrigin::Root) {
            self.program
                .connection
                .n_active_root_statements
                .fetch_add(1, Ordering::SeqCst);
            self.counted_as_active_root = true;
        }
    }

    fn _step(&mut self, waker: Option<&Waker>) -> Result<StepResult> {
        self.activate_root();
        self.stepped = true;
        if matches!(self.state.execution_state, ProgramExecutionState::Init) {
            if self.program.connection.mvcc_enabled() {
                // MVCC checkpoints can publish internal schema roots without changing
                // SQLite's schema cookie, so refresh before deciding whether to reprepare.
                self.program.connection.maybe_update_schema();
            }
            if !self
                .program
                .prepare_context
                .matches_connection(&self.program.connection)
            {
                if let Err(err) = self.reprepare() {
                    self.release_active_root_if_counted();
                    return Err(err);
                }
            }
        }

        self.arm_query_timeout_if_needed();

        // If we're waiting for a busy handler timeout, check if we can proceed
        if let Some(busy_state) = self.busy_handler_state.as_ref() {
            if self.pager.io.current_time_monotonic() < busy_state.timeout() {
                // Yield the query as the timeout has not been reached yet
                if let Some(waker) = waker {
                    waker.wake_by_ref();
                }
                return Ok(StepResult::IO);
            }
        }

        const MAX_SCHEMA_RETRY: usize = 50;
        let mut res = self
            .program
            .step(&mut self.state, &self.pager, self.query_mode, waker);
        for attempt in 0..MAX_SCHEMA_RETRY {
            // Only reprepare if we still need to update schema
            if !matches!(res, Err(LimboError::SchemaUpdated)) {
                break;
            }
            // In a write transaction, reprepare may not help (e.g. cross-process
            // schema change where the in-memory schema hasn't been refreshed from
            // disk). Allow a few retries for the in-process case where reprepare
            // *can* resolve the issue, but bail early to avoid burning 50 attempts.
            if attempt >= 2
                && !self.program.connection.get_auto_commit()
                && matches!(
                    self.program.connection.get_tx_state(),
                    TransactionState::Write { .. } | TransactionState::PendingUpgrade { .. }
                )
            {
                break;
            }
            tracing::debug!("reprepare: attempt={}", attempt);
            if let Err(err) = self.reprepare() {
                self.release_active_root_if_counted();
                return Err(err);
            }
            res = self
                .program
                .step(&mut self.state, &self.pager, self.query_mode, waker);
        }

        // Aggregate metrics when statement completes
        if matches!(res, Ok(StepResult::Done)) {
            self.program
                .connection
                .metrics
                .write()
                .record_statement(&self.metrics());
            self.busy = false;
            self.busy_handler_state = None; // Reset busy state on completion
            self.state.query_deadline = None;

            // After ANALYZE completes, refresh in-memory stats so planners can use them.
            let sql = self.program.sql.trim_start().as_bytes();
            if sql.len() >= 7 && sql[..7].eq_ignore_ascii_case(b"ANALYZE") {
                refresh_analyze_stats(&self.program.connection);
            }
        } else {
            self.busy = true;
        }

        // Handle busy result by invoking the busy handler
        if matches!(res, Ok(StepResult::Busy)) {
            let now = self.pager.io.current_time_monotonic();
            let handler = self.program.connection.get_busy_handler();

            // Initialize or get existing busy handler state
            let busy_state = self
                .busy_handler_state
                .get_or_insert_with(|| BusyHandlerState::new(now));

            // Invoke the busy handler to determine if we should retry
            if busy_state.invoke(&handler, now) {
                // Handler says retry, yield with IO to wait for timeout
                if let Some(waker) = waker {
                    waker.wake_by_ref();
                }
                res = Ok(StepResult::IO);
                #[cfg(shuttle)]
                crate::thread::spin_loop();
            }
            // else: Handler says stop, res stays as Busy
        }

        // Track when a write statement yields its first Row. With ephemeral-buffered
        // RETURNING, this proves all DML completed — only the scan-back remains.
        if matches!(res, Ok(StepResult::Row))
            && self.query_mode == QueryMode::Normal
            && self.program.change_cnt_on
            && !self.program.result_columns.is_empty()
        {
            self.has_returned_row = true;
        }

        if self.counted_as_active_root
            && (matches!(res, Ok(StepResult::Done | StepResult::Interrupt)) || res.is_err())
        {
            self.release_active_root_if_counted();
        }

        let ended = matches!(res, Ok(StepResult::Done | StepResult::Interrupt)) || res.is_err();
        self.fix_transaction_generation(ended);
        match &res {
            Ok(StepResult::Done) => self.terminal = Some(RootTerminal::Done),
            // An interrupt is returned only before the commit append is
            // submitted; the execution was rolled back.
            Ok(StepResult::Interrupt) => self.terminal = Some(RootTerminal::RolledBack),
            Ok(_) => {}
            Err(_) => self.terminal = Some(self.terminal_after_error(false)),
        }

        res
    }

    /// Fixes the transaction this execution ran in once it is known: while
    /// the execution runs inside one, or when it ends.
    fn fix_transaction_generation(&mut self, ended: bool) {
        let connection = &self.program.connection;
        if self.state.outcome.transaction_generation.is_none()
            && (ended || connection.get_tx_state() != TransactionState::None)
        {
            self.state.outcome.transaction_generation = Some(connection.transaction_generation());
        }
    }

    /// How an execution that failed, or was torn down unfinished, ended.
    /// `abandoned` is true for a teardown that raised no error.
    fn terminal_after_error(&self, abandoned: bool) -> RootTerminal {
        let outcome = &self.state.outcome;
        if outcome.append_failed {
            RootTerminal::Unknown
        } else if outcome.publication.is_some()
            || self.pager.commit_finality() == CommitFinality::Published
        {
            // A published commit stands; a teardown that raised no error
            // finished it.
            if abandoned {
                RootTerminal::Done
            } else {
                RootTerminal::Failed
            }
        } else {
            RootTerminal::RolledBack
        }
    }

    /// Evidence about this execution: its identity, how far its commit got,
    /// how it ended, and where it first observed a cancellation.
    pub fn outcome_snapshot(&self) -> StatementOutcome {
        let outcome = &self.state.outcome;
        StatementOutcome {
            identity: TxnIdentity {
                connection_generation: self.program.connection.generation(),
                // 0 until the execution's transaction is known; never the
                // connection's current one, which a later transaction moves.
                transaction_generation: outcome.transaction_generation.unwrap_or(0),
                root_generation: self.root_generation,
            },
            phase: match outcome.publication {
                _ if !self.stepped => CommitPhase::NotStarted,
                Some(publication) => CommitPhase::Published {
                    transaction_count: publication.transaction_count,
                    max_frame: publication.max_frame,
                },
                None => CommitPhase::NotPublished,
            },
            terminal: self.terminal,
            cancel_observed: outcome.cancel_observed,
            checkpoint_failure: outcome.checkpoint_failure.clone(),
        }
    }

    #[inline]
    pub fn step(&mut self) -> Result<StepResult> {
        self._step(None)
    }

    #[inline]
    pub fn step_with_waker(&mut self, waker: &Waker) -> Result<StepResult> {
        self._step(Some(waker))
    }

    /// Fast step for trigger/FK subprograms: skips reprepare checks, timeout
    /// arming, busy handler, metrics recording, and schema retry.
    /// The parent statement handles all of those concerns.
    #[inline]
    pub fn step_subprogram(&mut self) -> Result<StepResult> {
        self.program
            .step(&mut self.state, &self.pager, self.query_mode, None)
    }

    pub fn run_ignore_rows(&mut self) -> Result<()> {
        loop {
            match self.step()? {
                vdbe::StepResult::Done => return Ok(()),
                vdbe::StepResult::IO => self.pager.io.step()?,
                vdbe::StepResult::Row => continue,
                vdbe::StepResult::Interrupt | vdbe::StepResult::Busy => {
                    return Err(LimboError::Busy)
                }
            }
        }
    }

    pub fn run_collect_rows(&mut self) -> Result<Vec<Vec<Value>>> {
        let mut values = Vec::new();
        loop {
            match self.step()? {
                vdbe::StepResult::Done => return Ok(values),
                vdbe::StepResult::IO => self.pager.io.step()?,
                vdbe::StepResult::Row => {
                    values.push(self.row().unwrap().get_values().cloned().collect());
                    continue;
                }
                vdbe::StepResult::Interrupt | vdbe::StepResult::Busy => {
                    return Err(LimboError::Busy)
                }
            }
        }
    }

    /// Blocks execution, advances IO, and runs to completion of the statement
    pub fn run_with_row_callback(
        &mut self,
        mut func: impl FnMut(&Row) -> Result<()>,
    ) -> Result<()> {
        loop {
            match self.step()? {
                vdbe::StepResult::Done => break,
                vdbe::StepResult::IO => self.pager.io.step()?,
                vdbe::StepResult::Row => {
                    func(self.row().expect("row should be present"))?;
                }
                vdbe::StepResult::Interrupt => return Err(LimboError::Interrupt),
                vdbe::StepResult::Busy => return Err(LimboError::Busy),
            }
        }
        Ok(())
    }

    /// Blocks execution, advances IO, and stops at any StepResult except IO
    /// You can optionally pass a handler to run after IO is advanced
    pub fn run_one_step_blocking(
        &mut self,
        mut pre_io_func: impl FnMut() -> Result<()>,
        mut post_io_func: impl FnMut() -> Result<()>,
    ) -> Result<Option<&Row>> {
        let result = loop {
            match self.step()? {
                vdbe::StepResult::Done => break None,
                vdbe::StepResult::IO => {
                    pre_io_func()?;
                    self.pager.io.step()?;
                    post_io_func()?;
                }
                vdbe::StepResult::Row => break Some(self.row().expect("row should be present")),
                vdbe::StepResult::Interrupt => return Err(LimboError::Interrupt),
                vdbe::StepResult::Busy => return Err(LimboError::Busy),
            }
        };
        Ok(result)
    }

    #[instrument(skip_all, level = Level::DEBUG)]
    fn reprepare(&mut self) -> Result<()> {
        tracing::trace!("repreparing statement");
        let conn = self.program.connection.clone();
        let main_pager = conn.pager.load().clone();

        // SchemaUpdated bypasses the normal abort rollback path, so in
        // autocommit mode we must unwind any implicit transaction state here
        // before reparsing. This must clear both pager locks and MVCC tx ids;
        // otherwise the retried statement can stack a fresh snapshot on top of
        // leaked transaction state from the failed attempt.
        let attached_leaked = conn.with_all_attached_pagers_with_index(|pagers| {
            pagers
                .iter()
                .any(|(_, pager)| pager.holds_write_lock() || pager.holds_read_lock())
        });
        let has_implicit_txn_state = conn.get_tx_state() != TransactionState::None
            || conn.get_mv_tx().is_some()
            || conn.next_attached_mv_tx().is_some()
            || attached_leaked
            || self.state.auto_txn_cleanup != vdbe::TxnCleanup::None;
        if conn.get_auto_commit() && has_implicit_txn_state {
            conn.rollback_current_txn_state(&main_pager, true);
            self.state.auto_txn_cleanup = vdbe::TxnCleanup::None;
        }
        if conn.get_auto_commit() && !conn.schema_reparse_in_progress() {
            conn.maybe_reparse_schema()?;
        }

        // End transactions on attached database pagers so they get a fresh view
        // of the database. Without this, the pager would still see the old page 1
        // with the stale schema cookie, causing an infinite SchemaUpdated loop.
        // SchemaUpdated can occur at different points in the Transaction opcode,
        // so the attached pager may or may not hold locks at this point.
        let attached_db_ids: BitSet = self
            .program
            .prepared
            .write_databases
            .iter()
            .chain(self.program.prepared.read_databases.iter())
            .filter(|&id| id != crate::MAIN_DB_ID)
            .collect();
        for db_id in &attached_db_ids {
            // Discard any connection-local schema changes for this non-main DB
            // (temp or attached) so the re-translate reads the committed schema.
            conn.database_schemas().write().remove(&db_id);
            if db_id == crate::TEMP_DB_ID && conn.temp.database.read().is_none() {
                continue;
            }
            let pager = conn.get_pager_from_database_index(&db_id)?;
            if pager.holds_read_lock() {
                pager.rollback_attached();
            }
        }

        // Refresh from shared schema only when shared is newer; this preserves a
        // connection-local schema that is ahead of shared. An MVCC checkpoint can
        // publish new btree roots without bumping the schema cookie, so
        // same-version reprepare still refreshes it.
        conn.refresh_schema_from_shared_for_reprepare();
        let new_program = {
            let mut parser = Parser::new(self.program.sql.as_bytes());
            let cmd = parser.next_cmd()?;
            let cmd = cmd.expect("Same SQL string should be able to be parsed");

            let syms = conn.syms.read();
            let mode = self.query_mode;
            #[cfg(debug_assertions)]
            crate::turso_assert_eq!(QueryMode::new(&cmd), mode);
            let (Cmd::Stmt(stmt) | Cmd::Explain(stmt) | Cmd::ExplainQueryPlan(stmt)) = cmd;
            let schema = conn.schema.read().clone();
            translate::translate(
                &schema,
                stmt,
                self.pager.clone(),
                conn.clone(),
                &syms,
                mode,
                &self.program.sql,
            )?
        };

        // Save parameters before they are reset
        let parameters = std::mem::take(&mut self.state.parameters);
        let (max_registers, cursor_count) = match self.query_mode {
            QueryMode::Normal => (new_program.max_registers, new_program.cursor_ref.len()),
            QueryMode::Explain => (EXPLAIN_COLUMNS.len(), 0),
            QueryMode::ExplainQueryPlan => (EXPLAIN_QUERY_PLAN_COLUMNS.len(), 0),
        };
        // Repreparing a root statement must not make it disappear from
        // `n_active_root_statements` while it is still logically in progress.
        self.reset_internal(
            Some(max_registers),
            Some(cursor_count),
            self.counted_as_active_root,
        )
        .1?;
        self.state.metrics.reprepares = self.state.metrics.reprepares.saturating_add(1);
        self.program = new_program;
        // Load the parameters back into the state
        self.state.parameters = parameters;
        Ok(())
    }

    pub fn num_columns(&self) -> usize {
        match self.query_mode {
            QueryMode::Normal => self.program.result_columns.len(),
            QueryMode::Explain => EXPLAIN_COLUMNS.len(),
            QueryMode::ExplainQueryPlan => EXPLAIN_QUERY_PLAN_COLUMNS.len(),
        }
    }

    pub fn get_column_name(&self, idx: usize) -> Cow<'_, str> {
        if self.query_mode == QueryMode::Explain {
            return Cow::Owned(EXPLAIN_COLUMNS.get(idx).expect("No column").to_string());
        }
        if self.query_mode == QueryMode::ExplainQueryPlan {
            return Cow::Owned(
                EXPLAIN_QUERY_PLAN_COLUMNS
                    .get(idx)
                    .expect("No column")
                    .to_string(),
            );
        }
        match self.query_mode {
            QueryMode::Normal => {
                let column = &self.program.result_columns.get(idx).expect("No column");

                // 1. Explicit alias (AS clause) or SELECT * expansion always wins.
                if let Some(alias) = &column.alias {
                    return Cow::Borrowed(alias);
                }

                let full = self.program.connection.get_full_column_names();
                let short = self.program.connection.get_short_column_names();

                // 2. For column references, apply full/short column name logic.
                match &column.expr {
                    turso_parser::ast::Expr::Column {
                        table,
                        column: col_idx,
                        ..
                    } => {
                        if full {
                            // full_column_names=ON: use REAL_TABLE_NAME.COLUMN
                            if let Some((_, table_ref)) = self
                                .program
                                .table_references
                                .find_table_by_internal_id(*table)
                            {
                                let col_name = table_ref
                                    .get_column_at(*col_idx)
                                    .and_then(|c| c.name.as_deref())
                                    .unwrap_or("?");
                                return Cow::Owned(format!(
                                    "{}.{}",
                                    table_ref.get_name(),
                                    col_name
                                ));
                            }
                        }
                        if short || full {
                            // short_column_names=ON: use just COLUMN
                            if let Some(name) = column.name(&self.program.table_references) {
                                return Cow::Borrowed(name);
                            }
                        }
                        // Both OFF: use original expression text
                        if let Some(name) = &column.implicit_column_name {
                            Cow::Borrowed(name.as_str())
                        } else {
                            let tables = [&self.program.table_references];
                            let ctx = PlanContext(&tables);
                            Cow::Owned(column.expr.displayer(&ctx).to_string())
                        }
                    }
                    _ => {
                        // Non-column-ref: use implicit_column_name or displayer
                        match column.name(&self.program.table_references) {
                            Some(name) => Cow::Borrowed(name),
                            None => {
                                let tables = [&self.program.table_references];
                                let ctx = PlanContext(&tables);
                                Cow::Owned(column.expr.displayer(&ctx).to_string())
                            }
                        }
                    }
                }
            }
            QueryMode::Explain => Cow::Borrowed(EXPLAIN_COLUMNS[idx]),
            QueryMode::ExplainQueryPlan => Cow::Borrowed(EXPLAIN_QUERY_PLAN_COLUMNS[idx]),
        }
    }

    pub fn get_column_table_name(&self, idx: usize) -> Option<Cow<'_, str>> {
        if self.query_mode == QueryMode::Explain || self.query_mode == QueryMode::ExplainQueryPlan {
            return None;
        }
        let column = &self.program.result_columns.get(idx).expect("No column");
        match &column.expr {
            turso_parser::ast::Expr::Column { table, .. } => self
                .program
                .table_references
                .find_table_by_internal_id(*table)
                .map(|(_, table_ref)| Cow::Borrowed(table_ref.get_name())),
            _ => None,
        }
    }

    /// Returns the declared type of a result column.
    ///
    /// This behaves similarly to SQLite's `sqlite3_column_decltype()`:
    /// If the Nth column of the returned result set of a SELECT is a table column
    /// (not an expression or subquery) then the declared type of the table column
    /// is returned. If the Nth column of the result set is an expression or subquery,
    /// then None is returned. The returned string is always UTF-8 encoded.
    ///
    /// See: <https://sqlite.org/c3ref/column_decltype.html>
    pub fn get_column_decltype(&self, idx: usize) -> Option<String> {
        if self.query_mode == QueryMode::Explain {
            return Some(
                EXPLAIN_COLUMNS_TYPE
                    .get(idx)
                    .expect("No column")
                    .to_string(),
            );
        }
        if self.query_mode == QueryMode::ExplainQueryPlan {
            return Some(
                EXPLAIN_QUERY_PLAN_COLUMNS_TYPE
                    .get(idx)
                    .expect("No column")
                    .to_string(),
            );
        }
        let column = &self.program.result_columns.get(idx).expect("No column");
        match &column.expr {
            turso_parser::ast::Expr::Column {
                table,
                column: column_idx,
                ..
            } => {
                let (_, table_ref) = self
                    .program
                    .table_references
                    .find_table_by_internal_id(*table)?;
                let table_column = table_ref.get_column_at(*column_idx)?;
                let ty_str = &table_column.ty_str;
                if ty_str.is_empty() {
                    None
                } else {
                    Some(ty_str.clone())
                }
            }
            _ => None,
        }
    }

    /// Returns the type affinity name of a result column (e.g., "INTEGER", "TEXT", "REAL", "BLOB", "NUMERIC").
    ///
    /// Unlike `get_column_decltype` which returns the original declared type string,
    /// this method returns the normalized SQLite type affinity name.
    pub fn get_column_type_name(&self, idx: usize) -> Option<String> {
        if self.query_mode == QueryMode::Explain {
            return Some(
                EXPLAIN_COLUMNS_TYPE
                    .get(idx)
                    .expect("No column")
                    .to_string(),
            );
        }
        if self.query_mode == QueryMode::ExplainQueryPlan {
            return Some(
                EXPLAIN_QUERY_PLAN_COLUMNS_TYPE
                    .get(idx)
                    .expect("No column")
                    .to_string(),
            );
        }
        let column = &self.program.result_columns.get(idx).expect("No column");
        match &column.expr {
            turso_parser::ast::Expr::Column {
                table,
                column: column_idx,
                ..
            } => {
                let (_, table_ref) = self
                    .program
                    .table_references
                    .find_table_by_internal_id(*table)?;
                let table_column = table_ref.get_column_at(*column_idx)?;
                match &table_column.ty() {
                    crate::schema::Type::Integer => Some("INTEGER".to_string()),
                    crate::schema::Type::Real => Some("REAL".to_string()),
                    crate::schema::Type::Text => Some("TEXT".to_string()),
                    crate::schema::Type::Blob => Some("BLOB".to_string()),
                    crate::schema::Type::Numeric => Some("NUMERIC".to_string()),
                    crate::schema::Type::Null => None,
                }
            }
            _ => None,
        }
    }

    pub fn parameters(&self) -> &parameters::Parameters {
        &self.program.parameters
    }

    pub fn parameters_count(&self) -> usize {
        self.program.parameters.count()
    }

    pub fn parameter_index(&self, name: &str) -> Option<NonZero<usize>> {
        self.program.parameters.index(name)
    }

    pub fn bind_at(&mut self, index: NonZero<usize>, value: Value) {
        self.state.bind_at(index, value);
    }

    pub fn clear_bindings(&mut self) {
        self.state.clear_bindings();
    }

    pub fn reset(&mut self) -> Result<()> {
        self.reset_internal(None, None, false).1
    }

    /// Resets the statement like [`Statement::reset`], and returns the outcome
    /// of the execution it ended. A reset can finish that execution (a
    /// RETURNING write whose first row was read commits here), so the outcome
    /// is taken after the reset's own work and before the state is cleared.
    pub fn reset_with_outcome(&mut self) -> (StatementOutcome, Result<()>) {
        self.reset_internal(None, None, false)
    }

    pub fn reset_best_effort(&mut self) {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.reset())) {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                tracing::error!("Statement reset failed during best-effort cleanup: {err}");
            }
            Err(_) => {
                tracing::error!("Statement reset panicked during best-effort cleanup");
            }
        }
    }

    /// Lightweight reset for reusing a cached subprogram statement.
    /// Skips transaction handling and abort(): the caller (op_program) has
    /// already handled trigger execution tracking. Only resets ProgramState
    /// fields so the subprogram can run again from the beginning.
    pub fn reset_for_subprogram_reuse(&mut self) {
        self.state.reset(None, None);
        self.state
            .n_change
            .store(0, std::sync::atomic::Ordering::Release);
        self.busy = false;
        self.has_returned_row = false;
    }

    /// `preserve_active_root_count` is set by a reprepare, which keeps the
    /// execution (its identity and outcome evidence) going.
    fn reset_internal(
        &mut self,
        max_registers: Option<usize>,
        max_cursors: Option<usize>,
        preserve_active_root_count: bool,
    ) -> (StatementOutcome, Result<()>) {
        fn capture_reset_error(
            reset_error: &mut Option<LimboError>,
            err: LimboError,
            context: &str,
        ) {
            tracing::error!("{context}: {err}");
            if reset_error.is_none() {
                *reset_error = Some(err);
            }
        }

        let mut reset_error: Option<LimboError> = None;
        // A failed submitted append the reset itself observed; the teardown
        // must not drive that commit on as if it succeeded.
        let mut append_error: Option<LimboError> = None;

        if let Some(io) = self.state.io_completions.take() {
            if let Err(err) = io.wait(self.pager.io.as_ref()) {
                match self
                    .program
                    .classify_commit_io_failure(&self.pager, &mut self.state, &err)
                {
                    // Recorded; the published commit stands (D8).
                    vdbe::CommitIoFailure::CheckpointGivenUp => {}
                    vdbe::CommitIoFailure::AppendFailed => {
                        append_error = Some(err.clone());
                        capture_reset_error(
                            &mut reset_error,
                            err,
                            "The commit append failed while draining IO during statement reset",
                        );
                    }
                    vdbe::CommitIoFailure::NotCommitting => capture_reset_error(
                        &mut reset_error,
                        err,
                        "Error while draining pending IO during statement reset",
                    ),
                }
            }
        }

        if self.state.execution_state.is_running() {
            if self.query_mode == QueryMode::Normal
                && self.program.change_cnt_on
                && self.has_returned_row
                && append_error.is_none()
            {
                // Write statement with RETURNING, user got at least one Row.
                // With ephemeral-buffered RETURNING, ALL DML completed before any
                // rows were yielded. The remaining work is just the scan-back
                // (in-memory) + Halt. Commit the transaction via halt(). A commit
                // already in flight ends like any teardown's: no new
                // auto-checkpoint, and its failed IO classified.
                let mut halt_completed = false;
                loop {
                    self.program
                        .hold_teardown_boundary(&self.state, &self.pager);
                    match vdbe::execute::halt(
                        &self.program,
                        &mut self.state,
                        &self.pager,
                        0,
                        "",
                        None,
                    ) {
                        Ok(vdbe::execute::InsnFunctionStepResult::Done) => {
                            halt_completed = true;
                            break;
                        }
                        Ok(vdbe::execute::InsnFunctionStepResult::IO(io)) => {
                            let Err(err) = io.wait(self.pager.io.as_ref()) else {
                                continue;
                            };
                            match self.program.classify_commit_io_failure(
                                &self.pager,
                                &mut self.state,
                                &err,
                            ) {
                                vdbe::CommitIoFailure::CheckpointGivenUp => {}
                                vdbe::CommitIoFailure::AppendFailed => {
                                    append_error = Some(err.clone());
                                    capture_reset_error(
                                        &mut reset_error,
                                        err,
                                        "The commit append failed while committing during statement reset",
                                    );
                                    break;
                                }
                                vdbe::CommitIoFailure::NotCommitting => {
                                    capture_reset_error(
                                        &mut reset_error,
                                        err,
                                        "Error committing during statement reset",
                                    );
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            capture_reset_error(
                                &mut reset_error,
                                e,
                                "Error halting statement during reset",
                            );
                            break;
                        }
                        Ok(vdbe::execute::InsnFunctionStepResult::Row)
                        | Ok(vdbe::execute::InsnFunctionStepResult::Step) => {
                            capture_reset_error(
                                &mut reset_error,
                                LimboError::InternalError(
                                    "Unexpected halt result during reset".to_string(),
                                ),
                                "Statement reset encountered unexpected halt result",
                            );
                            break;
                        }
                    }
                }

                if halt_completed {
                    self.terminal = Some(RootTerminal::Done);
                } else {
                    let abort_cause = append_error.as_ref().or(reset_error.as_ref());
                    if let Err(abort_err) =
                        self.program
                            .abort(&self.pager, abort_cause, &mut self.state)
                    {
                        capture_reset_error(
                            &mut reset_error,
                            abort_err,
                            "Abort failed during statement reset",
                        );
                    }
                    self.terminal = Some(self.terminal_after_error(reset_error.is_none()));
                }
            } else {
                // Either a read-only statement, a write statement that never
                // yielded a Row (DML still in progress or hit Busy/error), a
                // write statement without RETURNING, or one whose failed commit
                // append the reset observed. Rollback to avoid committing
                // partial DML or silently retrying after transient errors (Busy).
                if let Err(abort_err) =
                    self.program
                        .abort(&self.pager, append_error.as_ref(), &mut self.state)
                {
                    capture_reset_error(
                        &mut reset_error,
                        abort_err,
                        "Abort failed during statement reset",
                    );
                }
                self.terminal = Some(self.terminal_after_error(reset_error.is_none()));
            }
        } else {
            // Statement not running (Done/Failed/Init) — cleanup only.
            if let Err(abort_err) = self.program.abort(&self.pager, None, &mut self.state) {
                capture_reset_error(
                    &mut reset_error,
                    abort_err,
                    "Abort failed during statement reset",
                );
            }
        }
        // Safety net: if end_statement wasn't reached (e.g. statement dropped
        // mid-execution), ensure n_active_writes is decremented before reset
        // clears the flag.
        if self.state.is_active_write {
            self.program
                .connection
                .n_active_writes
                .fetch_sub(1, Ordering::SeqCst);
            self.state.is_active_write = false;
        }
        if self.counted_as_active_root && !preserve_active_root_count {
            self.release_active_root_if_counted();
        }
        if self.stepped {
            self.fix_transaction_generation(true);
        }
        let outcome = self.outcome_snapshot();
        let evidence = std::mem::take(&mut self.state.outcome);
        self.state.reset(max_registers, max_cursors);
        if preserve_active_root_count {
            self.state.outcome = evidence;
        } else {
            self.root_generation = 0;
            self.stepped = false;
            self.terminal = None;
        }
        self.state.n_change.store(0, Ordering::SeqCst);
        self.busy = false;
        self.busy_handler_state = None;
        self.query_timeout_override = None;
        self.has_returned_row = false;

        if let Some(err) = reset_error {
            return (outcome, Err(err));
        }
        (outcome, Ok(()))
    }

    pub fn row(&self) -> Option<&Row> {
        self.state.result_row.as_ref()
    }

    pub fn get_sql(&self) -> &str {
        &self.program.sql
    }

    pub fn is_busy(&self) -> bool {
        self.busy
    }

    /// Internal method to get IO from a statement.
    /// Used by select internal crate
    ///
    /// Avoid using this method for advancing IO while iteration over `step`.
    /// Prefer to use helper methods instead such as [Self::run_with_row_callback]
    pub fn _io(&self) -> &dyn crate::IO {
        self.pager.io.as_ref()
    }
}

impl Drop for Statement {
    fn drop(&mut self) {
        // Keep helper statements nested while drop-time reset/abort cleanup runs.
        // That cleanup consults `is_nested_stmt()` to decide whether top-level
        // transaction/savepoint finalization belongs to this statement or to its
        // parent, so we release the nested guard only after reset completes.
        self.reset_best_effort();
        if self.nested_guard_active {
            self.program.connection.end_nested();
            self.nested_guard_active = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Database, DatabaseOpts, MemoryIO, OpenFlags, IO};

    fn open_test_connection() -> crate::Result<Arc<crate::Connection>> {
        let io: Arc<dyn IO> = Arc::new(MemoryIO::new());
        let db = Database::open_file_with_flags(
            io,
            ":memory:",
            OpenFlags::Create,
            DatabaseOpts::new(),
            None,
        )?;
        db.connect()
    }

    #[test]
    fn test_metrics_persist_across_reset() {
        let conn = open_test_connection().unwrap();
        conn.execute("CREATE TABLE t(x)").unwrap();
        conn.metrics.write().reset();

        let mut stmt = conn.prepare("INSERT INTO t VALUES (1)").unwrap();
        stmt.run_ignore_rows().unwrap();
        assert_eq!(stmt.metrics().rows_written, 1);

        stmt.reset().unwrap();
        assert_eq!(stmt.metrics().rows_written, 1);

        stmt.run_ignore_rows().unwrap();
        assert_eq!(stmt.metrics().rows_written, 2);

        stmt.reset_metrics();
        assert_eq!(stmt.metrics().rows_written, 0);
    }

    #[test]
    fn test_metrics_include_subprogram_writes() {
        let conn = open_test_connection().unwrap();
        conn.execute("CREATE TABLE src(x)").unwrap();
        conn.execute("CREATE TABLE log(x)").unwrap();
        conn.execute(
            "CREATE TRIGGER src_log AFTER INSERT ON src BEGIN INSERT INTO log VALUES (new.x); END",
        )
        .unwrap();

        let mut stmt = conn.prepare("INSERT INTO src VALUES (1), (2)").unwrap();
        stmt.run_ignore_rows().unwrap();

        assert_eq!(
            stmt.metrics().rows_written,
            6,
            "cumulative metrics should include root and trigger writes"
        );
    }

    /// The connection ignores an interrupt while no root statement is active,
    /// so one delivered before the first step is lost without activation.
    #[test]
    fn interrupt_before_the_first_step_is_lost_without_activation() {
        let conn = open_test_connection().unwrap();
        conn.execute("CREATE TABLE t(x)").unwrap();
        let mut stmt = conn.prepare("INSERT INTO t VALUES (1)").unwrap();
        conn.interrupt();
        assert!(!conn.is_interrupted(), "no root statement is active yet");
        stmt.run_ignore_rows().unwrap();
        assert_eq!(stmt.outcome_snapshot().terminal, Some(RootTerminal::Done));
    }

    /// Activated first, the statement takes that interrupt at its first step
    /// and ends rolled back, under the identity it got at activation.
    #[test]
    fn activate_root_keeps_an_interrupt_delivered_before_the_first_step() {
        let conn = open_test_connection().unwrap();
        conn.execute("CREATE TABLE t(x)").unwrap();
        let mut stmt = conn.prepare("INSERT INTO t VALUES (1)").unwrap();
        stmt.activate_root();
        let activated = stmt.outcome_snapshot();
        assert_eq!(activated.phase, CommitPhase::NotStarted);
        assert_ne!(activated.identity.root_generation, 0);
        conn.interrupt();
        assert!(conn.is_interrupted(), "the activated statement is active");

        let result = stmt.step();
        assert!(
            matches!(result, Ok(StepResult::Interrupt)),
            "activate_root_interrupt: the first step takes the interrupt, got {result:?}"
        );
        let outcome = stmt.outcome_snapshot();
        assert_eq!(
            outcome.identity.root_generation,
            activated.identity.root_generation
        );
        assert_eq!(outcome.phase, CommitPhase::NotPublished);
        assert_eq!(outcome.terminal, Some(RootTerminal::RolledBack));
        assert_eq!(
            outcome.cancel_observed,
            Some(CancelObserved::BeforeIrreversibleCommit)
        );
        drop(stmt);
        assert!(
            !conn.is_interrupted(),
            "no root statement is active any more"
        );
        let mut count = conn.prepare("SELECT count(*) FROM t").unwrap();
        assert_eq!(
            count.run_collect_rows().unwrap(),
            vec![vec![Value::from_i64(0)]]
        );
    }

    /// An activated statement dropped before its first step leaves the
    /// connection without an active root statement.
    #[test]
    fn activated_statement_dropped_before_stepping_is_released() {
        let conn = open_test_connection().unwrap();
        let mut stmt = conn.prepare("SELECT 1").unwrap();
        stmt.activate_root();
        stmt.activate_root();
        assert_eq!(conn.n_active_root_statements.load(Ordering::SeqCst), 1);
        drop(stmt);
        assert_eq!(conn.n_active_root_statements.load(Ordering::SeqCst), 0);
    }
}
