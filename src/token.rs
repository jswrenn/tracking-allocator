use std::{
    cell::RefCell,
    mem,
    num::NonZeroUsize,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::util::PhantomNotSend;

thread_local! {
    /// The currently executing allocation token.
    ///
    /// Any allocations which occur on this thread will be associated with whichever token is
    /// present at the time of the allocation.
    pub (crate) static CURRENT_ALLOCATION_TOKEN: RefCell<Option<AllocationGroupId>> = 
        RefCell::new(Some(AllocationGroupId::ROOT));
}

/// The identifier that uniquely identifiers an allocation group.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct AllocationGroupId(NonZeroUsize);

impl AllocationGroupId {
    /// The group ID used for allocations which are not made within a registered allocation group.
    pub const ROOT: Self = Self(unsafe { NonZeroUsize::new_unchecked(1) });

    const fn as_usize(&self) -> usize {
        Self::ROOT.0.get()
    }

    fn next() -> Option<AllocationGroupId> {
        static GROUP_ID: AtomicUsize = AtomicUsize::new(AllocationGroupId::ROOT.as_usize() + 1);
        static HIGHEST_GROUP_ID: AtomicUsize =
            AtomicUsize::new(AllocationGroupId::ROOT.as_usize() + 1);

        let group_id = GROUP_ID.fetch_add(1, Ordering::Relaxed);
        let highest_group_id = HIGHEST_GROUP_ID.fetch_max(group_id, Ordering::AcqRel);

        if group_id >= highest_group_id {
            let group_id = NonZeroUsize::new(group_id).expect("bug: group_id overflowed");
            Some(AllocationGroupId(group_id))
        } else {
            None
        }
    }
}

/// A token that uniquely identifies an allocation group.
///
/// Allocation groups are the core grouping mechanism of `tracking-allocator` and drive much of its
/// behavior.  While the allocator must be overridden, and a global track provided, no allocations
/// are tracked unless a group is associated with the current thread making the allocation.
///
/// Practically speaking, allocation groups are simply an internal identifier that is used to
/// identify the "owner" of an allocation.  Additional tags can be provided when acquire an
/// allocation group token, which is provided to [`AllocationTracker`][crate::AllocationTracker]
/// whenever an allocation occurs.
///
/// ## Usage
///
/// In order for an allocation group to be attached to an allocation, its must be "entered."
/// [`AllocationGroupToken`] functions similarly to something like a mutex, where "entering" the
/// token conumes the token and provides a guard: [`AllocationGuard`].  This guard is tied to the
/// allocation group being active: if the guard is dropped, or if it is exited manually, the
/// allocation group is no longer active.
///
/// [`AllocationGuard`] also tracks if another allocation group was active prior to entering, and
/// ensures it is set back as the active allocation group when the guard is dropped.  This allows
/// allocation groups to be nested within each other.
pub struct AllocationGroupToken(AllocationGroupId);

impl AllocationGroupToken {
    /// Registers an allocation group token.
    ///
    /// Allocation group IDs are assigned atomically and monotonically, and cannot exceed the
    /// maximum value of the target's pointer size i.e. the upper bound for group IDs on 32-bit
    /// platforms is around 4 billion.
    ///
    /// If this call would cause a group ID to be generated that had already been acquired within
    /// this process, `None` will be returned.  Once `None` has been returned, all subsequent calls
    /// to `register` will return `None`.
    ///
    /// Otherwise, `Some(token)` is returned.
    pub fn register() -> Option<AllocationGroupToken> {
        AllocationGroupId::next().map(AllocationGroupToken)
    }

    /// The ID associated with this allocation group.
    pub fn id(&self) -> AllocationGroupId {
        self.0.clone()
    }

    #[cfg(feature = "tracing-compat")]
    pub(crate) fn into_unsafe(self) -> UnsafeAllocationGroupToken {
        UnsafeAllocationGroupToken::new(self.0)
    }

    /// Marks the associated allocation group as the active allocation group on this thread.
    ///
    /// If another allocation group is currently active, it is replaced, and restored either when
    /// this allocation guard is dropped, or when [`AllocationGuard::exit`] is called.
    pub fn enter(self) -> AllocationGuard {
        AllocationGuard::enter(self)
    }
}

#[cfg(feature = "tracing-compat")]
#[cfg_attr(docsrs, doc(cfg(feature = "tracing-compat")))]
impl AllocationGroupToken {
    /// Attaches this allocation group to a tracing [`Span`][tracing::Span].
    ///
    /// When the span is entered or exited, the allocation group will also transition from idle to
    /// active, or active to idle.  In effect, all allocations that occur while the span is entered
    /// will be associated with the allocation group.
    pub fn attach_to_span(self, span: &tracing::Span) {
        use crate::tracing::WithAllocationGroup;

        let mut unsafe_token = Some(self.into_unsafe());

        tracing::dispatcher::get_default(move |dispatch| {
            if let Some(id) = span.id() {
                if let Some(ctx) = dispatch.downcast_ref::<WithAllocationGroup>() {
                    let unsafe_token = unsafe_token.take().expect("token already consumed");
                    ctx.with_allocation_group(dispatch, &id, unsafe_token);
                }
            }
        });
    }
}

enum GuardState {
    // Guard is idle.  We aren't the active allocation group.
    Idle(AllocationGroupId),

    // Guard is active.  We're the active allocation group, so we hold on to the previous
    // allocation group ID, if there was one, so we can switch back to it when we transition to
    // being idle.
    Active(Option<AllocationGroupId>),
}

impl GuardState {
    fn transition_to_active(&mut self) {
        let new_state = match self {
            Self::Idle(id) => {
                // Set the current allocation token to the new token, keeping the previous.
                let previous =
                    CURRENT_ALLOCATION_TOKEN.with(|current| current.replace(Some(id.clone())));
                Self::Active(previous)
            }
            Self::Active(ref previous) => {
                let current = CURRENT_ALLOCATION_TOKEN.with(|current| current.borrow().clone());
                panic!(
                    "tid {:?}: transitioning active->active is invalid; current={:?} previous={:?}",
                    std::thread::current().id(),
                    current,
                    previous
                );
            }
        };
        *self = new_state;
    }

    fn transition_to_idle(&mut self) -> AllocationGroupId {
        match self.try_transition_to_idle() {
            None => panic!(
                "tid {:?}: transitioning idle->idle is invalid",
                std::thread::current().id()
            ),
            Some(id) => id,
        }
    }

    fn try_transition_to_idle(&mut self) -> Option<AllocationGroupId> {
        let (id, new_state) = match self {
            Self::Idle(_) => return None,
            Self::Active(previous) => {
                // Reset the current allocation token to the previous one:
                let current = CURRENT_ALLOCATION_TOKEN.with(|current| {
                    let old = mem::replace(&mut *current.borrow_mut(), previous.take());
                    old.expect("transitioned to idle state with empty CURRENT_ALLOCATION_TOKEN")
                });
                (Some(current.clone()), Self::Idle(current))
            }
        };
        *self = new_state;
        id
    }
}
/// Guard that updates the current thread to track allocations for the associated allocation group.
///
/// ## Drop behavior
///
/// This guard has a [`Drop`] implementation that resets the active allocation group back to the
/// previous allocation group.  Calling `exit` is generally preferred for being explicit about when
/// the allocation group begins and ends, though.
///
/// ## Moving across threads
///
/// [`AllocationGuard`] is specifically marked as `!Send` as the active allocation group is tracked
/// at a per-thread level.  If you acquire an `AllocationGuard` and need to resume computation on
/// another thread, such as across an await point or when simply sending objects to another thread,
/// you must first [`exit`][exit] the guard and move the resulting [`AllocationGroupToken`].  Once
/// on the new thread, you can then reacquire the guard.
///
/// [exit]: AllocationGuard::exit
pub struct AllocationGuard {
    state: GuardState,

    /// ```compile_fail
    /// use tracking_allocator::AllocationGuard;
    /// trait AssertSend: Send {}
    ///
    /// impl AssertSend for AllocationGuard {}
    /// ```
    _ns: PhantomNotSend,
}

impl AllocationGuard {
    pub(crate) fn enter(token: AllocationGroupToken) -> AllocationGuard {
        let mut state = GuardState::Idle(token.0);
        state.transition_to_active();

        AllocationGuard {
            state,
            _ns: PhantomNotSend::default(),
        }
    }

    /// Unmarks this allocation group as the active allocation group on this thread, resetting the
    /// active allocation group to the previous value.
    pub fn exit(mut self) -> AllocationGroupToken {
        // Reset the current allocation token to the previous one.
        let current = self.state.transition_to_idle();

        AllocationGroupToken(current)
    }
}

impl Drop for AllocationGuard {
    fn drop(&mut self) {
        let _ = self.state.try_transition_to_idle();
    }
}

/// Unmanaged allocation group token used specifically with `tracing`.
///
/// ## Safety
///
/// While normally users would work directly with [`AllocationGroupToken`] and [`AllocationGuard`],
/// we cannot store [`AllocationGuard`] in span data as it is `!Send`, and tracing spans can be sent
/// across threads.
///
/// However, `tracing` itself employs a guard for entering spans.  The guard is `!Send`, which
/// ensures that the guard cannot be sent across threads.  Since the same guard is used to know when
/// a span has been exited, `tracing` ensures that between a span being entered and exited, it
/// cannot move threads.
///
/// Thus, we build off of that invariant, and use this stripped down token to manually enter and
/// exit the allocation group in a specialized `tracing_subscriber` layer that we control.
#[cfg(feature = "tracing-compat")]
pub(crate) struct UnsafeAllocationGroupToken {
    state: GuardState,
}

#[cfg(feature = "tracing-compat")]
impl UnsafeAllocationGroupToken {
    /// Creates a new `UnsafeAllocationGroupToken`.
    pub fn new(id: AllocationGroupId) -> Self {
        Self {
            state: GuardState::Idle(id),
        }
    }

    /// Marks the associated allocation group as the active allocation group on this thread.
    ///
    /// If another allocation group is currently active, it is replaced, and restored either when
    /// this allocation guard is dropped, or when [`AllocationGuard::exit`] is called.
    ///
    /// Functionally equivalent to [`AllocationGroupToken::enter`].
    pub fn enter(&mut self) {
        self.state.transition_to_active();
    }

    /// Unmarks this allocation group as the active allocation group on this thread, resetting the
    /// active allocation group to the previous value.
    ///
    /// Functionally equivalent to [`AllocationGuard::exit`].
    pub fn exit(&mut self) {
        let _ = self.state.transition_to_idle();
    }
}

/// Calls `f` with the current allocation token, without tracking allocations in `f`.
#[inline(always)]
pub(crate) fn with_suspended_allocation_group_id<F>(mut f: F)
where
    F: FnMut(AllocationGroupId),
{
    let _ = CURRENT_ALLOCATION_TOKEN.try_with(
        #[inline(always)]
        |current| {
            if let Ok(mut token) = current.try_borrow_mut() {
                if let Some(group_id) = token.take() {
                    *token = None;
                    f(group_id.clone());
                    *token = Some(group_id);
                }
            }
        },
    );
}
