use std::alloc::{GlobalAlloc, Layout, System};
use std::mem;

use crate::token::CURRENT_ALLOCATION_TOKEN;
use crate::{get_global_tracker, AllocationGroupId};

/// Tracking allocator implementation.
///
/// This allocator must be installed via `#[global_allocator]` in order to take effect.  More
/// information on using this allocator can be found in the examples, or directly in the standard
/// library docs for [`GlobalAlloc`].
pub struct Allocator<A> {
    inner: A,
}

impl<A> Allocator<A> {
    /// Creates a new `Allocator` that wraps another allocator.
    pub const fn from_allocator(allocator: A) -> Self {
        Self { inner: allocator }
    }
}

impl Allocator<System> {
    /// Creates a new `Allocator` that wraps the system allocator.
    pub const fn system() -> Allocator<System> {
        Self::from_allocator(System)
    }
}

impl Default for Allocator<System> {
    fn default() -> Self {
        Self::from_allocator(System)
    }
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for Allocator<A> {
    #[track_caller]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        CURRENT_ALLOCATION_TOKEN
            .try_with(
                #[inline(always)]
                |current| {
                    if let Ok(mut token) = current.try_borrow_mut() {
                        let maybe_group_id = token.take();

                        let metadata_size = mem::size_of::<AllocationGroupId>();

                        let ptr = if let Some(augmented_size) =
                            layout.size().checked_add(metadata_size)
                        {
                            // safety: layout.align() is already known to be a valid alignment
                            let augmented_layout = unsafe {
                                Layout::from_size_align_unchecked(augmented_size, layout.align())
                            };

                            let ptr = self.inner.alloc(augmented_layout);

                            if !ptr.is_null() {
                                // safety:
                                //  - ptr isn't null
                                //  - we're writing up to the end of `ptr`'s allocation, but not
                                //    past it
                                unsafe {
                                    ptr.add(layout.size())
                                        .cast::<Option<AllocationGroupId>>()
                                        .write_unaligned(maybe_group_id.clone());
                                }
                            }

                            ptr
                        } else {
                            // if the requested allocation is so huge we can't add a few bytes to the
                            // end, restore the allocation group id, and return a null pointer.
                            *token = maybe_group_id;
                            return std::ptr::null_mut();
                        };

                        if let Some(tracker) = get_global_tracker() {
                            if let Some(group_id) = maybe_group_id.clone() {
                                let addr = ptr as usize;
                                tracker.allocated(addr, layout, group_id);
                            }
                        }

                        *token = maybe_group_id;

                        return ptr;
                    } else {
                        unreachable!()
                    }
                },
            )
            .unwrap_or(std::ptr::null_mut())
    }

    #[track_caller]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CURRENT_ALLOCATION_TOKEN
            .try_with(
                #[inline(always)]
                |current| {
                    if let Ok(mut token) = current.try_borrow_mut() {
                        let maybe_deallocating_group_id = token.take();
                        self.inner.dealloc(ptr, layout);

                        // safety: layout.align() is already known to be a valid alignment
                        let underlying_layout = unsafe {
                            Layout::from_size_align_unchecked(
                                layout.size() - mem::size_of::<AllocationGroupId>(),
                                layout.align(),
                            )
                        };

                        let maybe_allocating_group_id = ptr
                            .add(underlying_layout.size())
                            .cast::<Option<AllocationGroupId>>()
                            .read_unaligned();

                        if let Some(tracker) = get_global_tracker() {
                            if let (Some(allocating_group_id), Some(deallocating_group_id)) = (
                                maybe_allocating_group_id,
                                maybe_deallocating_group_id.clone(),
                            ) {
                                // only log a deallocation event if both the allocating AND
                                // deallocating group ids are `Some`. why? if the allocating group
                                // id is `None`, it suggests that the allocation stemmed from
                                // whatever the end user set up for processing allocation events.
                                // likewise, if the deallocation group id is null, this deallocation
                                // probably stems from event processing happening *right now*.
                                let addr = ptr as usize;
                                tracker.deallocated(
                                    addr,
                                    underlying_layout,
                                    allocating_group_id,
                                    deallocating_group_id,
                                );
                            }
                        }
                        *token = maybe_deallocating_group_id;
                        return ptr;
                    } else {
                        // unreachable
                        return std::ptr::null_mut();
                    }
                },
            )
            .unwrap_or(std::ptr::null_mut());
    }
}
