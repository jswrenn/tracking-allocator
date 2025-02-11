use tracking_allocator::{
    AllocationGroupId, AllocationGroupToken, AllocationRegistry, AllocationTracker, Allocator,
};

use std::{
    alloc::System,
    sync::mpsc::{sync_channel, SyncSender},
};

// This is where we actually set the global allocator to be the shim allocator implementation from
// `tracking_allocator`.  This allocator is purely a facade to the logic provided by the crate,
// which is controlled by setting a global tracker and registering allocation groups.  All of that
// is covered below.
//
// As well, you can see here that we're wrapping the system allocator.  If you want, you can
// construct `Allocator` by wrapping another allocator that implements `GlobalAlloc`.  Since this is
// a static, you need a way to construct ther allocator to be wrapped in a const fashion, but it
// _is_ possible.
#[global_allocator]
static GLOBAL: Allocator<System> = Allocator::system();

enum AllocationEvent {
    Allocated {
        addr: usize,
        size: usize,
        group_id: AllocationGroupId,
    },
    Deallocated {
        addr: usize,
        current_group_id: AllocationGroupId,
    },
}

struct ChannelBackedTracker {
    // Our sender is using a bounded (fixed-size) channel to push allocation events to.  This is
    // important because we do _not_ want to allocate in order to actually send our allocation
    // events.  We would end up in a reentrant loop that could either overflow the stack, or
    // deadlock, depending on what the tracker does under the hood.
    //
    // This can happen for many different resources we might take for granted.  For example, we
    // can't write to stdout via `println!` as that would allocate, which causes an issue where the
    // reentrant call to `println!` ends up hitting a mutex around stdout, deadlocking the process.
    //
    // Also take care to note that the `AllocationEvent` structure has a fixed size and requires no
    // allocations itself to create.  This follows the same principle as using a bounded channel.
    sender: SyncSender<AllocationEvent>,
}

// This is our tracker implementation.  You will always need to create an implementation of
// `AllocationTracker` in order to actually handle allocation events.  The interface is
// straightforward: you're notified when an allocation occurs, and when a deallocation occurs.
impl AllocationTracker for ChannelBackedTracker {
    fn allocated(&self, addr: usize, size: usize, group_id: AllocationGroupId) {
        // Allocations have all the pertinent information upfront, which you must store if you want
        // to do any correlation with deallocations.
        let _ = self.sender.send(AllocationEvent::Allocated {
            addr,
            size,
            group_id,
        });
    }

    fn deallocated(&self, addr: usize, current_group_id: AllocationGroupId) {
        // As `tracking_allocator` itself strives to add as little overhead as possible, we only
        // forward the address being deallocated.  Your tracker implementation will need to handle
        // mapping the allocation address back to allocation group if you need to know the total
        // in-use memory, vs simply knowing how many or when allocations are occurring.
        let _ = self.sender.send(AllocationEvent::Deallocated {
            addr,
            current_group_id,
        });
    }
}

impl From<SyncSender<AllocationEvent>> for ChannelBackedTracker {
    fn from(sender: SyncSender<AllocationEvent>) -> Self {
        ChannelBackedTracker { sender }
    }
}

fn main() {
    // Create our channels for receiving the allocation events.
    let (tx, rx) = sync_channel(32);

    // Create and set our allocation tracker.  Even with the tracker set, we're still not tracking
    // allocations yet.  We need to enable tracking explicitly.
    let _ = AllocationRegistry::set_global_tracker(ChannelBackedTracker::from(tx))
        .expect("no other global tracker should be set yet");

    AllocationRegistry::enable_tracking();

    // Register an allocation group.  Allocation groups are what allocations are associated with,
    // and allocations are only tracked if an allocation group is "active".  This gives us a way to
    // actually have another task or thread processing the allocation events -- which may require
    // allocating storage to do so -- without ending up in a weird re-entrant situation if we just
    // instrumented all allocations throughout the process.
    //
    // Callers can attach tags to their group by calling `AllocationRegistry::register_with_tags`
    // instead, but we're going to register our group without any tags for now.
    let local_token =
        AllocationGroupToken::register().expect("failed to register allocation group");

    // Now, get an allocation guard from our token.  This guard ensures the allocation group is
    // marked as the current allocation group, so that our allocations are properly associated.
    let local_guard = local_token.enter();

    // Now we can finally make some allocations!
    let s = String::from("Hello world!");
    let mut v = Vec::new();
    v.push(s);

    // Drop our "local" group guard.  You can also call `exit` on `AllocationGuard` to transform it
    // back to an `AllocationToken` for further reuse.  Exiting/dropping the guard will update the
    // thread state so that any allocations afterwards are once again attributed to the "global"
    // allocation group.
    drop(local_guard);

    // Drop the vector to generate some deallocations.
    drop(v);

    // Disable tracking and read the allocation events from our receiver.
    AllocationRegistry::disable_tracking();

    // We should end up seeing four events here: two allocations for the `String` and the `Vec`
    // associated with the local allocation group, and two deallocations when we drop the `Vec`.
    //
    // The two allocations should be attributed to our local allocation group
    // (group ID #1) while the deallocations will be attributed to the global allocation group.
    // This simply means they were deallocated within the global allocation group (aka not actually
    // within a registered allocation group) but you would still need to track addresses to group
    // IDs on your own to know who "owns" an allocation.
    let mut events = rx.try_iter();
    while let Some(event) = events.next() {
        match event {
            AllocationEvent::Allocated {
                addr,
                size,
                group_id,
            } => {
                println!(
                    "allocation -> addr={:#x} size={} group_id={:?}",
                    addr, size, group_id
                );
            }
            AllocationEvent::Deallocated {
                addr,
                current_group_id,
            } => {
                println!(
                    "deallocation -> addr={:#x} current_group_id={:?}",
                    addr, current_group_id
                );
            }
        }
    }
}
