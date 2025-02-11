use std::{any::TypeId, marker::PhantomData};

use tracing::{Dispatch, Id, Subscriber};
use tracing_subscriber::{layer::Context, registry::LookupSpan, Layer};

use crate::token::UnsafeAllocationGroupToken;

pub(crate) struct WithAllocationGroup {
    with_allocation_group: fn(&Dispatch, &Id, UnsafeAllocationGroupToken),
}

impl WithAllocationGroup {
    pub fn with_allocation_group(
        &self,
        dispatch: &Dispatch,
        id: &Id,
        unsafe_token: UnsafeAllocationGroupToken,
    ) {
        (self.with_allocation_group)(dispatch, id, unsafe_token)
    }
}

/// [`AllocationLayer`] is a [`tracing_subscriber::Layer`] that handles entering and exiting an allocation
/// group as the span it is attached to is itself entered and exited.
///
/// More information on using this layer can be found in the examples, or directly in the
/// `tracing_subscriber` docs, found [here][tracing_subscriber::layer].
#[cfg_attr(docsrs, doc(cfg(feature = "tracing-compat")))]
pub struct AllocationLayer<S> {
    ctx: WithAllocationGroup,
    _subscriber: PhantomData<fn(S)>,
}

impl<S> AllocationLayer<S>
where
    S: Subscriber + for<'span> LookupSpan<'span>,
{
    /// Create a new `TracingLayer`.
    pub fn new() -> Self {
        let ctx = WithAllocationGroup {
            with_allocation_group: Self::with_allocation_group,
        };

        Self {
            ctx,
            _subscriber: PhantomData,
        }
    }

    fn with_allocation_group(
        dispatch: &Dispatch,
        id: &Id,
        unsafe_token: UnsafeAllocationGroupToken,
    ) {
        let subscriber = dispatch
            .downcast_ref::<S>()
            .expect("subscriber should downcast to expected type; this is a bug!");
        let span = subscriber
            .span(id)
            .expect("registry should have a span for the current ID");

        span.extensions_mut().insert(unsafe_token);
    }
}

impl<S> Layer<S> for AllocationLayer<S>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        if let Some(span_ref) = ctx.span(id) {
            if let Some(token) = span_ref
                .extensions_mut()
                .get_mut::<UnsafeAllocationGroupToken>()
            {
                token.enter();
            }
        }
    }

    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        if let Some(span_ref) = ctx.span(id) {
            if let Some(token) = span_ref
                .extensions_mut()
                .get_mut::<UnsafeAllocationGroupToken>()
            {
                token.exit();
            }
        }
    }

    unsafe fn downcast_raw(&self, id: TypeId) -> Option<*const ()> {
        match id {
            id if id == TypeId::of::<Self>() => Some(self as *const _ as *const ()),
            id if id == TypeId::of::<WithAllocationGroup>() => {
                Some(&self.ctx as *const _ as *const ())
            }
            _ => None,
        }
    }
}

impl<S> Default for AllocationLayer<S>
where
    S: Subscriber + for<'span> LookupSpan<'span>,
{
    fn default() -> Self {
        AllocationLayer::new()
    }
}
