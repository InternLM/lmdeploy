use crate::routes::interface::RouteHandle;
use crate::{AppError, ChatCompletionResponse};
use crate::{ChatCompletionRequest, routes::SingleServerRoute};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct RoundRobinRoute {
    children: Vec<SingleServerRoute>,
    idx: AtomicUsize,
}

impl RoundRobinRoute {
    pub fn new(servers: Vec<(&str, u16)>) -> Self {
        Self {
            children: servers
                .into_iter()
                .map(|(host, port)| SingleServerRoute::new(host, port))
                .collect(),
            idx: AtomicUsize::new(0),
        }
    }
}

impl RouteHandle for RoundRobinRoute {
    fn route<'a>(
        &'a self,
        request: &'a ChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatCompletionResponse, AppError>> + Send + 'a>> {
        Box::pin(async move {
            let idx = self.idx.fetch_add(1, Ordering::Relaxed) % self.children.len();
            let child = self.children.get(idx).unwrap();

            Ok(child.route(request).await?)
        })
    }
}
