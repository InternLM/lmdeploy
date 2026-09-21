use std::pin::Pin;

use crate::types::{AppError, ChatCompletionRequest, ChatCompletionResponse};

pub trait RouteHandle {
    fn route<'a>(
        &'a self,
        request: &'a ChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatCompletionResponse, AppError>> + Send + 'a>>;
}
