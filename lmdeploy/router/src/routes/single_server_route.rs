use crate::routes::interface::RouteHandle;
use crate::types::{AppError, ChatCompletionRequest, ChatCompletionResponse};
use reqwest::header::CONTENT_TYPE;
use std::pin::Pin;

pub struct SingleServerRoute {
    host: String,
    port: u16,
    client: reqwest::Client,
}

impl SingleServerRoute {
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            host: host.to_string(),
            port,
            client: reqwest::Client::new(),
        }
    }
}

impl RouteHandle for SingleServerRoute {
    fn route<'a>(
        &'a self,
        request: &'a ChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatCompletionResponse, AppError>> + Send + 'a>> {
        Box::pin(async move {
            let response = self
                .client
                .post(format!(
                    "http://{}:{}/v1/chat/completions",
                    self.host, self.port
                ))
                .header(CONTENT_TYPE, "application/json")
                .json(&request)
                .send()
                .await?;

            Ok(response.json::<ChatCompletionResponse>().await?)
        })
    }
}
