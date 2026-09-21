use crate::ChatCompletionRequest;
use crate::routes::RoutingTreeBuilder;
use crate::routes::interface::RouteHandle;
use crate::types::{AppError, ConfigurationError, Request, Response};
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::service::Service;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;

#[derive(Clone)]
pub struct RequestHandler {
    core: Arc<Core>,
}
struct Core {
    root: Box<dyn RouteHandle + Send + Sync>,
}

impl RequestHandler {
    pub fn new(config: &std::path::PathBuf) -> Result<Self, ConfigurationError> {
        let builder = RoutingTreeBuilder::from_file(config)?;
        let root = builder.build_routing_tree()?;

        Ok(Self {
            core: Arc::new(Core { root }),
        })
    }

    async fn process_request(&self, request: Request) -> Response {
        log::debug!("Got request");
        let result = match request.uri().path() {
            "/v1/chat/completions" => self.handle_chat_completion(request).await,
            nonexistent_route => Err(AppError::NotFound(format!(
                "{} does not exist",
                nonexistent_route
            ))),
        };

        match result {
            Ok(response) => {
                log::debug!("Success");
                response
            }
            Err(app_error) => {
                log::error!("Error");
                app_error.to_response()
            }
        }
    }

    async fn handle_chat_completion(&self, request: Request) -> Result<Response, AppError> {
        let body = request
            .collect()
            .await
            .map_err(|e| AppError::InternalError(e.to_string()))?
            .to_bytes();
        let chat_request: ChatCompletionRequest = serde_json::from_slice(&body).map_err(|e| {
            AppError::BadRequest(format!("Invalid JSON in request body: {}", e.to_string()))
        })?;

        let response = self.core.root.route(&chat_request).await?;

        Ok(hyper::Response::builder()
            .body(Full::new(Bytes::from(
                serde_json::to_string(&response).unwrap(),
            )))
            .unwrap())
    }
}

impl Service<Request> for RequestHandler {
    // TODO Support streaming responses
    type Response = Response;
    // All errors are converted to proper HTTP responses instead of simply terminating connections
    type Error = Infallible;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&self, request: Request) -> Self::Future {
        let handler = self.clone();

        Box::pin(async move { Ok(handler.process_request(request).await) })
    }
}
