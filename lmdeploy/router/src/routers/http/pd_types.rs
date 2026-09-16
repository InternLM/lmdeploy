// Custom error type for PD router operations
#[derive(Debug, thiserror::Error)]
pub enum PDRouterError {
    #[error("Worker already exists: {url}")]
    WorkerAlreadyExists { url: String },

    #[error("Worker not found: {url}")]
    WorkerNotFound { url: String },

    #[error("Lock acquisition failed: {operation}")]
    LockError { operation: String },

    #[error("Health check failed for worker: {url}")]
    HealthCheckFailed { url: String },

    #[error("Invalid worker configuration: {reason}")]
    InvalidConfiguration { reason: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Timeout waiting for worker: {url}")]
    Timeout { url: String },
}

/// Format a full error chain for debugging (walks source() recursively).
/// Produces output like: "outer error caused by: middle error caused by: root cause"
pub fn error_chain(err: &dyn std::error::Error) -> String {
    let mut chain = vec![err.to_string()];
    let mut source = err.source();
    while let Some(s) = source {
        chain.push(s.to_string());
        source = s.source();
    }
    chain.join(" caused by: ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;

    // Simple custom error for testing error chains
    #[derive(Debug)]
    struct TestError {
        msg: String,
        source: Option<Box<dyn std::error::Error>>,
    }

    impl fmt::Display for TestError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.msg)
        }
    }

    impl std::error::Error for TestError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            self.source.as_deref()
        }
    }

    #[test]
    fn test_error_chain_single_error() {
        let err = TestError {
            msg: "something broke".into(),
            source: None,
        };
        assert_eq!(error_chain(&err), "something broke");
    }

    #[test]
    fn test_error_chain_nested_errors() {
        let inner = TestError {
            msg: "root cause".into(),
            source: None,
        };
        let outer = TestError {
            msg: "outer error".into(),
            source: Some(Box::new(inner)),
        };
        assert_eq!(error_chain(&outer), "outer error caused by: root cause");
    }

    #[test]
    fn test_error_chain_triple_nested() {
        let root = TestError {
            msg: "connection reset".into(),
            source: None,
        };
        let middle = TestError {
            msg: "HTTP send failed".into(),
            source: Some(Box::new(root)),
        };
        let top = TestError {
            msg: "prefill request failed".into(),
            source: Some(Box::new(middle)),
        };
        assert_eq!(
            error_chain(&top),
            "prefill request failed caused by: HTTP send failed caused by: connection reset"
        );
    }
}
