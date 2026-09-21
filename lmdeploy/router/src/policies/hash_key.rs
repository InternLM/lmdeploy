//! Shared hash key extraction for consistent hashing policies
//!
//! Extracts routing keys from HTTP headers and request bodies.
//! Used by both ConsistentHashPolicy and RendezvousHashPolicy.

use super::RequestHeaders;
use crate::policies::ConsistentHashPolicy;
use tracing::debug;

/// HTTP header names to check for session ID (case-insensitive, checked in order)
pub(crate) const SESSION_HEADER_NAMES: &[&str] = &[
    "x-session-id",
    "x-user-id",
    "x-tenant-id",
    "x-correlation-id", // per-session — check before per-request
    "x-request-id",
    "x-trace-id",
];

/// Extract hash key with priority: HTTP headers > body fields > request content hash
///
/// Priority order:
/// 1. HTTP Headers: x-session-id, x-user-id, x-tenant-id, x-correlation-id, x-request-id, x-trace-id
/// 2. Body: user field (OpenAI format)
/// 3. Body: session_id (LMDeploy /generate format)
/// 4. Body: user_id (legacy)
/// 5. Fallback: hash of request body (long) or raw text (short)
pub(crate) fn extract_hash_key(
    request_text: Option<&str>,
    headers: Option<&RequestHeaders>,
) -> String {
    // 1. First priority: HTTP headers
    if let Some(hdrs) = headers {
        if let Some(key) = extract_hash_key_from_headers(hdrs) {
            return key;
        }
    }

    // 2. Second priority: Body fields
    if let Some(key) = extract_hash_key_from_body(request_text) {
        return key;
    }

    // 3. Final fallback: hash of request body
    let text = request_text.unwrap_or("");
    if text.len() > 100 {
        format!("request_hash:{:016x}", ConsistentHashPolicy::fbi_hash(text))
    } else {
        format!("request:{}", text)
    }
}

/// Extract hash key from HTTP headers
pub(crate) fn extract_hash_key_from_headers(headers: &RequestHeaders) -> Option<String> {
    for header_name in SESSION_HEADER_NAMES {
        if let Some(value) = headers.get(*header_name) {
            if !value.is_empty() {
                debug!(
                    "Hash key extraction: found session key in header '{}': {}",
                    header_name, value
                );
                return Some(format!("header:{}:{}", header_name, value));
            }
        }
    }
    None
}

/// Extract hash key from request body fields
///
/// Priority: user > session_id > user_id
pub(crate) fn extract_hash_key_from_body(request_text: Option<&str>) -> Option<String> {
    let text = request_text.unwrap_or("");
    if text.is_empty() {
        return None;
    }

    // 1. Try to extract direct user field (from OpenAI ChatCompletion/Completion requests)
    if let Some(user) = extract_field_value(text, "user") {
        debug!("Hash key extraction: found user field: {}", user);
        return Some(format!("user:{}", user));
    }

    // 2. Extract the LMDeploy top-level session_id field
    if let Some(session_id) = extract_field_value(text, "session_id") {
        return Some(format!("session:{}", session_id));
    }

    // 3. Fallback: try legacy user_id field
    if let Some(user_id) = extract_field_value(text, "user_id") {
        return Some(format!("user:{}", user_id));
    }

    None
}

/// Extract field value from JSON-like text (simple parser)
///
/// Supports double-quoted, single-quoted, and unquoted values.
pub(crate) fn extract_field_value(text: &str, field_name: &str) -> Option<String> {
    let patterns = [
        format!("\"{}\"", field_name),
        format!("'{}'", field_name),
        field_name.to_string(),
    ];

    for pattern in &patterns {
        if let Some(field_pos) = text.find(pattern) {
            let after_field = &text[field_pos + pattern.len()..];

            // Skip whitespace and look for colon
            let mut colon_pos = None;
            for (i, ch) in after_field.char_indices() {
                if ch == ':' {
                    colon_pos = Some(i);
                    break;
                } else if !ch.is_whitespace() {
                    break;
                }
            }

            if let Some(colon_idx) = colon_pos {
                let after_colon = &after_field[colon_idx + 1..];
                let trimmed = after_colon.trim_start();

                // Extract quoted string (double or single quotes)
                if trimmed.starts_with('"') {
                    if let Some(stripped) = trimmed.strip_prefix('"') {
                        if let Some(end_quote) = stripped.find('"') {
                            return Some(stripped[..end_quote].to_string());
                        }
                    }
                } else if trimmed.starts_with('\'') {
                    if let Some(stripped) = trimmed.strip_prefix('\'') {
                        if let Some(end_quote) = stripped.find('\'') {
                            return Some(stripped[..end_quote].to_string());
                        }
                    }
                } else {
                    // Unquoted value - extract until delimiter
                    let end_pos = trimmed
                        .find(&[',', ' ', '}', ']', '\n', '\r', '\t'][..])
                        .unwrap_or(trimmed.len());
                    if end_pos > 0 {
                        return Some(trimmed[..end_pos].to_string());
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // === extract_field_value tests ===

    #[test]
    fn test_extract_field_value_double_quoted() {
        let text = r#"{"session_id": "abc123", "prompt": "hello"}"#;
        assert_eq!(
            extract_field_value(text, "session_id"),
            Some("abc123".to_string())
        );
    }

    #[test]
    fn test_extract_field_value_single_quoted() {
        let text = r#"{'session_id': 'def456', 'prompt': 'world'}"#;
        assert_eq!(
            extract_field_value(text, "session_id"),
            Some("def456".to_string())
        );
    }

    #[test]
    fn test_extract_field_value_unquoted() {
        let text = r#"{"count": 42, "name": "test"}"#;
        assert_eq!(extract_field_value(text, "count"), Some("42".to_string()));
    }

    #[test]
    fn test_extract_field_value_missing() {
        let text = r#"{"other": "val"}"#;
        assert_eq!(extract_field_value(text, "session_id"), None);
    }

    #[test]
    fn test_extract_field_value_no_space_after_colon() {
        let text = r#"{"session_id":"compact_value"}"#;
        assert_eq!(
            extract_field_value(text, "session_id"),
            Some("compact_value".to_string())
        );
    }

    #[test]
    fn test_extract_field_value_multiple_fields() {
        let text = r#"{"user": "bob", "prompt": "hi", "session_id": "sess1"}"#;
        assert_eq!(extract_field_value(text, "user"), Some("bob".to_string()));
        assert_eq!(
            extract_field_value(text, "session_id"),
            Some("sess1".to_string())
        );
    }

    // === extract_hash_key_from_headers tests ===

    #[test]
    fn test_header_extraction_priority() {
        let mut headers = HashMap::new();
        headers.insert("x-request-id".to_string(), "req-1".to_string());
        headers.insert("x-session-id".to_string(), "sess-1".to_string());

        // x-session-id has higher priority than x-request-id
        let key = extract_hash_key_from_headers(&headers).unwrap();
        assert_eq!(key, "header:x-session-id:sess-1");
    }

    #[test]
    fn test_header_extraction_skips_empty() {
        let mut headers = HashMap::new();
        headers.insert("x-session-id".to_string(), "".to_string());
        headers.insert("x-user-id".to_string(), "user-1".to_string());

        let key = extract_hash_key_from_headers(&headers).unwrap();
        assert_eq!(key, "header:x-user-id:user-1");
    }

    #[test]
    fn test_header_extraction_no_match() {
        let mut headers = HashMap::new();
        headers.insert("x-custom-header".to_string(), "val".to_string());
        assert_eq!(extract_hash_key_from_headers(&headers), None);
    }

    // === extract_hash_key_from_body tests ===

    #[test]
    fn test_body_extraction_user_field() {
        let text = r#"{"user": "alice", "prompt": "hi"}"#;
        let key = extract_hash_key_from_body(Some(text)).unwrap();
        assert_eq!(key, "user:alice");
    }

    #[test]
    fn test_body_extraction_session_id() {
        let text = r#"{"session_id": "session123", "prompt": "hi"}"#;
        let key = extract_hash_key_from_body(Some(text)).unwrap();
        assert_eq!(key, "session:session123");
    }

    #[test]
    fn test_body_extraction_legacy_user_id() {
        let text = r#"{"user_id": "uid456", "prompt": "hi"}"#;
        let key = extract_hash_key_from_body(Some(text)).unwrap();
        assert_eq!(key, "user:uid456");
    }

    #[test]
    fn test_body_extraction_empty() {
        assert_eq!(extract_hash_key_from_body(None), None);
        assert_eq!(extract_hash_key_from_body(Some("")), None);
    }

    #[test]
    fn test_body_extraction_no_known_fields() {
        let text = r#"{"prompt": "hello", "model": "llama"}"#;
        assert_eq!(extract_hash_key_from_body(Some(text)), None);
    }

    // === extract_hash_key (top-level) tests ===

    #[test]
    fn test_hash_key_headers_over_body() {
        let mut headers = HashMap::new();
        headers.insert("x-session-id".to_string(), "from-header".to_string());
        let body = r#"{"session_id": "from-body"}"#;

        let key = extract_hash_key(Some(body), Some(&headers));
        assert_eq!(key, "header:x-session-id:from-header");
    }

    #[test]
    fn test_hash_key_fallback_short_text() {
        let key = extract_hash_key(Some("short"), None);
        assert_eq!(key, "request:short");
    }

    #[test]
    fn test_hash_key_fallback_long_text() {
        let long_text = "x".repeat(200);
        let key = extract_hash_key(Some(&long_text), None);
        assert!(key.starts_with("request_hash:"));
        assert_eq!(key.len(), "request_hash:".len() + 16); // 16 hex chars

        // Same long text should produce same hash
        let key2 = extract_hash_key(Some(&long_text), None);
        assert_eq!(key, key2);
    }

    #[test]
    fn test_hash_key_fallback_none() {
        let key = extract_hash_key(None, None);
        assert_eq!(key, "request:");
    }
}
