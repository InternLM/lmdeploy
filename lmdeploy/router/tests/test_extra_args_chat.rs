//! Tests that unknown/extra fields in ChatCompletionRequest are preserved through serde roundtrip.
use lmdeploy_router_rs::protocols::spec::ChatCompletionRequest;

#[test]
fn test_extra_fields_preserved_on_deserialize() {
    let json = r#"{
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
        "return_token_ids": true,
        "some_custom_param": "value"
    }"#;

    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();

    assert!(
        req.other.contains_key("return_token_ids"),
        "return_token_ids should be captured in other"
    );
    assert!(
        req.other.contains_key("some_custom_param"),
        "some_custom_param should be captured in other"
    );
    assert_eq!(
        req.other["some_custom_param"],
        serde_json::Value::String("value".to_string())
    );
}

#[test]
fn test_extra_fields_survive_roundtrip() {
    let json = r#"{
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
        "return_token_ids": true
    }"#;

    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    let serialized = serde_json::to_string(&req).unwrap();
    let roundtrip: ChatCompletionRequest = serde_json::from_str(&serialized).unwrap();

    assert_eq!(roundtrip.other["return_token_ids"], serde_json::json!(true));
}

#[test]
fn test_extra_fields_appear_at_top_level_in_serialized_json() {
    let json = r#"{
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "return_token_ids": true
    }"#;

    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    let value: serde_json::Value = serde_json::to_value(&req).unwrap();

    assert!(
        value.get("return_token_ids").is_some(),
        "return_token_ids should be at the top level of the serialized JSON"
    );
    assert_eq!(value["return_token_ids"], serde_json::json!(true));
}

#[test]
fn test_no_other_fields_gives_empty_map() {
    let json = r#"{
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}]
    }"#;

    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert!(req.other.is_empty());
}

#[test]
fn test_lmdeploy_input_ids_is_typed_not_flattened() {
    let json = r#"{
        "model": "test-model",
        "messages": [],
        "input_ids": [1, 2, 3],
        "return_token_ids": true
    }"#;

    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.input_ids, Some(vec![1, 2, 3]));
    assert!(!req.other.contains_key("input_ids"));
    assert_eq!(req.other["return_token_ids"], serde_json::json!(true));
}
