use serde_json::{Map, Number, Value};
use std::convert::TryFrom;

#[derive(Debug, thiserror::Error)]
pub enum RequireError {
    #[error("Expected object, found {0}")]
    NotAnObject(String),
    #[error("Missing required field: {0}")]
    MissingField(String),
    #[error("Expected field '{field}' to be {expected}, but got {actual}")]
    MismatchedFieldType {
        field: String,
        expected: String,
        actual: String,
    },
}

fn value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

pub trait JsonTypeName {
    fn type_name() -> &'static str;
}

macro_rules! json_type_name {
    ($type:ty, $name:expr) => {
        impl JsonTypeName for $type {
            fn type_name() -> &'static str {
                $name
            }
        }
    };
}

json_type_name!(bool, "boolean");
json_type_name!(i64, "i64");
json_type_name!(u16, "u16");
json_type_name!(u64, "u64");
json_type_name!(f64, "f64");
json_type_name!(&str, "string");
json_type_name!(&Vec<Value>, "array");
json_type_name!(&Map<String, Value>, "object");

fn json_type_name<T: JsonTypeName>() -> &'static str {
    T::type_name()
}

pub trait AsU16 {
    fn as_u16(&self) -> Option<u16>;
}

impl AsU16 for Number {
    fn as_u16(&self) -> Option<u16> {
        let ui: u64 = match self.as_u64() {
            Some(x) => x,
            None => return None,
        };

        u16::try_from(ui).ok()
    }
}

pub struct ValueRef<'a>(&'a Value);

impl<'a> TryFrom<ValueRef<'a>> for bool {
    type Error = ();

    fn try_from(value: ValueRef<'a>) -> Result<Self, Self::Error> {
        match value {
            ValueRef(Value::Bool(b)) => Ok(*b),
            _ => Err(()),
        }
    }
}

impl<'a> TryFrom<ValueRef<'a>> for i64 {
    type Error = ();

    fn try_from(value: ValueRef<'a>) -> Result<Self, Self::Error> {
        match value {
            ValueRef(Value::Number(n)) => n.as_i64().ok_or(()),
            _ => Err(()),
        }
    }
}

impl<'a> TryFrom<ValueRef<'a>> for u16 {
    type Error = ();

    fn try_from(value: ValueRef<'a>) -> Result<Self, Self::Error> {
        match value {
            ValueRef(Value::Number(n)) => n.as_u16().ok_or(()),
            _ => Err(()),
        }
    }
}

impl<'a> TryFrom<ValueRef<'a>> for u64 {
    type Error = ();

    fn try_from(value: ValueRef<'a>) -> Result<Self, Self::Error> {
        match value {
            ValueRef(Value::Number(n)) => n.as_u64().ok_or(()),
            _ => Err(()),
        }
    }
}

impl<'a> TryFrom<ValueRef<'a>> for f64 {
    type Error = ();

    fn try_from(value: ValueRef<'a>) -> Result<Self, Self::Error> {
        match value {
            ValueRef(Value::Number(n)) => n.as_f64().ok_or(()),
            _ => Err(()),
        }
    }
}

impl<'a> TryFrom<ValueRef<'a>> for &'a str {
    type Error = ();

    fn try_from(value: ValueRef<'a>) -> Result<Self, Self::Error> {
        match value {
            ValueRef(Value::String(s)) => Ok(&s),
            _ => Err(()),
        }
    }
}

impl<'a> TryFrom<ValueRef<'a>> for &'a Vec<Value> {
    type Error = ();

    fn try_from(value: ValueRef<'a>) -> Result<Self, Self::Error> {
        match value {
            ValueRef(Value::Array(a)) => Ok(&a),
            _ => Err(()),
        }
    }
}

impl<'a> TryFrom<ValueRef<'a>> for &'a Map<String, Value> {
    type Error = ();

    fn try_from(value: ValueRef<'a>) -> Result<Self, Self::Error> {
        match value {
            ValueRef(Value::Object(o)) => Ok(&o),
            _ => Err(()),
        }
    }
}

pub trait RequireField {
    fn require<'a, T>(&'a self, field: &str) -> Result<T, RequireError>
    where
        T: JsonTypeName + TryFrom<ValueRef<'a>>;
}

impl RequireField for Map<String, Value> {
    fn require<'a, T>(&'a self, field: &str) -> Result<T, RequireError>
    where
        T: JsonTypeName + TryFrom<ValueRef<'a>>,
    {
        match self.get(field) {
            Some(value) => {
                ValueRef(value)
                    .try_into()
                    .map_err(|_| RequireError::MismatchedFieldType {
                        field: field.to_string(),
                        expected: json_type_name::<T>().to_string(),
                        actual: value_type(value).to_string(),
                    })
            }
            None => Err(RequireError::MissingField(field.to_string())),
        }
    }
}

impl RequireField for Value {
    fn require<'a, T>(&'a self, field: &str) -> Result<T, RequireError>
    where
        T: JsonTypeName + TryFrom<ValueRef<'a>>,
    {
        let obj = match self {
            Value::Object(obj) => obj,
            other => return Err(RequireError::NotAnObject(value_type(other).to_string())),
        };

        obj.require(field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Number;

    #[test]
    fn test_require_bool() {
        let json_str = r#"
        {
            "bool": true
        }
        "#;

        let json: Value = serde_json::from_str(&json_str).unwrap();

        let b: bool = json.require("bool").unwrap();
        assert_eq!(b, true);
    }

    #[test]
    fn test_require_i64() {
        let json_str = r#"
        {
            "i64": -17,
            "not_i64": 3.14
        }
        "#;

        let json: Value = serde_json::from_str(&json_str).unwrap();

        let x: i64 = json.require("i64").unwrap();
        assert_eq!(x, -17);

        let y = json.require::<i64>("not_i64");
        assert!(y.is_err());
    }

    #[test]
    fn test_require_u16() {
        let json_str = r#"
        {
            "u16": 65535,
            "not_u16": 100000
        }
        "#;

        let json: Value = serde_json::from_str(&json_str).unwrap();

        let x: u16 = json.require("u16").unwrap();
        assert_eq!(x, u16::MAX);

        let y = json.require::<u16>("not_u64");
        assert!(y.is_err());
    }

    #[test]
    fn test_require_u64() {
        let json_str = r#"
        {
            "u64": 17,
            "not_u64": -42
        }
        "#;

        let json: Value = serde_json::from_str(&json_str).unwrap();

        let x: u64 = json.require("u64").unwrap();
        assert_eq!(x, 17);

        let y = json.require::<u64>("not_u64");
        assert!(y.is_err());
    }

    #[test]
    fn test_require_f64() {
        let json_str = r#"
        {
            "f64": 3.14159,
            "also_f64": -42
        }
        "#;

        let json: Value = serde_json::from_str(&json_str).unwrap();

        let x: f64 = json.require("f64").unwrap();
        assert_eq!(x, 3.14159);

        let x: f64 = json.require("also_f64").unwrap();
        assert_eq!(x, -42.0);
    }

    #[test]
    fn test_require_string() {
        let json_str = r#"
        {
            "field": "some_string"
        }
        "#;

        let json: Value = serde_json::from_str(&json_str).unwrap();
        let value: &str = json.require("field").unwrap();

        assert_eq!(value, "some_string");
    }

    #[test]
    fn test_require_array() {
        let json_str = r#"
        {
            "field": [1, "b"]
        }
        "#;

        let json: Value = serde_json::from_str(&json_str).unwrap();
        let a: &Vec<Value> = json.require("field").unwrap();

        assert_eq!(a[0], Value::Number(Number::from(1)));
        assert_eq!(a[1], Value::String("b".to_string()));
    }

    #[test]
    fn test_require_object() {
        let json_str = r#"
        {
            "field": {
                "subfield": "subvalue"
            }
        }
        "#;

        let json: Value = serde_json::from_str(&json_str).unwrap();
        let o: &Map<String, Value> = json.require("field").unwrap();

        assert_eq!(o["subfield"], "subvalue");
    }
}
