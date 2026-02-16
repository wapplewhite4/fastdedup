//! Length-based filtering for text content

use serde_json::Value;

/// Filter configuration for length-based filtering
#[derive(Debug, Clone)]
pub struct LengthFilterConfig {
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub field_name: String,
}

impl Default for LengthFilterConfig {
    fn default() -> Self {
        Self {
            min_length: None,
            max_length: None,
            field_name: "text".to_string(),
        }
    }
}

/// Check if a record passes length filter
pub fn passes_length_filter(value: &Value, config: &LengthFilterConfig) -> bool {
    if let Some(text) = value.get(&config.field_name).and_then(|v| v.as_str()) {
        let length = text.len();

        if let Some(min) = config.min_length {
            if length < min {
                return false;
            }
        }

        if let Some(max) = config.max_length {
            if length > max {
                return false;
            }
        }

        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_length_filter_min() {
        let config = LengthFilterConfig {
            min_length: Some(5),
            max_length: None,
            field_name: "text".to_string(),
        };

        assert!(passes_length_filter(&json!({"text": "hello"}), &config));
        assert!(passes_length_filter(&json!({"text": "hello world"}), &config));
        assert!(!passes_length_filter(&json!({"text": "hi"}), &config));
    }

    #[test]
    fn test_length_filter_max() {
        let config = LengthFilterConfig {
            min_length: None,
            max_length: Some(10),
            field_name: "text".to_string(),
        };

        assert!(passes_length_filter(&json!({"text": "hello"}), &config));
        assert!(!passes_length_filter(&json!({"text": "hello world!"}), &config));
    }
}
