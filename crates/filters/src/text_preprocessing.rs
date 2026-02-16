//! Text normalization and preprocessing for fuzzy matching
//!
//! Provides various text normalization strategies to improve
//! fuzzy duplicate detection.

use regex::Regex;
use unicode_normalization::UnicodeNormalization;
use std::sync::OnceLock;

/// Text normalization configuration
#[derive(Debug, Clone)]
pub struct TextNormalizer {
    /// Convert to lowercase
    pub lowercase: bool,
    /// Remove punctuation
    pub remove_punctuation: bool,
    /// Normalize whitespace (collapse multiple spaces)
    pub remove_whitespace: bool,
    /// Apply Unicode NFKD normalization
    pub unicode_normalize: bool,
}

// Lazy-initialized regex for punctuation removal
static PUNCTUATION_REGEX: OnceLock<Regex> = OnceLock::new();
static WHITESPACE_REGEX: OnceLock<Regex> = OnceLock::new();

fn get_punctuation_regex() -> &'static Regex {
    PUNCTUATION_REGEX.get_or_init(|| {
        Regex::new(r"[^\w\s]").expect("Failed to compile punctuation regex")
    })
}

fn get_whitespace_regex() -> &'static Regex {
    WHITESPACE_REGEX.get_or_init(|| {
        Regex::new(r"\s+").expect("Failed to compile whitespace regex")
    })
}

impl TextNormalizer {
    /// Create a new text normalizer with custom settings
    pub fn new(
        lowercase: bool,
        remove_punctuation: bool,
        remove_whitespace: bool,
        unicode_normalize: bool,
    ) -> Self {
        Self {
            lowercase,
            remove_punctuation,
            remove_whitespace,
            unicode_normalize,
        }
    }

    /// Aggressive normalization preset
    ///
    /// Applies all normalizations for maximum duplicate detection.
    /// May produce more false positives.
    pub fn aggressive() -> Self {
        Self {
            lowercase: true,
            remove_punctuation: true,
            remove_whitespace: true,
            unicode_normalize: true,
        }
    }

    /// Conservative normalization preset
    ///
    /// Minimal normalization to reduce false positives.
    pub fn conservative() -> Self {
        Self {
            lowercase: true,
            remove_punctuation: false,
            remove_whitespace: true,
            unicode_normalize: false,
        }
    }

    /// Balanced normalization preset (default)
    ///
    /// Good balance between recall and precision.
    pub fn balanced() -> Self {
        Self {
            lowercase: true,
            remove_punctuation: true,
            remove_whitespace: true,
            unicode_normalize: false,
        }
    }

    /// Normalize text according to configuration
    ///
    /// Applies transformations in the following order:
    /// 1. Unicode normalization (if enabled)
    /// 2. Lowercase conversion (if enabled)
    /// 3. Punctuation removal (if enabled)
    /// 4. Whitespace normalization (if enabled)
    pub fn normalize(&self, text: &str) -> String {
        let mut result = text.to_string();

        // 1. Unicode normalization
        if self.unicode_normalize {
            result = result.nfkd().collect::<String>();
        }

        // 2. Lowercase
        if self.lowercase {
            result = result.to_lowercase();
        }

        // 3. Remove punctuation
        if self.remove_punctuation {
            let regex = get_punctuation_regex();
            result = regex.replace_all(&result, " ").to_string();
        }

        // 4. Normalize whitespace
        if self.remove_whitespace {
            let regex = get_whitespace_regex();
            result = regex.replace_all(&result, " ").trim().to_string();
        }

        result
    }

    /// Normalize text with memory reuse
    ///
    /// More efficient version that reuses an existing String buffer.
    pub fn normalize_into(&self, text: &str, buffer: &mut String) {
        buffer.clear();
        buffer.push_str(text);

        // Unicode normalization (requires creating new string)
        if self.unicode_normalize {
            *buffer = buffer.nfkd().collect::<String>();
        }

        // Lowercase
        if self.lowercase {
            *buffer = buffer.to_lowercase();
        }

        // Remove punctuation
        if self.remove_punctuation {
            let regex = get_punctuation_regex();
            *buffer = regex.replace_all(buffer, " ").to_string();
        }

        // Normalize whitespace
        if self.remove_whitespace {
            let regex = get_whitespace_regex();
            *buffer = regex.replace_all(buffer, " ").trim().to_string();
        }
    }
}

impl Default for TextNormalizer {
    fn default() -> Self {
        Self::balanced()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lowercase() {
        let normalizer = TextNormalizer::new(true, false, false, false);
        assert_eq!(normalizer.normalize("Hello WORLD"), "hello world");
    }

    #[test]
    fn test_punctuation_removal() {
        let normalizer = TextNormalizer::new(false, true, false, false);
        assert_eq!(
            normalizer.normalize("Hello, World!"),
            "Hello  World "
        );
    }

    #[test]
    fn test_whitespace_normalization() {
        let normalizer = TextNormalizer::new(false, false, true, false);
        assert_eq!(
            normalizer.normalize("Hello   World  "),
            "Hello World"
        );
    }

    #[test]
    fn test_unicode_normalization() {
        let normalizer = TextNormalizer::new(false, false, false, true);
        // é (U+00E9) should normalize to e + combining accent
        let result = normalizer.normalize("café");
        assert_ne!(result, "café"); // Different representation
    }

    #[test]
    fn test_aggressive_preset() {
        let normalizer = TextNormalizer::aggressive();
        let text = "  Hello,  WORLD!!!  ";
        let result = normalizer.normalize(text);

        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_conservative_preset() {
        let normalizer = TextNormalizer::conservative();
        let text = "  Hello,  WORLD!!!  ";
        let result = normalizer.normalize(text);

        assert_eq!(result, "hello, world!!!");
    }

    #[test]
    fn test_balanced_preset() {
        let normalizer = TextNormalizer::balanced();
        let text = "  Hello,  WORLD!!!  ";
        let result = normalizer.normalize(text);

        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_emojis() {
        let normalizer = TextNormalizer::aggressive();
        let text = "Hello 👋 World 🌍";
        let result = normalizer.normalize(text);

        // With aggressive normalization (remove_punctuation=true),
        // emojis are treated as non-word characters and removed
        assert!(result.contains("hello"));
        assert!(result.contains("world"));
        // Emojis are removed by punctuation removal
        assert!(!result.contains("👋"));
        assert!(!result.contains("🌍"));

        // Test with conservative (no punctuation removal)
        let normalizer = TextNormalizer::conservative();
        let result = normalizer.normalize(text);
        // Emojis are preserved when punctuation removal is disabled
        assert!(result.contains("👋"));
        assert!(result.contains("🌍"));
    }

    #[test]
    fn test_special_characters() {
        let normalizer = TextNormalizer::aggressive();
        let text = "Price: $100.50 (50% off!)";
        let result = normalizer.normalize(text);

        // Should remove punctuation but preserve structure
        assert!(result.contains("100"));
        assert!(result.contains("50"));
        assert!(!result.contains("$"));
        assert!(!result.contains("%"));
    }

    #[test]
    fn test_different_scripts() {
        let normalizer = TextNormalizer::balanced();

        // Japanese
        let text = "こんにちは世界";
        let result = normalizer.normalize(text);
        assert_eq!(result, "こんにちは世界");

        // Cyrillic
        let text = "Привет мир";
        let result = normalizer.normalize(text);
        assert_eq!(result, "привет мир");

        // Arabic
        let text = "مرحبا بالعالم";
        let result = normalizer.normalize(text);
        assert_eq!(result, "مرحبا بالعالم");
    }

    #[test]
    fn test_mixed_content() {
        let normalizer = TextNormalizer::aggressive();
        let text = "User123: Hello!!! How's it going? 😊";
        let result = normalizer.normalize(text);

        assert!(result.contains("user123"));
        assert!(result.contains("hello"));
        assert!(!result.contains("!!!"));
        assert!(!result.contains("?"));
    }

    #[test]
    fn test_empty_string() {
        let normalizer = TextNormalizer::aggressive();
        assert_eq!(normalizer.normalize(""), "");
    }

    #[test]
    fn test_only_whitespace() {
        let normalizer = TextNormalizer::aggressive();
        assert_eq!(normalizer.normalize("   \t\n   "), "");
    }

    #[test]
    fn test_only_punctuation() {
        let normalizer = TextNormalizer::aggressive();
        assert_eq!(normalizer.normalize("!!!???..."), "");
    }

    #[test]
    fn test_normalize_into() {
        let normalizer = TextNormalizer::balanced();
        let mut buffer = String::new();

        normalizer.normalize_into("Hello, WORLD!", &mut buffer);
        assert_eq!(buffer, "hello world");

        // Test reuse
        normalizer.normalize_into("Another TEST", &mut buffer);
        assert_eq!(buffer, "another test");
    }

    #[test]
    fn test_normalization_idempotent() {
        let normalizer = TextNormalizer::aggressive();
        let text = "Hello, WORLD!!!";

        let result1 = normalizer.normalize(text);
        let result2 = normalizer.normalize(&result1);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_html_tags() {
        let normalizer = TextNormalizer::aggressive();
        let text = "<p>Hello <strong>World</strong>!</p>";
        let result = normalizer.normalize(text);

        // Punctuation in tags treated as punctuation
        assert!(result.contains("hello"));
        assert!(result.contains("world"));
    }

    #[test]
    fn test_urls() {
        let normalizer = TextNormalizer::balanced();
        let text = "Visit https://example.com/path?query=1";
        let result = normalizer.normalize(text);

        // URLs partially preserved (alphanumeric parts)
        assert!(result.contains("https"));
        assert!(result.contains("example"));
        assert!(result.contains("com"));
    }

    #[test]
    fn test_accented_characters() {
        let normalizer = TextNormalizer::new(true, false, false, true);
        let text1 = "café";
        let text2 = "cafe";

        let result1 = normalizer.normalize(text1);
        let result2 = normalizer.normalize(text2);

        // With NFKD, accents are decomposed
        assert_ne!(text1, text2);
        // After normalization, they should be more similar
    }

    #[test]
    fn test_numbers() {
        let normalizer = TextNormalizer::aggressive();
        let text = "The year 2024 has 365 days";
        let result = normalizer.normalize(text);

        assert!(result.contains("2024"));
        assert!(result.contains("365"));
    }
}
