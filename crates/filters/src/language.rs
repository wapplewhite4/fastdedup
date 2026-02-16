//! Language detection and filtering
//!
//! Fast language detection using whatlang to filter documents based on language.
//! Supports multiple language configurations and confidence thresholds.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use whatlang::{detect, Lang};

/// Minimum text length for language detection (chars)
const MIN_DETECTION_LENGTH: usize = 50;

/// Language filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageFilterConfig {
    /// List of allowed language codes (ISO 639-3)
    pub allowed_languages: Vec<String>,
    /// Minimum confidence threshold (0.0 to 1.0)
    pub confidence_threshold: f64,
    /// Skip detection for short texts (chars)
    pub min_text_length: usize,
}

impl Default for LanguageFilterConfig {
    fn default() -> Self {
        Self {
            allowed_languages: vec!["eng".to_string()], // English only by default
            confidence_threshold: 0.5,
            min_text_length: MIN_DETECTION_LENGTH,
        }
    }
}

impl LanguageFilterConfig {
    /// Create English-only filter
    pub fn english_only() -> Self {
        Self {
            allowed_languages: vec!["eng".to_string()],
            confidence_threshold: 0.5,
            min_text_length: MIN_DETECTION_LENGTH,
        }
    }

    /// Create English + code filter
    /// Detects programming languages by common keywords
    pub fn english_with_code() -> Self {
        Self {
            allowed_languages: vec!["eng".to_string()],
            confidence_threshold: 0.3, // Lower threshold for code-heavy text
            min_text_length: MIN_DETECTION_LENGTH,
        }
    }

    /// Create multilingual filter (top 10 languages)
    pub fn multilingual() -> Self {
        Self {
            allowed_languages: vec![
                "eng".to_string(), // English
                "spa".to_string(), // Spanish
                "fra".to_string(), // French
                "deu".to_string(), // German
                "por".to_string(), // Portuguese
                "rus".to_string(), // Russian
                "jpn".to_string(), // Japanese
                "zho".to_string(), // Chinese
                "ara".to_string(), // Arabic
                "hin".to_string(), // Hindi
            ],
            confidence_threshold: 0.5,
            min_text_length: MIN_DETECTION_LENGTH,
        }
    }
}

/// Language detection and filtering
pub struct LanguageFilter {
    allowed_languages: HashSet<Lang>,
    confidence_threshold: f64,
    min_text_length: usize,
}

impl LanguageFilter {
    /// Create a new language filter from configuration
    pub fn new(config: LanguageFilterConfig) -> Result<Self> {
        let mut allowed_languages = HashSet::new();

        for lang_code in &config.allowed_languages {
            let lang = Self::parse_language_code(lang_code)?;
            allowed_languages.insert(lang);
        }

        if allowed_languages.is_empty() {
            return Err(anyhow!("At least one language must be specified"));
        }

        if config.confidence_threshold < 0.0 || config.confidence_threshold > 1.0 {
            return Err(anyhow!(
                "Confidence threshold must be between 0.0 and 1.0"
            ));
        }

        Ok(Self {
            allowed_languages,
            confidence_threshold: config.confidence_threshold,
            min_text_length: config.min_text_length,
        })
    }

    /// Create from a list of language codes
    pub fn from_codes(language_codes: Vec<String>) -> Result<Self> {
        Self::new(LanguageFilterConfig {
            allowed_languages: language_codes,
            ..Default::default()
        })
    }

    /// Check if text passes the language filter
    pub fn is_accepted(&self, text: &str) -> bool {
        // Skip very short texts
        if text.len() < self.min_text_length {
            return true; // Accept by default
        }

        // Check if text contains significant code patterns
        if self.looks_like_code(text) {
            return true; // Accept code
        }

        // Detect language
        match self.detect_with_confidence(text) {
            Some((lang, confidence)) => {
                self.allowed_languages.contains(&lang) && confidence >= self.confidence_threshold
            }
            None => false, // Reject if detection fails
        }
    }

    /// Detect language with confidence score
    pub fn detect_with_confidence(&self, text: &str) -> Option<(Lang, f64)> {
        detect(text).map(|info| (info.lang(), info.confidence()))
    }

    /// Detect if text looks like programming code
    fn looks_like_code(&self, text: &str) -> bool {
        let code_indicators = [
            "function", "class", "import", "return", "const", "let", "var",
            "def", "public", "private", "static", "void", "int", "string",
            "async", "await", "try", "catch", "throw", "=>", "->", "::", "==",
            "!=", "&&", "||", "++", "--",
        ];

        let text_lower = text.to_lowercase();
        let mut matches = 0;

        for indicator in &code_indicators {
            if text_lower.contains(indicator) {
                matches += 1;
            }
        }

        // If 3+ code indicators found, likely code
        matches >= 3
    }

    /// Parse language code (ISO 639-3)
    fn parse_language_code(code: &str) -> Result<Lang> {
        match code.to_lowercase().as_str() {
            "eng" | "en" => Ok(Lang::Eng),
            "spa" | "es" => Ok(Lang::Spa),
            "fra" | "fr" => Ok(Lang::Fra),
            "deu" | "de" => Ok(Lang::Deu),
            "por" | "pt" => Ok(Lang::Por),
            "rus" | "ru" => Ok(Lang::Rus),
            "jpn" | "ja" => Ok(Lang::Jpn),
            "zho" | "zh" => Ok(Lang::Cmn),
            "ara" | "ar" => Ok(Lang::Ara),
            "hin" | "hi" => Ok(Lang::Hin),
            "ita" | "it" => Ok(Lang::Ita),
            "nld" | "nl" => Ok(Lang::Nld),
            "pol" | "pl" => Ok(Lang::Pol),
            "tur" | "tr" => Ok(Lang::Tur),
            "vie" | "vi" => Ok(Lang::Vie),
            "kor" | "ko" => Ok(Lang::Kor),
            "swe" | "sv" => Ok(Lang::Swe),
            "dan" | "da" => Ok(Lang::Dan),
            "fin" | "fi" => Ok(Lang::Fin),
            "nor" | "no" => Ok(Lang::Nob),
            _ => Err(anyhow!("Unsupported language code: {}", code)),
        }
    }

    /// Get statistics about detected languages in a batch
    pub fn batch_stats(&self, texts: &[&str]) -> LanguageStats {
        let mut stats = LanguageStats::default();

        for text in texts {
            stats.total += 1;

            if text.len() < self.min_text_length {
                stats.skipped_short += 1;
                continue;
            }

            if self.looks_like_code(text) {
                stats.code_detected += 1;
                continue;
            }

            match self.detect_with_confidence(text) {
                Some((lang, confidence)) => {
                    if self.allowed_languages.contains(&lang)
                        && confidence >= self.confidence_threshold
                    {
                        stats.accepted += 1;
                    } else {
                        stats.rejected += 1;
                    }
                    *stats.lang_distribution.entry(lang).or_insert(0) += 1;
                }
                None => {
                    stats.detection_failed += 1;
                }
            }
        }

        stats
    }
}

/// Statistics about language detection
#[derive(Debug, Default)]
pub struct LanguageStats {
    pub total: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub skipped_short: usize,
    pub code_detected: usize,
    pub detection_failed: usize,
    pub lang_distribution: std::collections::HashMap<Lang, usize>,
}

impl LanguageStats {
    /// Get acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.accepted as f64 / self.total as f64
    }

    /// Get rejection rate
    pub fn rejection_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.rejected as f64 / self.total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_english_detection() {
        let filter = LanguageFilter::new(LanguageFilterConfig::english_only()).unwrap();

        let english_text = "This is a sample English text that should be detected correctly.";
        assert!(filter.is_accepted(english_text));

        let spanish_text = "Este es un texto en español que debería ser detectado correctamente.";
        assert!(!filter.is_accepted(spanish_text));
    }

    #[test]
    fn test_short_text_skipped() {
        let filter = LanguageFilter::new(LanguageFilterConfig::english_only()).unwrap();

        // Very short text - should be accepted (skipped)
        let short_text = "Hi";
        assert!(filter.is_accepted(short_text));
    }

    #[test]
    fn test_code_detection() {
        let filter = LanguageFilter::new(LanguageFilterConfig::english_with_code()).unwrap();

        let code_text = r#"
            function calculateTotal(items) {
                return items.reduce((sum, item) => sum + item.price, 0);
            }
        "#;

        assert!(filter.is_accepted(code_text));
    }

    #[test]
    fn test_multilingual() {
        let filter = LanguageFilter::new(LanguageFilterConfig::multilingual()).unwrap();

        let texts = vec![
            "This is English text.",
            "C'est un texte français.",
            "Dies ist ein deutscher Text.",
            "Este es un texto español.",
        ];

        for text in texts {
            assert!(filter.is_accepted(text));
        }
    }

    #[test]
    fn test_confidence_threshold() {
        let filter = LanguageFilter::new(LanguageFilterConfig {
            allowed_languages: vec!["eng".to_string()],
            confidence_threshold: 0.9, // Very high threshold
            min_text_length: 50,
        })
        .unwrap();

        let clear_english = "The quick brown fox jumps over the lazy dog. This is clearly English text.";
        assert!(filter.is_accepted(clear_english));
    }

    #[test]
    fn test_language_code_parsing() {
        assert!(LanguageFilter::parse_language_code("eng").is_ok());
        assert!(LanguageFilter::parse_language_code("en").is_ok());
        assert!(LanguageFilter::parse_language_code("spa").is_ok());
        assert!(LanguageFilter::parse_language_code("es").is_ok());
        assert!(LanguageFilter::parse_language_code("invalid").is_err());
    }

    #[test]
    fn test_batch_stats() {
        let filter = LanguageFilter::new(LanguageFilterConfig::english_only()).unwrap();

        let texts = vec![
            "This is English text that meets the minimum length requirement.",
            "C'est français avec suffisamment de longueur pour la détection.",
            "Hi", // Too short
            "function test() { return 42; } // Code with function keyword async await",
        ];

        let stats = filter.batch_stats(&texts);
        assert_eq!(stats.total, 4);
        assert!(stats.accepted > 0);
        assert!(stats.rejected > 0);
    }

    #[test]
    fn test_detect_with_confidence() {
        let filter = LanguageFilter::new(LanguageFilterConfig::english_only()).unwrap();

        let english_text = "This is a clear English sentence with enough words for detection.";
        let result = filter.detect_with_confidence(english_text);

        assert!(result.is_some());
        let (lang, confidence) = result.unwrap();
        assert_eq!(lang, Lang::Eng);
        assert!(confidence > 0.5);
    }
}
