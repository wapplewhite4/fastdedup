//! Quality scoring and filtering for text data
//!
//! This module provides comprehensive quality metrics including:
//! - Length and word count constraints
//! - Repetition detection (n-gram analysis)
//! - Unique word ratio
//! - URL and HTML density
//! - Special character ratio
//! - Profanity filtering (optional)

use regex::Regex;
use rustrict::CensorStr;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Configuration for quality scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityConfig {
    // Length constraints
    pub min_length: usize,
    pub max_length: usize,
    pub min_word_count: usize,
    pub max_word_count: usize,

    // Content quality thresholds
    pub max_repetition_ratio: f64,
    pub min_unique_words_ratio: f64,
    pub max_url_ratio: f64,
    pub max_special_char_ratio: f64,

    // Optional filters
    pub reject_html: bool,
    pub filter_profanity: bool,
    pub min_avg_word_length: f64,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            min_length: 50,
            max_length: 100_000,
            min_word_count: 10,
            max_word_count: 10_000,
            max_repetition_ratio: 0.3,
            min_unique_words_ratio: 0.3,
            max_url_ratio: 0.1,
            max_special_char_ratio: 0.3,
            reject_html: true,
            filter_profanity: false,
            min_avg_word_length: 2.5,
        }
    }
}

impl QualityConfig {
    /// Strict quality config for high-quality datasets
    pub fn strict() -> Self {
        Self {
            min_length: 100,
            max_length: 50_000,
            min_word_count: 20,
            max_word_count: 5_000,
            max_repetition_ratio: 0.2,
            min_unique_words_ratio: 0.5,
            max_url_ratio: 0.05,
            max_special_char_ratio: 0.2,
            reject_html: true,
            filter_profanity: true,
            min_avg_word_length: 3.0,
        }
    }

    /// Lenient quality config for more permissive filtering
    pub fn lenient() -> Self {
        Self {
            min_length: 20,
            max_length: 200_000,
            min_word_count: 5,
            max_word_count: 20_000,
            max_repetition_ratio: 0.5,
            min_unique_words_ratio: 0.2,
            max_url_ratio: 0.2,
            max_special_char_ratio: 0.4,
            reject_html: false,
            filter_profanity: false,
            min_avg_word_length: 2.0,
        }
    }
}

/// Quality scorer for text documents
pub struct QualityScorer {
    config: QualityConfig,
    url_regex: Regex,
    html_tag_regex: Regex,
}

impl QualityScorer {
    /// Create a new quality scorer with configuration
    pub fn new(config: QualityConfig) -> Self {
        let url_regex = Regex::new(
            r"(?i)\b(?:https?://|www\.)[^\s<>]+\b|\b\w+\.(com|org|net|edu|gov)\b"
        ).unwrap();

        let html_tag_regex = Regex::new(r"<[^>]+>").unwrap();

        Self {
            config,
            url_regex,
            html_tag_regex,
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(QualityConfig::default())
    }

    /// Score a text and return quality metrics
    pub fn score(&self, text: &str) -> QualityScore {
        let length = text.len();
        let words = Self::tokenize_words(text);
        let word_count = words.len();

        // Compute metrics
        let unique_word_ratio = self.compute_unique_words_ratio(&words);
        let repetition_ratio = self.compute_repetition_ratio(text);
        let url_ratio = self.compute_url_ratio(text);
        let special_char_ratio = self.compute_special_char_ratio(text);
        let avg_word_length = self.compute_avg_word_length(&words);
        let has_html = self.contains_html(text);
        let has_profanity = if self.config.filter_profanity {
            text.is_inappropriate()
        } else {
            false
        };

        // Check against thresholds
        let mut reasons = Vec::new();
        let mut passes = true;

        if length < self.config.min_length {
            passes = false;
            reasons.push(format!("Length {} < min {}", length, self.config.min_length));
        }

        if length > self.config.max_length {
            passes = false;
            reasons.push(format!("Length {} > max {}", length, self.config.max_length));
        }

        if word_count < self.config.min_word_count {
            passes = false;
            reasons.push(format!(
                "Word count {} < min {}",
                word_count, self.config.min_word_count
            ));
        }

        if word_count > self.config.max_word_count {
            passes = false;
            reasons.push(format!(
                "Word count {} > max {}",
                word_count, self.config.max_word_count
            ));
        }

        if repetition_ratio > self.config.max_repetition_ratio {
            passes = false;
            reasons.push(format!(
                "Repetition ratio {:.2} > max {:.2}",
                repetition_ratio, self.config.max_repetition_ratio
            ));
        }

        if unique_word_ratio < self.config.min_unique_words_ratio {
            passes = false;
            reasons.push(format!(
                "Unique word ratio {:.2} < min {:.2}",
                unique_word_ratio, self.config.min_unique_words_ratio
            ));
        }

        if url_ratio > self.config.max_url_ratio {
            passes = false;
            reasons.push(format!(
                "URL ratio {:.2} > max {:.2}",
                url_ratio, self.config.max_url_ratio
            ));
        }

        if special_char_ratio > self.config.max_special_char_ratio {
            passes = false;
            reasons.push(format!(
                "Special char ratio {:.2} > max {:.2}",
                special_char_ratio, self.config.max_special_char_ratio
            ));
        }

        if avg_word_length < self.config.min_avg_word_length {
            passes = false;
            reasons.push(format!(
                "Avg word length {:.2} < min {:.2}",
                avg_word_length, self.config.min_avg_word_length
            ));
        }

        if self.config.reject_html && has_html {
            passes = false;
            reasons.push("Contains HTML tags".to_string());
        }

        if has_profanity {
            passes = false;
            reasons.push("Contains profanity".to_string());
        }

        if passes {
            reasons.push("Passed all quality checks".to_string());
        }

        QualityScore {
            passes,
            length,
            word_count,
            unique_word_ratio,
            repetition_ratio,
            url_ratio,
            special_char_ratio,
            avg_word_length,
            has_html,
            has_profanity,
            reasons,
        }
    }

    /// Tokenize text into words
    fn tokenize_words(text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|w| !w.is_empty())
            .collect()
    }

    /// Compute ratio of unique words to total words
    fn compute_unique_words_ratio(&self, words: &[String]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }

        let unique_words: HashSet<&String> = words.iter().collect();
        unique_words.len() as f64 / words.len() as f64
    }

    /// Compute repetition ratio using n-grams (3-5 word sequences)
    fn compute_repetition_ratio(&self, text: &str) -> f64 {
        let words = Self::tokenize_words(text);
        if words.len() < 5 {
            return 0.0;
        }

        let mut ngram_counts: HashMap<Vec<String>, usize> = HashMap::new();
        let mut total_ngrams = 0;
        let mut repeated_ngrams = 0;

        // Use 3-grams and 4-grams
        for n in 3..=4 {
            for window in words.windows(n) {
                let ngram = window.to_vec();
                total_ngrams += 1;

                let count = ngram_counts.entry(ngram).or_insert(0);
                *count += 1;

                if *count > 1 {
                    repeated_ngrams += 1;
                }
            }
        }

        if total_ngrams == 0 {
            0.0
        } else {
            repeated_ngrams as f64 / total_ngrams as f64
        }
    }

    /// Compute ratio of URL characters to total characters
    fn compute_url_ratio(&self, text: &str) -> f64 {
        if text.is_empty() {
            return 0.0;
        }

        let url_chars: usize = self
            .url_regex
            .find_iter(text)
            .map(|m| m.as_str().len())
            .sum();

        url_chars as f64 / text.len() as f64
    }

    /// Compute ratio of special (non-alphanumeric) characters
    fn compute_special_char_ratio(&self, text: &str) -> f64 {
        if text.is_empty() {
            return 0.0;
        }

        let special_chars = text
            .chars()
            .filter(|c| !c.is_alphanumeric() && !c.is_whitespace())
            .count();

        special_chars as f64 / text.len() as f64
    }

    /// Compute average word length
    fn compute_avg_word_length(&self, words: &[String]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }

        let total_length: usize = words.iter().map(|w| w.len()).sum();
        total_length as f64 / words.len() as f64
    }

    /// Check if text contains HTML tags
    fn contains_html(&self, text: &str) -> bool {
        self.html_tag_regex.is_match(text)
            || text.contains("<!DOCTYPE")
            || text.contains("<html")
            || text.contains("</html>")
    }

    /// Batch score multiple texts
    pub fn batch_score(&self, texts: &[&str]) -> Vec<QualityScore> {
        texts.iter().map(|text| self.score(text)).collect()
    }

    /// Get statistics from batch scoring
    pub fn batch_stats(&self, texts: &[&str]) -> QualityStats {
        let scores = self.batch_score(texts);

        let total = scores.len();
        let passed = scores.iter().filter(|s| s.passes).count();
        let failed = total - passed;

        let mut failure_reasons: HashMap<String, usize> = HashMap::new();
        for score in &scores {
            if !score.passes {
                for reason in &score.reasons {
                    *failure_reasons.entry(reason.clone()).or_insert(0) += 1;
                }
            }
        }

        QualityStats {
            total,
            passed,
            failed,
            pass_rate: if total > 0 {
                passed as f64 / total as f64
            } else {
                0.0
            },
            failure_reasons,
        }
    }
}

/// Quality score result for a single document
#[derive(Debug, Clone)]
pub struct QualityScore {
    pub passes: bool,
    pub length: usize,
    pub word_count: usize,
    pub unique_word_ratio: f64,
    pub repetition_ratio: f64,
    pub url_ratio: f64,
    pub special_char_ratio: f64,
    pub avg_word_length: f64,
    pub has_html: bool,
    pub has_profanity: bool,
    pub reasons: Vec<String>,
}

/// Statistics from batch quality scoring
#[derive(Debug)]
pub struct QualityStats {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub pass_rate: f64,
    pub failure_reasons: HashMap<String, usize>,
}

impl QualityStats {
    /// Get top failure reasons
    pub fn top_failure_reasons(&self, limit: usize) -> Vec<(String, usize)> {
        let mut reasons: Vec<_> = self.failure_reasons.iter().collect();
        reasons.sort_by(|a, b| b.1.cmp(a.1));
        reasons
            .into_iter()
            .take(limit)
            .map(|(r, c)| (r.clone(), *c))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_scoring_pass() {
        let scorer = QualityScorer::default();

        let good_text = "This is a well-written document with appropriate length and content. \
                        It contains enough words to meet the minimum requirements. \
                        The text is diverse and does not repeat itself excessively. \
                        Overall, this should pass quality checks easily.";

        let score = scorer.score(good_text);
        assert!(score.passes, "Good text should pass: {:?}", score.reasons);
        assert!(score.unique_word_ratio > 0.5);
        assert!(score.repetition_ratio < 0.3);
    }

    #[test]
    fn test_length_rejection() {
        let scorer = QualityScorer::default();

        let short_text = "Too short";
        let score = scorer.score(short_text);
        assert!(!score.passes);
        assert!(score.reasons.iter().any(|r| r.contains("Length")));
    }

    #[test]
    fn test_repetition_detection() {
        let scorer = QualityScorer::default();

        let repetitive_text = "The same words repeat again. \
                              The same words repeat again. \
                              The same words repeat again. \
                              The same words repeat again. \
                              The same words repeat again. \
                              The same words repeat again.";

        let score = scorer.score(repetitive_text);
        assert!(score.repetition_ratio > 0.3);
        assert!(!score.passes);
    }

    #[test]
    fn test_url_detection() {
        let scorer = QualityScorer::default();

        let url_heavy_text = "Check out https://example.com and www.test.com and also visit \
                             website.org and another.net for more information about stuff and things.";

        let score = scorer.score(url_heavy_text);
        assert!(score.url_ratio > 0.0);
        println!("URL ratio: {:.2}", score.url_ratio);
    }

    #[test]
    fn test_html_detection() {
        let scorer = QualityScorer::default();

        let html_text = "<html><body><p>This is HTML content</p></body></html>";
        let score = scorer.score(html_text);
        assert!(score.has_html);
        assert!(!score.passes);
    }

    #[test]
    fn test_special_char_ratio() {
        let scorer = QualityScorer::default();

        let special_text = r#"!@#$%^&*()_+{}|:"<>?[]\;',./!@#$%^&*()_+{}|:"<>?[]\;',./"#;
        let score = scorer.score(special_text);
        assert!(score.special_char_ratio > 0.5);
    }

    #[test]
    fn test_unique_words_ratio() {
        let scorer = QualityScorer::default();

        let low_diversity_text = "word word word word word word word word word word \
                                 word word word word word word word word word word";
        let score = scorer.score(low_diversity_text);
        assert!(score.unique_word_ratio < 0.1);
        assert!(!score.passes);
    }

    #[test]
    fn test_profanity_filter() {
        let config = QualityConfig {
            filter_profanity: true,
            ..Default::default()
        };
        let scorer = QualityScorer::new(config);

        // Test with clean text
        let clean_text = "This is a perfectly clean and appropriate text with sufficient length \
                         to pass the minimum requirements and no inappropriate content at all.";
        let score = scorer.score(clean_text);
        assert!(!score.has_profanity);
    }

    #[test]
    fn test_avg_word_length() {
        let scorer = QualityScorer::default();

        let short_words = "a b c d e f g h i j k l m n o p q r s t u v w x y z";
        let score = scorer.score(short_words);
        assert!(score.avg_word_length < 2.0);
    }

    #[test]
    fn test_strict_config() {
        let scorer = QualityScorer::new(QualityConfig::strict());

        let marginal_text = "This text is okay but might not meet strict requirements. \
                            It has some content but not super diverse vocabulary.";

        let score = scorer.score(marginal_text);
        // Strict config should be more demanding
        println!("Strict score: {:?}", score);
    }

    #[test]
    fn test_batch_scoring() {
        let scorer = QualityScorer::default();

        let texts = vec![
            "This is a good quality text with sufficient length and diversity of words and content.",
            "Short",
            "Repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat",
            "Another good quality text that should pass all the quality checks without problems.",
        ];

        let stats = scorer.batch_stats(&texts);
        assert_eq!(stats.total, 4);
        assert!(stats.passed >= 1);
        assert!(stats.failed >= 1);
        println!("Pass rate: {:.2}%", stats.pass_rate * 100.0);
        println!("Top failures: {:?}", stats.top_failure_reasons(3));
    }

    #[test]
    fn test_word_tokenization() {
        let text = "Hello, world! This is a test... with punctuation.";
        let words = QualityScorer::tokenize_words(text);
        assert_eq!(words, vec!["hello", "world", "this", "is", "a", "test", "with", "punctuation"]);
    }

    #[test]
    fn test_code_text_quality() {
        let scorer = QualityScorer::new(QualityConfig::lenient());

        let code_text = r#"
        function example() {
            const x = 42;
            return x * 2;
        }
        "#;

        let score = scorer.score(code_text);
        // Code might have special chars but lenient config should handle it
        println!("Code quality score: {:?}", score);
    }
}
