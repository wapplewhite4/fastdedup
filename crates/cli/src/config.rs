//! Configuration file support for deduplication pipelines

use anyhow::{Context, Result};
use fastdedup_filters::language::LanguageFilterConfig;
use fastdedup_filters::quality::QualityConfig;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Complete pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub input: InputConfig,
    pub output: OutputConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deduplication: Option<DedupConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<FiltersConfig>,
}

impl PipelineConfig {
    /// Load configuration from a file (YAML or TOML)
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");

        match extension {
            "yaml" | "yml" => {
                serde_yaml::from_str(&content)
                    .with_context(|| format!("Failed to parse YAML config: {}", path.display()))
            }
            "toml" => toml::from_str(&content)
                .with_context(|| format!("Failed to parse TOML config: {}", path.display())),
            _ => Err(anyhow::anyhow!(
                "Unsupported config file format: {}. Use .yaml, .yml, or .toml",
                extension
            )),
        }
    }

    /// Save configuration to a file
    #[allow(dead_code)]
    pub fn save(&self, path: &Path) -> Result<()> {
        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");

        let content = match extension {
            "yaml" | "yml" => serde_yaml::to_string(self)?,
            "toml" => toml::to_string_pretty(self)?,
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported config file format: {}. Use .yaml, .yml, or .toml",
                    extension
                ))
            }
        };

        std::fs::write(path, content)
            .with_context(|| format!("Failed to write config file: {}", path.display()))?;

        Ok(())
    }

    /// Create a default English-only pipeline config
    #[allow(dead_code)]
    pub fn default_english() -> Self {
        Self {
            input: InputConfig {
                path: "input.jsonl".to_string(),
                format: FormatType::Jsonl,
            },
            output: OutputConfig {
                path: "output.jsonl".to_string(),
                format: FormatType::Jsonl,
                compression: CompressionType::None,
            },
            deduplication: Some(DedupConfig {
                exact: Some(ExactDedupConfig {
                    field: Some("text".to_string()),
                    normalize: true,
                }),
                fuzzy: Some(FuzzyDedupConfig {
                    threshold: 0.85,
                    field: "text".to_string(),
                }),
            }),
            filters: Some(FiltersConfig {
                language: LanguageFilterConfig::english_only(),
                quality: QualityConfig::default(),
            }),
        }
    }

    /// Create a multilingual pipeline config
    #[allow(dead_code)]
    pub fn default_multilingual() -> Self {
        Self {
            input: InputConfig {
                path: "input.jsonl".to_string(),
                format: FormatType::Jsonl,
            },
            output: OutputConfig {
                path: "output.jsonl".to_string(),
                format: FormatType::Jsonl,
                compression: CompressionType::None,
            },
            deduplication: Some(DedupConfig {
                exact: Some(ExactDedupConfig {
                    field: Some("text".to_string()),
                    normalize: true,
                }),
                fuzzy: Some(FuzzyDedupConfig {
                    threshold: 0.8,
                    field: "text".to_string(),
                }),
            }),
            filters: Some(FiltersConfig {
                language: LanguageFilterConfig::multilingual(),
                quality: QualityConfig::lenient(),
            }),
        }
    }
}

/// Input configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputConfig {
    pub path: String,
    pub format: FormatType,
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub path: String,
    pub format: FormatType,
    #[serde(default)]
    pub compression: CompressionType,
}

/// File format type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FormatType {
    Jsonl,
    Json,
    Parquet,
    Csv,
}

/// Compression type
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CompressionType {
    #[default]
    None,
    Gzip,
    Zstd,
}

/// Deduplication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exact: Option<ExactDedupConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fuzzy: Option<FuzzyDedupConfig>,
}

/// Exact deduplication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExactDedupConfig {
    /// Field to use for deduplication (None = full content)
    pub field: Option<String>,
    /// Whether to normalize text before hashing
    #[serde(default)]
    pub normalize: bool,
}

/// Fuzzy deduplication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyDedupConfig {
    /// Similarity threshold (0.0-1.0)
    pub threshold: f64,
    /// Text field to compare
    pub field: String,
}

/// Filter configuration combining language and quality filters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiltersConfig {
    pub language: LanguageFilterConfig,
    pub quality: QualityConfig,
}

/// Load filter config from file
pub fn load_filter_config(path: &Path) -> Result<FiltersConfig> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read filter config: {}", path.display()))?;

    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");

    match extension {
        "yaml" | "yml" => serde_yaml::from_str(&content)
            .with_context(|| format!("Failed to parse YAML filter config: {}", path.display())),
        "toml" => toml::from_str(&content)
            .with_context(|| format!("Failed to parse TOML filter config: {}", path.display())),
        _ => Err(anyhow::anyhow!(
            "Unsupported config file format: {}. Use .yaml, .yml, or .toml",
            extension
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_english_config() {
        let config = PipelineConfig::default_english();
        assert!(config.deduplication.is_some());
        assert!(config.filters.is_some());
    }

    #[test]
    fn test_save_and_load_yaml() {
        let config = PipelineConfig::default_english();

        let mut temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().with_extension("yaml");

        config.save(&path).unwrap();
        let loaded = PipelineConfig::load(&path).unwrap();

        assert_eq!(config.input.path, loaded.input.path);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_save_and_load_toml() {
        let config = PipelineConfig::default_multilingual();

        let mut temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().with_extension("toml");

        config.save(&path).unwrap();
        let loaded = PipelineConfig::load(&path).unwrap();

        assert_eq!(config.output.path, loaded.output.path);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_unsupported_format() {
        let config = PipelineConfig::default_english();
        let mut temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().with_extension("json");

        let result = config.save(&path);
        assert!(result.is_err());
    }
}
