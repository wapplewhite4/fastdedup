use dataset_dedup_filters::language::{LanguageFilter, LanguageFilterConfig};
use dataset_dedup_filters::quality::{QualityScorer, QualityConfig};

fn main() {
    println!("🎯 Quality Filters Demo\n");
    println!("======================\n");

    // Demo 1: Language Detection
    demo_language_detection();
    println!("\n---\n");

    // Demo 2: Quality Scoring
    demo_quality_scoring();
    println!("\n---\n");

    // Demo 3: Combined Filtering
    demo_combined_filtering();
}

fn demo_language_detection() {
    println!("📍 Demo 1: Language Detection");
    println!("==============================\n");

    let filter = LanguageFilter::new(LanguageFilterConfig::multilingual())
        .expect("Failed to create language filter");

    let test_texts = vec![
        ("English", "This is a sample text in English that should be detected correctly."),
        ("Spanish", "Este es un texto de prueba en español que debería ser detectado correctamente."),
        ("French", "Ceci est un texte d'exemple en français qui devrait être détecté correctement."),
        ("German", "Dies ist ein Beispieltext auf Deutsch, der korrekt erkannt werden sollte."),
        ("Code", "function hello() { const x = 42; return x * 2; } // JavaScript code"),
        ("Too Short", "Hi"),
    ];

    for (label, text) in test_texts {
        let accepted = filter.is_accepted(text);
        let detection = filter.detect_with_confidence(text);

        print!("{:12} | ", label);
        if let Some((lang, conf)) = detection {
            println!("Lang: {:?}, Conf: {:.2}, Accepted: {}", lang, conf, accepted);
        } else {
            println!("No detection (too short), Accepted: {}", accepted);
        }
    }
}

fn demo_quality_scoring() {
    println!("📊 Demo 2: Quality Scoring");
    println!("===========================\n");

    let scorer = QualityScorer::default();

    let test_texts = vec![
        (
            "Good Quality",
            "This is a well-written document with appropriate length and diverse vocabulary. \
             It contains enough unique words to meet quality standards. The content is meaningful \
             and free from excessive repetition or spam-like patterns.",
        ),
        (
            "Too Short",
            "Short text",
        ),
        (
            "Repetitive",
            "Same words again and again. Same words again and again. Same words again and again. \
             Same words again and again. Same words again and again. Same words again and again.",
        ),
        (
            "Low Diversity",
            "word word word word word word word word word word word word word word word word \
             word word word word word word word word word word word word word word",
        ),
        (
            "Has HTML",
            "<html><body><p>This is HTML content with tags that should be rejected by default.</p></body></html>",
        ),
        (
            "URL Heavy",
            "Check out https://example.com and www.test.com and visit website.org for more at another.net",
        ),
    ];

    for (label, text) in test_texts {
        let score = scorer.score(text);

        println!("{}:", label);
        println!("  Passes: {}", score.passes);
        println!("  Length: {}, Words: {}", score.length, score.word_count);
        println!("  Unique ratio: {:.2}, Repetition: {:.2}",
                 score.unique_word_ratio, score.repetition_ratio);

        if !score.passes {
            println!("  Reasons: {}", score.reasons.join(", "));
        }
        println!();
    }
}

fn demo_combined_filtering() {
    println!("🔍 Demo 3: Combined Language + Quality Filtering");
    println!("=================================================\n");

    let lang_filter = LanguageFilter::new(LanguageFilterConfig::english_only())
        .expect("Failed to create language filter");
    let quality_scorer = QualityScorer::new(QualityConfig::strict());

    let test_texts = vec![
        "This is a high-quality English document with sufficient length, diverse vocabulary, \
         and meaningful content that should pass both language and quality filters easily.",

        "Este es un documento de alta calidad en español con suficiente longitud y vocabulario diverso.",

        "ok yes no maybe test sample data info text words here stuff things content material",

        "word word word word word word word word word word word word word word word word word word",
    ];

    println!("Testing {} documents:\n", test_texts.len());

    let mut passed = 0;
    for (i, text) in test_texts.iter().enumerate() {
        let lang_ok = lang_filter.is_accepted(text);
        let quality_score = quality_scorer.score(text);
        let both_ok = lang_ok && quality_score.passes;

        println!("Document {}:", i + 1);
        println!("  Language: {}", if lang_ok { "✓" } else { "✗" });
        println!("  Quality:  {}", if quality_score.passes { "✓" } else { "✗" });
        println!("  Overall:  {}", if both_ok { "✓ PASS" } else { "✗ REJECT" });

        if !both_ok {
            if !lang_ok {
                if let Some((lang, conf)) = lang_filter.detect_with_confidence(text) {
                    println!("    → Wrong language: {:?} (conf: {:.2})", lang, conf);
                }
            }
            if !quality_score.passes {
                println!("    → Quality issues: {}", quality_score.reasons.join(", "));
            }
        }

        println!();

        if both_ok {
            passed += 1;
        }
    }

    println!("Summary: {}/{} documents passed all filters ({:.1}%)",
             passed, test_texts.len(),
             (passed as f64 / test_texts.len() as f64) * 100.0);
}
