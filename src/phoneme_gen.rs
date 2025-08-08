use std::collections::HashMap;

pub struct PhonemeGen {
    dictionary: HashMap<String, String>,
    arpabet_to_ipa: HashMap<String, String>,
    dictionary_path: String,
    arpabet_mapping_path: String,
}

impl PhonemeGen {
    pub fn new(dictionary_path: &str, arpabet_mapping_path: &str) -> Self {
        let dictionary = Self::load_dictionary(dictionary_path);
        let arpabet_to_ipa = Self::load_arpabet_mapping(arpabet_mapping_path);
        
        PhonemeGen {
            dictionary,
            arpabet_to_ipa,
            dictionary_path: dictionary_path.to_string(),
            arpabet_mapping_path: arpabet_mapping_path.to_string(),
        }
    }

    fn load_dictionary(path: &str) -> HashMap<String, String> {
        let bytes = std::fs::read(path)
            .expect("Failed to read the dictionary file");
        let dictionary_data = String::from_utf8_lossy(&bytes).to_string();

        let mut phoneme_id_map: HashMap<String, String> = HashMap::new();

        for line in dictionary_data.lines() {
            if line.starts_with(";;;") || line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }
            let word = parts[0].to_string();
            let phonemes = parts[1..].join(" ");
            phoneme_id_map.insert(word, phonemes);
        }
        phoneme_id_map
    }

    fn load_arpabet_mapping(path: &str) -> HashMap<String, String> {
        let bytes = std::fs::read(path)
            .expect("Failed to read the ARPAbet mapping file");
        let mapping_data = String::from_utf8_lossy(&bytes).to_string();

        let mut arpabet_to_ipa: HashMap<String, String> = HashMap::new();

        for line in mapping_data.lines() {
            let parts: Vec<&str> = line.split(", ").collect();
            if parts.len() != 2 {
                continue;
            }
            let arpabet = parts[0].trim().to_string();
            let ipa = parts[1].trim().to_string();
            arpabet_to_ipa.insert(arpabet, ipa);
        }
        arpabet_to_ipa
    }

    pub fn text_to_sentences(&self, text: &str) -> Vec<String> {
        let sentance_endings = vec![".", "!", "?", ";"];
        text.split_inclusive(|c: char| sentance_endings.contains(&c.to_string().as_str()))
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_string())
            .collect()
    }

    pub fn sentences_to_words(&self, sentences: &[String]) -> Vec<Vec<String>> {
        sentences.iter()
            .map(|s| s.split_whitespace().map(|w| w.to_string()).collect())
            .collect()
    }

    fn clear_word(word: &str, symbols: Option<&[(char, &str)]>) -> String {
        let symbols_to_replace = symbols.unwrap_or(&[
            ('^', ""),
            ('_', ""),
            (',', ""),
            (';', ""), 
            ('(', ""), 
            (')', ""), 
            ]);
            
        let mut cleaned_word = word.to_string();
        for (symbol, replacement) in symbols_to_replace {
            cleaned_word = cleaned_word.replace(*symbol, replacement);
        }
        cleaned_word
    }

    pub fn words_to_arpabet(&self, words: &[Vec<String>], sentance_endings: Option<&[&str]>, symbols_to_replace: Option<&[(char, &str)]>) -> Vec<Vec<String>> {
        let sentance_endings_ = sentance_endings.unwrap_or(&[".", "!", "?", ";"]);

        words.iter()
            .map(|word_list| {
                word_list.iter()
                    .map(|word| {
                        let cleaned_word = Self::clear_word(word, symbols_to_replace);
                        if cleaned_word.is_empty() {
                            return String::new();
                        }

                        let cleaned_word = cleaned_word.chars()
                            .filter(|c| !sentance_endings_.contains(&c.to_string().as_str()))
                            .collect::<String>();

                        self.dictionary.get(&cleaned_word.to_uppercase())
                            .cloned()
                            .unwrap_or_else(|| cleaned_word)
                    })
                    .collect()
            })
            .collect()
    }

    pub fn arpabet_to_ipa(&self, arpabet_words: &[Vec<String>]) -> Vec<Vec<String>> {
        arpabet_words.iter()
            .map(|word_list| {
                word_list.iter()
                    .map(|arpabet_word| {
                        if arpabet_word.is_empty() {
                            return String::new();
                        }
                        
                        let converted: Vec<String> = arpabet_word.split_whitespace()
                            .map(|arpabet| {
                                self.arpabet_to_ipa.get(arpabet)
                                    .map(|ipa| ipa.to_string())
                                    .unwrap_or_else(|| arpabet.to_string())
                            })
                            .collect();
                        
                        if converted.is_empty() {
                            arpabet_word.clone()
                        } else {
                            converted.join("")
                        }
                    })
                    .collect()
            })
            .collect()
    }

    pub fn format_ipa_string(&self, ipa_words: &[Vec<String>], char_separator: &str, string_start: &str) -> String {
        let ipa_string = ipa_words.iter()
            .map(|word_list| word_list.join(" ") + ".")
            .collect::<Vec<String>>()
            .join(" ");

        let formated_ipa_string = string_start.to_string() + &ipa_string.chars()
            .map(|c| c.to_string())
            .collect::<Vec<String>>()
            .join(char_separator);

        formated_ipa_string
    }

    pub fn text_to_ipa(&self, text: &str) -> String {
        let sentences = self.text_to_sentences(text);
        let words = self.sentences_to_words(&sentences);
        let arpabet_words = self.words_to_arpabet(&words, None, None);
        let ipa_words = self.arpabet_to_ipa(&arpabet_words);
        self.format_ipa_string(&ipa_words, "_", "^")
    }
}