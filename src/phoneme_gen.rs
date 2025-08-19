use std::collections::HashMap;

use ndarray::{Array2, Array3};
use ort::{
    session::{builder::GraphOptimizationLevel, Session}
};

pub struct PhonemeGen {
    decoder_path: String,
    encoder_path: String,
    tokenizer_path: String,
    vocab_path: String,
    arpabet_mapping_path: String,

    encoder: Option<Session>,
    decoder: Option<Session>,
    tokenizer: Option<tokenizers::Tokenizer>,
    arpabet_mapping: Option<HashMap<String, String>>,
    pub vocab: Option<(HashMap<String, usize>, HashMap<usize, String>)>,
}

impl PhonemeGen {
    pub fn new(
        decoder_path: String,
        encoder_path: String,
        tokenizer_path: String,
        vocab_path: String,
        arpabet_mapping_path: String,
    ) -> Self {
        Self {
            decoder_path,
            encoder_path,
            tokenizer_path,
            arpabet_mapping_path,
            vocab_path,
            encoder: None,
            decoder: None,
            tokenizer: None,
            vocab: None,
            arpabet_mapping: None,
        }
    }

    pub fn load(&mut self) -> ort::Result<()> {
        let encoder_model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(&self.encoder_path)?;

        let decoder_model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(&self.decoder_path)?;

        let tokenizer = tokenizers::Tokenizer::from_file(&self.tokenizer_path)
            .expect("Failed to load tokenizer");

        let vocab = {
            let vocab_data = std::fs::read_to_string(&self.vocab_path)
                .expect("Failed to read vocabulary file");
            let vocab_map: HashMap<String, usize> = serde_json::from_str(&vocab_data)
                .expect("Failed to parse vocabulary JSON");

            let mut reverse_vocab_map: HashMap<usize, String> = HashMap::new();

            for (key, value) in &vocab_map {
                reverse_vocab_map.insert(*value, key.clone());
            }
            
            (vocab_map, reverse_vocab_map)
        };

        let arpabet_mapping = {
            let bytes = std::fs::read(self.arpabet_mapping_path.clone())
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
        };

        self.encoder = Some(encoder_model);
        self.decoder = Some(decoder_model);
        self.tokenizer = Some(tokenizer);
        self.vocab = Some(vocab);
        self.arpabet_mapping = Some(arpabet_mapping);
        Ok(())
    }
    
    fn argmax(slice: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in slice.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        best
    }

    pub fn word_to_tokens(
        &mut self,
        word: &str,
    ) -> Result<(Vec<usize>, Vec<String>), Box<dyn std::error::Error + Send + Sync>> {        
        let encoded = self.tokenizer.as_mut().unwrap().encode(word, true)?;
        let input_ids: Vec<i64> = encoded.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = vec![1; input_ids.len()];

        let input_array = Array2::<i64>::from_shape_vec([1, input_ids.len()], input_ids)?;
        let attention_mask_array = Array2::<i64>::from_shape_vec([1, attention_mask.len()], attention_mask)?;
        
        let input_ids_tensor = ort::value::Tensor::from_array(input_array);
        let attention_mask_tensor = ort::value::Tensor::from_array(attention_mask_array.clone());

        let encoder_output_array = {
            let encoder_outputs = self.encoder.as_mut().unwrap().run(vec![
                ("input_ids", input_ids_tensor?),
                ("attention_mask", attention_mask_tensor?),
            ])?;

            let (encoder_output_shape, encoder_output_tensor) = encoder_outputs.get("last_hidden_state")
                .expect("Failed to get encoder output")
                .try_extract_tensor::<f32>()?;

            Array3::<f32>::from_shape_vec(
                [1, encoder_output_shape[1] as usize, encoder_output_shape[2] as usize],
                encoder_output_tensor.to_vec()
            )?
        };

        let (token_ids, tokens) = self.greedy_decode(
            &encoder_output_array,
            &attention_mask_array,
            50,
        )?;
        Ok((token_ids, tokens))
    }

    fn greedy_decode(
        &mut self,
        encoder_output: &Array3<f32>,
        encoder_attention_mask: &Array2<i64>,
        max_len: usize,
    ) -> Result<(Vec<usize>, Vec<String>), Box<dyn std::error::Error + Send + Sync>> {
        let bos_id = 2i64;  // </s> is used as BOS for BART decoder
        let eos_id = 2i64;  // </s>
        let pad_id = 1i64;  // <pad>
        let s_id = 0i64;    // <s>

        let mut decoder_ids: Vec<i64> = vec![bos_id];
        let mut decoded_ids: Vec<usize> = Vec::new();
        let mut decoded_tokens: Vec<String> = Vec::new();

        for _step in 0..max_len {
            let seq_len = decoder_ids.len();
            let dec_array = Array2::<i64>::from_shape_vec([1, seq_len], decoder_ids.clone())
                .expect("Failed to create decoder Array2");
            let dec_input_value = ort::value::Value::from_array(dec_array)?;
            let encoder_output_value = ort::value::Value::from_array(encoder_output.clone())?;
            let encoder_attention_mask_value = ort::value::Value::from_array(encoder_attention_mask.clone())?;
            let inputs = ort::inputs!{
                "encoder_attention_mask" => encoder_attention_mask_value,
                "input_ids" => dec_input_value,
                "encoder_hidden_states" => encoder_output_value,
            };
            let outputs = self.decoder.as_mut().unwrap().run(inputs)?;
            let (shape, flat_logits) = outputs
                .get("logits")
                .expect("No 'logits' output")
                .try_extract_tensor::<f32>()?;
            if shape.len() != 3 {
                panic!("Unexpected logits shape: {:?}", shape);
            }
            let vocab_size = shape[2] as usize;
            let cur_decoder_seq_len = shape[1] as usize;
            let start = (cur_decoder_seq_len - 1) * vocab_size;
            let end = start + vocab_size;
            let last_logits_slice = &flat_logits[start..end];

            let next_id_usize = PhonemeGen::argmax(last_logits_slice);
            let next_id = next_id_usize as i64;
            decoder_ids.push(next_id);

            let tok_str = self.vocab.as_ref().unwrap().1.get(&next_id_usize)
                .cloned()
                .unwrap_or_else(|| format!("<{}>", next_id_usize));

            if next_id == eos_id {
                break;
            }

            if next_id != bos_id && next_id != pad_id && next_id != eos_id && next_id != s_id {
                decoded_ids.push(next_id_usize);
                decoded_tokens.push(tok_str);
            }
        }

        Ok((decoded_ids, decoded_tokens))
    }

    pub fn arpabet_to_ipa(&self, word: Vec<String>) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(mapping) = &self.arpabet_mapping {
            let mut ipa_phonemes = Vec::new();
            for phoneme in word {
                if let Some(ipa) = mapping.get(&phoneme) {
                    ipa_phonemes.push(ipa.clone());
                } else {
                    ipa_phonemes.push(phoneme); // Fallback to original if no mapping found
                }
            }
            Ok(ipa_phonemes)
        } else {
            Err("ARPAbet mapping not loaded".into())
        }
    }

    pub fn process_word(
        &mut self,
        word: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        if self.encoder.is_none() || self.decoder.is_none() || self.tokenizer.is_none() {
            return Err("Models and tokenizer not loaded".into());
        }
        
        let tokens = self.word_to_tokens(word)?;
        if tokens.0.is_empty() {
            return Err("No tokens generated".into());
        }

        let ipa_phonemes = self.arpabet_to_ipa(tokens.1)?;
        Ok(ipa_phonemes)
    }

    pub fn text_to_sentences(
        &self,
        text: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let sentence_endings = vec![".", "!", "?"];
        let mut sentences: Vec<String> = Vec::new();

        let mut current_sentence = String::new();
        for word in text.split_whitespace() {
            if sentence_endings.iter().any(|&ending| word.ends_with(ending)) {
                current_sentence.push_str(word);
                sentences.push(current_sentence.trim().to_string());
                current_sentence.clear();
            } else {
                current_sentence.push_str(word);
                current_sentence.push(' ');
            }
        }

        if !current_sentence.is_empty() {
            sentences.push(current_sentence.trim().to_string());
        }
        Ok(sentences)
    }

    pub fn process_senteces(
        &mut self,
        sentences: Vec<String>,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let mut processed_sentences: Vec<String> = Vec::new();
        for sentence in sentences {
            let bos = "^";
            let eos = "$";
            let pad = "_";

            let mut processed_sentence: String = String::new();

            processed_sentence.push_str(bos);
            for word in sentence.split_whitespace() {                
                let punctuation = word.chars().last().map(|c| if c.is_ascii_punctuation() { c } else { ' ' }).unwrap_or(' ');
                let word_without_punctuation = word.trim_end_matches(punctuation);

                let token_phonemes = self.process_word(word_without_punctuation)?;

                if !token_phonemes.is_empty() {
                    processed_sentence.push_str(&token_phonemes.join(""));
                }            
                if punctuation != ' ' {
                    processed_sentence.push(punctuation);
                }
                processed_sentence.push(' ');
            }
            processed_sentence = processed_sentence.trim().to_string()
                .chars().map(|c| c.to_string()).collect::<Vec<String>>().join(pad);

            processed_sentence.push_str(pad);
            processed_sentence.push_str(eos);

            if !processed_sentence.is_empty() {
                processed_sentences.push(processed_sentence);
            }
        }

        Ok(processed_sentences)
    }

    pub fn process_text(
        &mut self,
        text: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let sentences = self.text_to_sentences(text)?;
        Ok(self.process_senteces(sentences).unwrap().join(""))
    }
}