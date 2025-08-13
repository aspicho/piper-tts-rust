use std::{collections::HashMap, fs, io::Write};
use ndarray::{Array2, Array3};
use ort::{
    session::{builder::GraphOptimizationLevel, Session}
};

mod phoneme_gen;
use phoneme_gen::PhonemeGen;
mod model_handler;
use model_handler::Model;
use tokenizers::Tokenizer;



fn main() -> ort::Result<()> {    
    fn load_vocab_id2token(path: &str) -> Result<HashMap<usize, String>, Box<dyn std::error::Error + Send + Sync>> {
        let s = fs::read_to_string(path)?;
        let token2id: HashMap<String, usize> = serde_json::from_str(&s)?;
        let mut id2token = HashMap::with_capacity(token2id.len());
        for (tok, id) in token2id {
            id2token.insert(id, tok);
        }
        Ok(id2token)
    }

    fn find_first_token_id(token_map: &HashMap<String, usize>, candidates: &[&str]) -> Option<usize> {
        for &cand in candidates {
            if let Some(&id) = token_map.get(cand) {
                return Some(id);
            }
        }
        None
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

    fn greedy_decode(
        decoder_session: &mut ort::session::Session, 
        encoder_output: &Array3<f32>,
        encoder_attention_mask: &Array2<i64>,
        vocab_json_path: &str,
        _tokenizer: &Tokenizer,
        max_len: usize,
    ) -> Result<(Vec<usize>, Vec<String>), Box<dyn std::error::Error + Send + Sync>> {

        let id2token = load_vocab_id2token(vocab_json_path)?;

        let bos_id = 2i64;  // </s> is used as BOS for BART decoder
        let eos_id = 2i64;  // </s>
        let pad_id = 1i64;  // <pad>
        let s_id = 0i64;    // <s>

        let mut decoder_ids: Vec<i64> = vec![bos_id];
        let mut decoded_ids: Vec<usize> = Vec::new();
        let mut decoded_tokens: Vec<String> = Vec::new();

        for step in 0..max_len {
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
            let outputs = decoder_session.run(inputs)?;
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

            let next_id_usize = argmax(last_logits_slice);
            let next_id = next_id_usize as i64;
            decoder_ids.push(next_id);

            let tok_str = id2token.get(&next_id_usize)
                .cloned()
                .unwrap_or_else(|| format!("<{}>", next_id_usize));

            println!("[decode][step {}] next_id={} token='{}'", step, next_id, tok_str);

            if next_id == eos_id {
                println!("[decode] Reached EOS; stopping.");
                break;
            }

            if next_id != bos_id && next_id != pad_id && next_id != eos_id && next_id != s_id {
                decoded_ids.push(next_id_usize);
                decoded_tokens.push(tok_str);
            }
        }

        Ok((decoded_ids, decoded_tokens))
    }

    fn process_word_g2p(
        word: &str,
        encoder_model: &mut Session,
        decoder_model: &mut Session,
        tokenizer: &Tokenizer,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        if word.chars().all(|c| c.is_ascii_punctuation() || c.is_whitespace()) {
            println!("[G2P] Keeping punctuation as-is: '{}'", word);
            return Ok(vec![word.to_string()]);
        }
        
        println!("[G2P] Processing word: '{}'", word);
        
        let encoded = tokenizer.encode(word, true)?;
        let input_ids: Vec<i64> = encoded.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = vec![1; input_ids.len()];

        println!("[G2P] Input IDs for '{}': {:?}", word, input_ids);

        let input_array = Array2::<i64>::from_shape_vec([1, input_ids.len()], input_ids)?;
        let attention_mask_array = Array2::<i64>::from_shape_vec([1, attention_mask.len()], attention_mask)?;
        
        let input_ids_tensor = ort::value::Tensor::from_array(input_array);
        let attention_mask_tensor = ort::value::Tensor::from_array(attention_mask_array.clone());

        let encoder_outputs = encoder_model.run(vec![
            ("input_ids", input_ids_tensor?),
            ("attention_mask", attention_mask_tensor?),
        ])?;

        let (encoder_output_shape, encoder_output_tensor) = encoder_outputs.get("last_hidden_state")
            .expect("Failed to get encoder output")
            .try_extract_tensor::<f32>()?;

        let encoder_output_array = Array3::<f32>::from_shape_vec(
            [1, encoder_output_shape[1] as usize, encoder_output_shape[2] as usize],
            encoder_output_tensor.to_vec()
        )?;

        let (_, tokens) = greedy_decode(
            decoder_model,
            &encoder_output_array,
            &attention_mask_array,
            "models/g2p/vocab.json",
            tokenizer,
            50,
        )?;

        println!("[G2P] Word '{}' -> phonemes: {:?}", word, tokens);
        Ok(tokens)
    }

    let encoder_model_path = "models/g2p/encoder_model_mini_bart_g2p.onnx";
    let mut encoder_model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(encoder_model_path)?;

    println!("ENCODER Model inputs: {:?}", encoder_model.inputs);
    println!("ENCODER Model outputs: {:?}", encoder_model.outputs);

    let decoder_model_path = "models/g2p/decoder_model_mini_bart_g2p.onnx";
    let mut decoder_model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(decoder_model_path)?;
    println!("DECODER Model inputs: {:?}", decoder_model.inputs);
    println!("DECODER Model outputs: {:?}", decoder_model.outputs);

    let input_text = "New incoming message from: dragooonat. Message: Is it still lagging TY bro".to_string().to_lowercase();
    
    let tokenizer = Tokenizer::from_file("models/g2p/tokenizer.json")
        .expect("Failed to load tokenizer");

    let dictionary_path = "cmudict-0.7b.txt";
    let arpabet_mapping_path = "arpabet-mapping.txt";
    let phoneme_gen = PhonemeGen::new(dictionary_path, arpabet_mapping_path);

    let tokens: Vec<String> = input_text
        .split_whitespace()
        .flat_map(|word| {
            let mut result = Vec::new();
            let mut current_word = String::new();
            
            for ch in word.chars() {
                if ch.is_ascii_punctuation() {
                    if !current_word.is_empty() {
                        result.push(current_word.clone());
                        current_word.clear();
                    }
                    result.push(ch.to_string());
                } else {
                    current_word.push(ch);
                }
            }
            
            if !current_word.is_empty() {
                result.push(current_word);
            }
            
            result
        })
        .collect();
    
    println!("Processing {} tokens: {:?}", tokens.len(), tokens);

    let mut all_phonemes: Vec<String> = Vec::new();

    for token in tokens {
        let token_phonemes = process_word_g2p(
            &token,
            &mut encoder_model,
            &mut decoder_model,
            &tokenizer,
        )?;
        all_phonemes.extend(token_phonemes);
    }

    println!("All phonemes: {:?}", all_phonemes);

    let mut model = Model::new(
        "models/en_GB-northern_english_male-medium.onnx",
        "models/en_GB-northern_english_male-medium.onnx.json",

        // "models/en_US-joe-medium.onnx",
        // "models/en_US-joe-medium.onnx.json",
    ).expect("Failed to create model handler");

    let concatenated_tokens = all_phonemes.join(" ");
    println!("Concatenated tokens: {}", concatenated_tokens);
    let ipa_string = phoneme_gen.arpabet_to_ipa(&[vec![concatenated_tokens]]);
    println!("IPA string: {:?}", ipa_string);
    let formated_ipa_string = phoneme_gen.format_ipa_string(&ipa_string, "_", "^");
    println!("Formatted IPA string: {}", formated_ipa_string);

    // // let example_text = "^_f_a_ɪ_v_,_ _p_ɪ_ŋ_k_ _ˈ_p_a_ɪ_n_a_p_ə_l_z_ _ɒ_n_ _ð_ə_ _b_a_ˈ_n_ɑ_ː_n_a_ _t_r_i_ː_!_".to_string().to_lowercase();
    // let example_text = "Now plaing: формалин. From Monetochka".to_string();
    // let formated_ipa_string = phoneme_gen.text_to_ipa(&example_text);

    // println!("Formatted IPA string: {}", formated_ipa_string);

    let phonemes_ids = model.ipa_string_to_phoneme_ids(&formated_ipa_string)
        .expect("Failed to convert IPA string to phoneme IDs");

    let sample_rate = model.config.audio.sample_rate;

    let start = std::time::Instant::now();
    let outputs = model.run_inference(phonemes_ids)
        .expect("Failed to run model inference");
    println!("Model inference took: {:?}", start.elapsed());

    println!("Model outputs: {:?}", outputs);

    let (waveform_tensor_shape, waveform_tensor) = outputs["output"].try_extract_tensor::<f32>()?;    
    println!("Waveform tensor shape: {:?}", waveform_tensor_shape);

    let output_path = "output.wav";
    let start = std::time::Instant::now();
    let mut file = std::fs::File::create(output_path)
        .expect("Failed to create output file");

    let header: Vec<u8> = vec![
        "RIFF".as_bytes().to_vec(),
        (36 + waveform_tensor.len() as u32 * 2).to_le_bytes().to_vec(),
        "WAVE".as_bytes().to_vec(),
        "fmt ".as_bytes().to_vec(),
        (16u32).to_le_bytes().to_vec(), // Subchunk1Size
        (1u16).to_le_bytes().to_vec(), // AudioFormat (PCM)
        (1u16).to_le_bytes().to_vec(), // NumChannels (Mono)
        (sample_rate.clone() as u32).to_le_bytes().to_vec(), // SampleRate
        (sample_rate.clone() as u32 * 2).to_le_bytes().to_vec(), // ByteRate
        (2u16).to_le_bytes().to_vec(), // BlockAlign
        (16u16).to_le_bytes().to_vec(), // BitsPerSample
        "data".as_bytes().to_vec(),
        (waveform_tensor.len() as u32 * 2).to_le_bytes().to_vec(), // Subchunk2Size
    ].concat();
    file.write_all(&header)
        .expect("Failed to write WAV header");

    let samples: Vec<i16> = waveform_tensor.iter()
        .map(|&sample| (sample * i16::MAX as f32) as i16)
        .collect();
    file.write_all(&samples.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>())
        .expect("Failed to write WAV samples");

    println!("WAV file created successfully at: {}", output_path);
    println!("WAV file creation took: {:?}", start.elapsed());

    Ok(())
}
