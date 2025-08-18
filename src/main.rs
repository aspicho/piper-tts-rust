use std::{collections::HashMap, fs, io::Write, os::unix::process};
use ndarray::{Array2, Array3};
use ort::{
    session::{builder::GraphOptimizationLevel, Session}
};

mod phoneme_gen;
use phoneme_gen::PhonemeGen;
mod model_handler;
use model_handler::Model;
use tokenizers::{pre_tokenizers::punctuation, Tokenizer};



fn main() -> ort::Result<()> {    
    let decoder_path = "models/g2p/decoder_model_mini_bart_g2p.onnx".to_string();
    let encoder_path = "models/g2p/encoder_model_mini_bart_g2p.onnx".to_string();
    let tokenizer_path = "models/g2p/tokenizer.json".to_string();
    let vocab_path = "models/g2p/vocab.json".to_string();
    let arpabet_mapping_path = "arpabet-mapping.txt".to_string();

    let mut phoneme_gen = PhonemeGen::new(decoder_path, encoder_path, tokenizer_path, vocab_path, arpabet_mapping_path);
    phoneme_gen.load()?;


    let input_text = "Hello, Dexter Morgan. I'm gay! How are you?".to_string().to_lowercase();
    
    let sentence_endings = vec![".", "!", "?"];
    let mut sentences: Vec<String> = Vec::new();

    let mut current_sentence = String::new();
    for word in input_text.split_whitespace() {
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

    println!("Sentences: {:?}", sentences);

    let mut processed_sentences: Vec<String> = Vec::new();
    for sentence in sentences {
        let bos = "^";
        let eos = "$";
        let pad = "_";

        let mut processed_sentence: String = String::new();

        processed_sentence.push_str(bos);
        for word in sentence.split_whitespace() {
            println!("Processing word: {}", word);
            
            let punctuation = word.chars().last().map(|c| if c.is_ascii_punctuation() { c } else { ' ' }).unwrap_or(' ');
            let word_without_punctuation = word.trim_end_matches(punctuation);

            let token_phonemes = phoneme_gen.process_word(word_without_punctuation)?;
            println!("Phonemes for word '{}': {:?}", word_without_punctuation, token_phonemes);

            if !token_phonemes.is_empty() {
                processed_sentence.push_str(&token_phonemes.join(""));
            }            
            if punctuation != ' ' {
                processed_sentence.push(punctuation);
            }
            processed_sentence.push(' ');
        }
        processed_sentence = processed_sentence.trim().to_string();

        println!("Pre-processed sentence: {}", processed_sentence);

        processed_sentence = processed_sentence.chars().map(|c| c.to_string()).collect::<Vec<String>>().join(pad);

        processed_sentence.push_str(pad);
        processed_sentence.push_str(eos);

        if !processed_sentence.is_empty() {
            processed_sentences.push(processed_sentence);
        }
    }
    println!("Processed sentences: {:?}", processed_sentences);


    // let words = input_text.split_inclusive(" ").map(|s| s.to_string()).collect::<Vec<String>>();
    // println!("Processing {} words: {:?}", words.len(), words);

    // let mut all_phonemes: Vec<String> = Vec::new();
    // for word in words {
    //     let start = std::time::Instant::now();
    //     let token_phonemes = phoneme_gen.process_word(&word)?;
    //     println!("Processing word '{}' took: {:?}", word, start.elapsed());
    //     all_phonemes.extend(token_phonemes);
    // }

    // println!("All phonemes: {:?}", all_phonemes);

    // let concatenated_tokens = all_phonemes.join("_");
    // println!("Concatenated tokens: {}", concatenated_tokens);

    // let formated_ipa_string = format!("^_{}_", concatenated_tokens);

    // println!("Formatted IPA string: {}", formated_ipa_string);


    let formated_ipa_string = processed_sentences.join("");

    println!("Formatted IPA string: {}", formated_ipa_string);

    let mut model = Model::new(
        "models/en_US-norman-medium.onnx",
        "models/en_US-norman-medium.onnx.json",
        ).expect("Failed to create model");

    let phonemes_ids = model.ipa_string_to_phoneme_ids(&formated_ipa_string)
        .expect("Failed to convert IPA string to phoneme IDs");

    println!("Phoneme IDs: {:?}", phonemes_ids);

    let sample_rate = model.config.audio.sample_rate;

    let start = std::time::Instant::now();
    let outputs = model.run_inference(phonemes_ids)
        .expect("Failed to run model inference");
    println!("Model inference took: {:?}", start.elapsed());

    // println!("Model outputs: {:?}", outputs);

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
