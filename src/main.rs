use std::{collections::HashMap, io::Write};
use serde::Deserialize;
use ndarray::{Array, Array1, Array2, Array4};
use ort::{
    session::{builder::GraphOptimizationLevel, output, Session},
    Error,
};

mod phoneme_gen;
use phoneme_gen::PhonemeGen;
mod model_handler;
use model_handler::Model;


fn main() -> ort::Result<()> {    
    let mut model = Model::new(
        "models/en_US-norman-medium.onnx",
        "models/en_US-norman-medium.onnx.json"
    ).expect("Failed to create model handler");

    let dictionary_path = "cmudict-0.7b.txt";
    let arpabet_mapping_path = "arpabet-mapping.txt";
    let phoneme_gen = PhonemeGen::new(dictionary_path, arpabet_mapping_path);

    // let example_text = "^_f_a_ɪ_v_,_ _p_ɪ_ŋ_k_ _ˈ_p_a_ɪ_n_a_p_ə_l_z_ _ɒ_n_ _ð_ə_ _b_a_ˈ_n_ɑ_ː_n_a_ _t_r_i_ː_!_".to_string().to_lowercase();
    let example_text = "Welcome aboard captain! All systems online.".to_string();
    let formated_ipa_string = phoneme_gen.text_to_ipa(&example_text);

    println!("Formatted IPA string: {}", formated_ipa_string);

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
