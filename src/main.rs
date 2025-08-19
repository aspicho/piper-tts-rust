mod phoneme_gen;
use phoneme_gen::PhonemeGen;
mod model_handler;
use model_handler::Model;



fn main() -> ort::Result<()> {    
    let decoder_path = "models/g2p/decoder_model_mini_bart_g2p.onnx".to_string();
    let encoder_path = "models/g2p/encoder_model_mini_bart_g2p.onnx".to_string();
    let tokenizer_path = "models/g2p/tokenizer.json".to_string();
    let vocab_path = "models/g2p/vocab.json".to_string();
    let arpabet_mapping_path = "arpabet-mapping.txt".to_string();

    let start = std::time::Instant::now();
    let mut phoneme_gen = PhonemeGen::new(
        decoder_path, encoder_path, 
        tokenizer_path, vocab_path, 
        arpabet_mapping_path
    );
    phoneme_gen.load()?;
    println!("PhonemeGen loaded in: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let mut model = Model::new(
    "models/en_US-norman-medium.onnx",
    "models/en_US-norman-medium.onnx.json",
    ).expect("Failed to create model");

    println!("Model loaded in: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let input_text = "harrison had blonde hair as a baby and now has brown hair. Are we sure Harrison isn't fooling Dexter? Couldn't he turn out to be the New York Ripper all along?".to_string().to_lowercase();

    let formated_ipa_string = phoneme_gen.process_text(&input_text)
        .expect("Failed to process text");

    println!("Phoneme generation completed in: {:?}", start.elapsed());
    println!("Formatted IPA string: {}", formated_ipa_string);

    let sample_rate = model.config.audio.sample_rate;

    let start = std::time::Instant::now();
    let (waveform_tensor_shape, waveform_tensor) = model.process_ipa_string(&formated_ipa_string)
        .expect("Failed to process IPA string"); 
    println!("Inference completed in: {:?}", start.elapsed());
    println!("Waveform tensor shape: {:?}", waveform_tensor_shape);

    model.write_wav_file(
        &waveform_tensor, sample_rate, "output.wav"
    ).expect("Failed to write WAV file");

    Ok(())
}
