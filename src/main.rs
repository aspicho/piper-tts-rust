use std::{collections::HashMap, io::Write};
use serde::Deserialize;
use ndarray::{Array, Array1, Array2, Array4};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    Error,
};

#[derive(Deserialize, Debug)]
struct Audio {
    sample_rate: u64,
    quality: String,
}

#[derive(Deserialize, Debug)]
struct Inference {
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
}

#[derive(Deserialize, Debug)]
struct Language {
    code: String,
    family: String,
    region: String,
    name_native: String,
    name_english: String,
    country_english: String,
}

#[derive(Deserialize, Debug)]
struct Config {
    audio: Audio,
    inference: Inference,
    phoneme_id_map: HashMap<String, Vec<i64>>,
    language: Language,
}

fn main() -> ort::Result<()> {
    let start = std::time::Instant::now();
    // Load the ONNX model
    let mut model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("models/en_US-norman-medium.onnx")?;
    println!("Model loaded successfully");
    println!("Model loading took: {:?}", start.elapsed());

    println!("Model inputs: {:?}", model.inputs);
    println!("Model outputs: {:?}", model.outputs);
    
    let config_path = "models/en_US-norman-medium.onnx.json";
    let config_data = std::fs::read_to_string(config_path)
        .expect("Failed to read the configuration file");
    let config: Config = serde_json::from_str(&config_data)
        .expect("Failed to parse the configuration file");
    

    let dictionary_path = "cmudict-0.7b.txt";
    let bytes = std::fs::read(dictionary_path)
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

    println!("Phoneme ID map loaded with {} entries", phoneme_id_map.len());



    // let example_text = "^_f_a_ɪ_v_,_ _p_ɪ_ŋ_k_ _ˈ_p_a_ɪ_n_a_p_ə_l_z_ _ɒ_n_ _ð_ə_ _b_a_ˈ_n_ɑ_ː_n_a_ _t_r_i_ː_!_".to_string().to_lowercase();
    let example_text = "I love it when you come inside".to_string();

    let sentance_endings = vec![".", "!", "?", ";"];

    let sentences = example_text
        .split_inclusive(|c: char| sentance_endings.contains(&c.to_string().as_str()))
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .collect::<Vec<String>>();

    println!("all sencences: {:?}", sentences);

    let words: Vec<Vec<String>> = sentences.iter()
        .map(|s| s.split_whitespace().map(|w| w.to_string()).collect())
        .collect();

    println!("all words: {:?}", words);

    let symbols_to_replace = vec![
        ('^', ""), // Remove stress marks
        ('_', ""), // Remove underscores
        (',', ""), // Remove commas
        (';', ""), // Remove semicolons
        ('(', ""), // Remove opening parentheses
        (')', ""), // Remove closing parentheses
    ];

    fn clear_word(word: &str, symbols: &[(char, &str)]) -> String {
        let mut cleaned_word = word.to_string();
        for (symbol, replacement) in symbols {
            cleaned_word = cleaned_word.replace(*symbol, replacement);
        }
        cleaned_word
    }

    let arpabet_words: Vec<Vec<String>> = words.iter()
        .map(|word_list| {
            word_list.iter()
                .map(|word| {
                    let cleaned_word = clear_word(word, &symbols_to_replace);
                    if cleaned_word.is_empty() {
                        return String::new();
                    }

                    let cleaned_word = cleaned_word.chars()
                        .filter(|c| !sentance_endings.contains(&c.to_string().as_str()))
                        .collect::<String>();

                    phoneme_id_map.get(&cleaned_word.to_uppercase())
                        .cloned()
                        .unwrap_or_else(|| {
                            cleaned_word
                        })
                })
                .collect()
        })
        .collect();

    println!("ARPAbet words: {:?}", arpabet_words);

    let arpabet_to_ipa: HashMap<&str, &str> = [
        ("AA", "ɑ"), ("AA0", "ɑ"), ("AA1", "ˈɑ"), ("AA2", "ˌɑ"),
        ("AE", "æ"), ("AE0", "æ"), ("AE1", "ˈæ"), ("AE2", "ˌæ"),
        ("AH", "ə"), ("AH0", "ə"), ("AH1", "ˈə"), ("AH2", "ˌə"),
        ("AO", "ɔ"), ("AO0", "ɔ"), ("AO1", "ˈɔ"), ("AO2", "ˌɔ"),
        ("AW", "aʊ"), ("AW0", "aʊ"), ("AW1", "ˈaʊ"), ("AW2", "ˌaʊ"),
        ("AY", "aɪ"), ("AY0", "aɪ"), ("AY1", "ˈaɪ"), ("AY2", "ˌaɪ"),
        ("B", "b"),
        ("CH", "tʃ"),
        ("D", "d"),
        ("DH", "ð"),
        ("EH", "ɛ"), ("EH0", "ɛ"), ("EH1", "ˈɛ"), ("EH2", "ˌɛ"),
        ("ER", "ɚ"), ("ER0", "ɚ"), ("ER1", "ˈɚ"), ("ER2", "ˌɚ"),
        ("EY", "eɪ"), ("EY0", "eɪ"), ("EY1", "ˈeɪ"), ("EY2", "ˌeɪ"),
        ("F", "f"),
        ("G", "ɡ"),
        ("HH", "h"),
        ("IH", "ɪ"), ("IH0", "ɪ"), ("IH1", "ˈɪ"), ("IH2", "ˌɪ"),
        ("IY", "i"), ("IY0", "i"), ("IY1", "ˈi"), ("IY2", "ˌi"),
        ("JH", "dʒ"),
        ("K", "k"),
        ("L", "l"),
        ("M", "m"),
        ("N", "n"),
        ("NG", "ŋ"),
        ("OW", "oʊ"), ("OW0", "oʊ"), ("OW1", "ˈoʊ"), ("OW2", "ˌoʊ"),
        ("OY", "ɔɪ"), ("OY0", "ɔɪ"), ("OY1", "ˈɔɪ"), ("OY2", "ˌɔɪ"),
        ("P", "p"),
        ("R", "r"),
        ("S", "s"),
        ("SH", "ʃ"),
        ("T", "t"),
        ("TH", "θ"),
        ("UH", "ʊ"), ("UH0", "ʊ"), ("UH1", "ˈʊ"), ("UH2", "ˌʊ"),
        ("UW", "u"), ("UW0", "u"), ("UW1", "ˈu"), ("UW2", "ˌu"),
        ("V", "v"),
        ("W", "w"),
        ("Y", "j"),
        ("Z", "z"),
        ("ZH", "ʒ"),
    ].iter().cloned().collect();

    let ipa_words: Vec<Vec<String>> = arpabet_words.iter()
        .map(|word_list| {
            word_list.iter()
                .map(|arpabet_word| {
                    if arpabet_word.is_empty() {
                        return String::new();
                    }
                    
                    let converted: Vec<String> = arpabet_word.split_whitespace()
                        .map(|arpabet| {
                            arpabet_to_ipa.get(arpabet)
                                .map(|&ipa| ipa.to_string())
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
        .collect();

    println!("IPA words: {:?}", ipa_words);
    let ipa_string = ipa_words.iter()
        .map(|word_list| word_list.join(" ") + ".")
        .collect::<Vec<String>>()
        .join(" ");

    let char_separator = "_";
    let string_start = "^";

    // Split IPA string into chars, join with separator, but only one separator between words
    let mut formated_ipa_string = String::from(string_start);
    let mut prev_was_space = false;

    for c in ipa_string.chars() {
        if c == ' ' {
            if !prev_was_space {
                formated_ipa_string.push_str(char_separator);
                prev_was_space = true;
            }
        } else {
            if !formated_ipa_string.ends_with('^') && !prev_was_space {
                formated_ipa_string.push_str(char_separator);
            }
            formated_ipa_string.push(c);
            prev_was_space = false;
        }
    }

    println!("Formatted IPA string: {}", formated_ipa_string);


    // panic!("Not implemented yet!"); // Remove this line when you implement the rest of the code
    
    // let ipa_string = "həlˈoʊ wˈɜːld"; // Your generated IPA string

    // Split every single character (including stress marks) with an underscore.
    // let example_text = ipa_string.chars().map(|c| c.to_string()).collect::<Vec<String>>().join("_");
    let example_text = formated_ipa_string;

    let start = std::time::Instant::now();    
    let phonemes_ids = example_text.chars()
        .filter_map(|c| config.phoneme_id_map.get(&c.to_string()))
        .flat_map(|ids| ids.iter())
        .cloned()
        .collect::<Vec<i64>>();
    println!("Phoneme IDs extraction took: {:?}", start.elapsed());
    println!("Phoneme IDs: {:?}", phonemes_ids);

    let start = std::time::Instant::now();
    let phonemes_len = phonemes_ids.len();
    let phonems_ids_array = Array2::<i64>::from_shape_vec(
        [1, phonemes_len], 
        phonemes_ids
    ).expect("Failed to create phoneme IDs array");
    println!("Phoneme IDs array creation took: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let phonems_len_array = Array1::<i64>::from_shape_vec(
        [1], 
        vec![phonemes_len as i64]
    ).expect("Failed to create phoneme length array");
    println!("Phoneme length array creation took: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let scales_array = Array1::<f32>::from_shape_vec(
        [3], 
        vec![config.inference.noise_scale, config.inference.length_scale, config.inference.noise_w]
    ).expect("Failed to create scales array");
    println!("Scales array creation took: {:?}", start.elapsed());


    let start = std::time::Instant::now();
    let phonems_ids_tensor = ort::value::Tensor::from_array(phonems_ids_array)
        .expect("Failed to create tensor from phoneme IDs array");
    println!("Phoneme IDs tensor creation took: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let phonems_len_tensor = ort::value::Tensor::from_array(phonems_len_array)
        .expect("Failed to create tensor from phoneme length array");
    println!("Phoneme length tensor creation took: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let scales_tensor = ort::value::Tensor::from_array(scales_array)
        .expect("Failed to create tensor from scales array");
    println!("Scales tensor creation took: {:?}", start.elapsed());


    let inputs = ort::inputs!{
        "input" => phonems_ids_tensor,
        "input_lengths" => phonems_len_tensor,
        "scales" => scales_tensor,
    };
    
    let start = std::time::Instant::now();
    let outputs = model.run(inputs)?;
    println!("Model inference took: {:?}", start.elapsed());

    println!("Model outputs: {:?}", outputs);
    
    let (waveform_tensor_shape, waveform_tensor) = outputs["output"].try_extract_tensor::<f32>()?;
    
    println!("Waveform tensor shape: {:?}", waveform_tensor_shape);
    // println!("Waveform tensor: {:?}", waveform_tensor);


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
        (config.audio.sample_rate as u32).to_le_bytes().to_vec(), // SampleRate
        (config.audio.sample_rate as u32 * 2).to_le_bytes().to_vec(), // ByteRate
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
