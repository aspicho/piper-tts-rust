use ndarray::{Array1, Array2};
use ort::{
    session::{builder::GraphOptimizationLevel, Session}, tensor::Shape, Error
};
use serde::Deserialize;
use std::{collections::HashMap, io::Write};

#[derive(Deserialize, Debug)]
pub struct Audio {
    pub sample_rate: u64,
    pub quality: String,
}

#[derive(Deserialize, Debug)]
pub struct Inference {
    pub noise_scale: f32,
    pub length_scale: f32,
    pub noise_w: f32,
}

#[derive(Deserialize, Debug)]
pub struct Language {
    pub code: String,
    pub family: String,
    pub region: String,
    pub name_native: String,
    pub name_english: String,
    pub country_english: String,
}

#[derive(Deserialize, Debug)]
pub struct Config {
    pub audio: Audio,
    pub inference: Inference,
    pub phoneme_id_map: HashMap<String, Vec<i64>>,
    pub language: Language,
}

pub struct Model  {
    pub config: Config,
    model: Session,
}

impl Model {
    pub fn new(model_path: &str, config_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
        
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(Model {
            config,
            model,
        })
    }

    pub fn ipa_string_to_phoneme_ids(
        &self,
        ipa_string: &str,
    ) -> Result<Vec<i64>, Error> {
        let phoneme_ids = ipa_string
            .chars()
            .filter_map(|c| self.config.phoneme_id_map.get(&c.to_string()))
            .flat_map(|ids| ids.iter())
            .cloned()
            .collect::<Vec<i64>>();
        
        Ok(phoneme_ids)
    }

    pub fn prepare_input(
        &self,
        phonemes_ids: Vec<i64>,
    ) -> Result<(Array2<i64>, Array1<i64>, Array1<f32>), Box<dyn std::error::Error>> {
        let phonemes_len = phonemes_ids.len();
        let phonems_ids_array = Array2::<i64>::from_shape_vec(
            [1, phonemes_len], 
            phonemes_ids
        )?;
        
        let phonems_len_array = Array1::<i64>::from_shape_vec(
            [1], 
            vec![phonemes_len as i64]
        )?;
        
        let scales_array = Array1::<f32>::from_shape_vec(
            [3], 
            vec![
                self.config.inference.noise_scale.clone(),
                self.config.inference.length_scale.clone(),
                self.config.inference.noise_w.clone()
            ]
        )?;

        Ok((phonems_ids_array, phonems_len_array, scales_array))
    }

    pub fn run_inference(
        &mut self,
        phonemes_ids: Vec<i64>
    ) -> Result<ort::session::SessionOutputs, Box<dyn std::error::Error>> {
        let (phonems_ids_array, phonems_len_array, scales_array) = self.prepare_input(phonemes_ids)?;

        let phonems_ids_tensor = ort::value::Tensor::from_array(phonems_ids_array)?;
        let phonems_len_tensor = ort::value::Tensor::from_array(phonems_len_array)?;
        let scales_tensor = ort::value::Tensor::from_array(scales_array)?;

        let inputs = ort::inputs!{
            "input" => phonems_ids_tensor,
            "input_lengths" => phonems_len_tensor,
            "scales" => scales_tensor,
        };

        Ok(self.model.run(inputs)?)
    }

    pub fn process_ipa_string(
        &mut self,
        ipa_string: &str,
    ) -> Result<(Shape, Vec<f32>), Box<dyn std::error::Error>> {
        let phoneme_ids = self.ipa_string_to_phoneme_ids(ipa_string)?;
        let outputs = self.run_inference(phoneme_ids)?;
        let (waveform_tensor_shape, waveform_tensor) = outputs["output"].try_extract_tensor::<f32>()?;
        
        Ok((waveform_tensor_shape.clone(), waveform_tensor.to_vec()))
    }

    pub fn write_wav_file(
        &self,
        waveform: &[f32],
        sample_rate: u64,
        output_path: &str,
    ) -> std::io::Result<()> {
        let start = std::time::Instant::now();
        let mut file = std::fs::File::create(output_path)
            .expect("Failed to create output file");

        let header: Vec<u8> = vec![
            "RIFF".as_bytes().to_vec(),
            (36 + waveform.len() as u32 * 2).to_le_bytes().to_vec(),
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
            (waveform.len() as u32 * 2).to_le_bytes().to_vec(), // Subchunk2Size
        ].concat();
        file.write_all(&header)
            .expect("Failed to write WAV header");

        let samples: Vec<i16> = waveform.iter()
            .map(|&sample| (sample * i16::MAX as f32) as i16)
            .collect();
        file.write_all(&samples.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>())
            .expect("Failed to write WAV samples");

        println!("WAV file created successfully at: {}", output_path);
        println!("WAV file creation took: {:?}", start.elapsed());

        Ok(())
    }
}

    
