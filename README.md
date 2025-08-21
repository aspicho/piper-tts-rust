# What's this?

This a simple interface for [Piper TTS](https://github.com/OHF-Voice/piper1-gpl) onnx models made with a [pykeio ort](https://ort.pyke.io) onnx runtime.

Unlike official C++/Python API It **does not** utilize [espeak-ng](https://github.com/espeak-ng/espeak-ng) for phonemization and instead relies on g2p seq2seq model based on the BART architecture.

Model intended to use is ~4.05M params [cisco-ai/mini-bart-g2p](https://huggingface.co/cisco-ai/mini-bart-g2p).

## Why?

Basically I needed to perform a TTS generation from Rust as a part of a web server, I could use official Python library as a microservice or C++ API to make a FFI binding, but I had a hard times building Piper TTS C++ binary on mac (skill issue tbh, I'm soy rust/python dev lol) and using *Python microservice* for my *Rust microservice* felt a little deranged.

I could use another TTS model/project, but Piper TTS fits perfectly for my use case as it fast, not resource intense and sounds good enough to be pleasant for ears while not crosses that border of sounding like slop trained straight up on ASMR and Hentai, so that's why I decided to make this.

## How it works?

Initially I wanted to use a `CMUdict-0.7` for phonemization, but ditched this idea in favor of using proper g2p model like [cisco-ai/mini-bart-g2p](https://huggingface.co/cisco-ai/mini-bart-g2p).

It generates an ARPAbet word phonetic transcription that looks like this:

`Hello world` -> `HH EH1 L OW0 W ER1 L D`

It later being converted into IPA representation via this [mapping](arpabet-mapping.txt):

`HH EH1 L OW0 W ER1 L D` -> `hˈɛloʊ wˈɜːld`

Then IPA string being formatted into the form Piper TTS uses:  

`hˈɛloʊ wˈɜːld` -> `^_h_ˈ_ɛ_l_o_ʊ_ _w_ˈ_ɜ_ː_l_d_$`

Finally this string being tokenized and after interference we get our waveform:

![example](example.wav)

## How to use

To use this interface you'll need [cisco-ai/mini-bart-g2p](https://huggingface.co/cisco-ai/mini-bart-g2p) `decoder_model.onnx` and `encoder_model.onnx` models,
any english [Piper TTS model](https://huggingface.co/rhasspy/piper-voices/tree/main/en) and ARPAbet to IPA [mapping](arpabet-mapping.txt).

```Rust
fn main() -> ort::Result<()> {    
    let decoder_path = "decoder_model.onnx".to_string();
    let encoder_path = "encoder_model.onnx".to_string();
    let tokenizer_path = "tokenizer.json".to_string();
    let vocab_path = "vocab.json".to_string();
    let arpabet_mapping_path = "arpabet-mapping.txt".to_string();

    let mut phoneme_gen = PhonemeGen::new(
        decoder_path, encoder_path, 
        tokenizer_path, vocab_path, 
        arpabet_mapping_path
    );
    phoneme_gen.load()?;

    let mut model = Model::new(
    "en_US-norman-medium.onnx",
    "en_US-norman-medium.onnx.json",
    ).expect("Failed to create model");

    let input_text = "Hello world".to_string().to_lowercase();

    let formated_ipa_string = phoneme_gen.process_text(&input_text)
        .expect("Failed to process text");

    let sample_rate = model.config.audio.sample_rate;

    let start = std::time::Instant::now();
    let (waveform_tensor_shape, waveform_tensor) = model.process_ipa_string(&formated_ipa_string)
        .expect("Failed to process IPA string"); 

    model.write_wav_file(
        &waveform_tensor, sample_rate, "output.wav"
    ).expect("Failed to write WAV file");

    Ok(())
}
```

## Limitations

[cisco-ai/mini-bart-g2p](https://huggingface.co/cisco-ai/mini-bart-g2p) is trained only on english words and can process only one word at a time, so sometimes because of lacking context it can produce a bit strange souding phonems. Also it split into `decoder_model.onnx` and `encoder_model.onnx`, so it can be not as straightforward as it could be to use another model, if even possible.
