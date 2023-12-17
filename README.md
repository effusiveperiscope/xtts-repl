# What is this
Quick and dirty REPL for running XTTS
# Installation
`pip install -r requirements.txt`
Install desired models and reference audios in `models` and `ref_audio`
respectively.
## Directory structure for added models
```
models/
    Celestia/
        config.json
        model_can_name_anything_as_long_as_it_is.pth
ref_audio/
    reference_audio1.flac
    reference_audio2.wav
    suggested_to_name_audios_after_speaker.wav
```

# Usage
`python main.py`
```
Command: h
        q                    Exits the program
        h                    Show this help
        M                    Select model checkpoint and load model
        R                    Select reference audios
        refr                 Refresh files
        temp <temperature>   Set temperature
        T <text>               TTS
        t                    Repeat last TTS request
        lang                 Select language
        s                    Show state (temp, lang, ref audio)
        p <idx>              Preview audio by index
        pre                  Select audio for preview
        S <indices>          Save audio by index
        Save                 Select audio to save
Command: T Hemanth stepped forward, only to be intercepted by one of Orgon's body guards.
Generating texts: 100%|██████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:05<00:00,  2.50s/it]
Command: p 0
Command: s 0
```

# Pruning
`python prune.py ckpt_to_prune.pth` can prune weights not used for inference
from the model, reducing disk size and file I/O time. *This will change hashes
for precomputed latents.*
