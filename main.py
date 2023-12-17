from omegaconf import OmegaConf
from repple import Repple
from pathlib import Path
from tqdm import tqdm
import os
import hashlib
import torch
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
from slugify import slugify

from TTS.utils.generic_utils import get_user_data_dir
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

CONF_PATH = 'xtts_repl_conf.yaml'
DEFAULT_LATENTS_FOLDER = 'latents'
DEFAULT_REF_AUDIO_FOLDER = 'ref_audio'
DEFAULT_TOKENIZER_PATH = 'vocab.json'
MODEL_DIR = 'models'
OUT_DIR = 'results'
VALID_LANGUAGES = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl',
    'cs', 'ar', 'zh-cn', 'ja', 'hu', 'ko', 'hi']
WAV_TRUNCATE_LENGTH = 30
XTTS_V2_SAMPLE_RATE = 24000

class XTTS_REPL:
    def find_files(self):
        ret = {}
        ret['ckpts'] = []
        ret['latents'] = []
        ret['reference_audios'] = []

        for path,_,files in os.walk(MODEL_DIR):
            for file in files:
                file = Path(os.path.join(path,file))
                if (file.name.endswith('.pth')):
                    config_file = file.parent / 'config.json'
                    print(config_file)
                    if not os.path.exists(config_file):
                        print(f"Found no associated config.json for {file};"
                            f" skipping")
                    ret['ckpts'].append(str(file))

        for path in self.conf['reference_audio_search_dirs']:
            for path,_,files in os.walk(path):
                for file in files:
                    file = Path(os.path.join(path,file))
                    if any(file.name.lower().endswith(ext) for ext in 
                        ['.mp3','.wav','.flac','.ogg','.aac','.wma']):
                        ret['reference_audios'].append(str(file))

        for path in self.conf['latents_search_dirs']:
            for path,_,files in os.walk(path):
                for file in files:
                    file = Path(os.path.join(path,file))
                    if (file.name.endswith('.latents')):
                        ret['latents'].append(str(file))
        return ret

    def load_model_from_paths(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        config_path = ckpt_path.parent / "config.json"
        assert(os.path.exists(config_path))
        config = XttsConfig()
        config.load_json(config_path)
        print(f"Loading model {ckpt_path}...")
        self.model = Xtts.init_from_config(config)
        self.model_path = str(ckpt_path)
        self.config_path = config_path
        self.model_label = ckpt_path.parent.name
        self.model.load_checkpoint(config, checkpoint_path = ckpt_path,
            vocab_path = DEFAULT_TOKENIZER_PATH,
            checkpoint_dir = ckpt_path.parent)
        print(f"Loaded")
        self.model.cuda()

    def load_latent_from_audio(self, ref_audios):
        if ref_audios is None or not len(ref_audios):
            return None
        self.current_ref = ref_audios
        h = hashlib.blake2b(digest_size=20)
        h.update(self.model_path.encode('utf-8'))
        ref_audios = sorted(ref_audios)
        for ref_audio in ref_audios:
            h.update(ref_audio.encode('utf-8'))
        latents_key = str(h.hexdigest())

        if self.latents_cache.get(latents_key):
            return self.latents_cache[latents_key]

        print(f"Computing speaker latents for model {self.model_path}, "
            f"ref audios {ref_audios}...")
        gpt_cond_latent, spk_emb = self.model.get_conditioning_latents(
            audio_path=ref_audios)
        print(f"Done")
        self.latents_cache[latents_key] = (gpt_cond_latent, spk_emb)

        save_path = os.path.join(DEFAULT_LATENTS_FOLDER,latents_key+".latents")
        torch.save((gpt_cond_latent, spk_emb), save_path)

        self.current_latent = self.latents_cache[latents_key]

    def load_latent(self, l):
        if not os.path.exists(l):
            return
        latents_tuple = torch.load(l)
        latents_key = Path(l).name.removesuffix('.latents')
        self.latents_cache[latents_key] = latents_tuple

    def tts(self, text):
        if self.current_latent is None:
            print("No current latent/reference audio, select one first")
        gpt_cond_latent, speaker_embedding = self.current_latent
        self.last_text = text
        self.out_wavs = []
        for i in tqdm(range(self.conf.gen_repeats), "Generating texts"):
            out = self.model.inference(text, self.language, gpt_cond_latent,
                speaker_embedding, temperature=self.temperature)
            self.out_wavs.append(out["wav"])

    def __init__(self):
        if not os.path.exists(CONF_PATH):
            conf = OmegaConf.create({
                'reference_audio_search_dirs': [DEFAULT_REF_AUDIO_FOLDER],
                'latents_search_dirs': [DEFAULT_LATENTS_FOLDER],
                'default_ckpt': '',
                'default_reference_audio': '',
                'default_temperature': 0.7,
                'random_seed': 42,
                'gen_repeats': 3
            })
        else:
            conf = OmegaConf.load(CONF_PATH)
        files = self.find_files(conf)
        
        if not os.path.exists(DEFAULT_LATENTS_FOLDER):
            os.makedirs(DEFAULT_LATENTS_FOLDER)
        if not os.path.exists(DEFAULT_REF_AUDIO_FOLDER):
            os.makedirs(DEFAULT_REF_AUDIO_FOLDER)
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        if not os.path.exists(conf['default_ckpt']):
            conf['default_ckpt'] = ''
        if not os.path.exists(conf['default_reference_audio']):
            conf['default_reference_audio'] = ''
        
        files_checks = True
        if not len(files['ckpts']):
            print("No compatible XTTS checkpoints found")
            files_checks = False
        elif not len(files['reference_audios']):
            print("No reference audios found")
            files_checks = False

        if len(files['ckpts']) and not len(conf['default_ckpt']):
            conf['default_ckpt'] = files['ckpts'][0]
        if len(files['reference_audios']) and not len(
            conf['default_reference_audio']):
            conf['default_reference_audio'] = files['reference_audios'][0]

        self.conf = conf

        if not files_checks:
            print(f"Some files missing; please add files or "
                f"configure directories in conf ("
                f"{CONF_PATH})")
            return

        xtts_config = XttsConfig()
        self.load_model_from_paths(conf['default_ckpt'])
        
        self.latents_cache = {}
        self.files = files
        self.current_ref = [conf['default_reference_audio']]
        self.current_latent = self.load_latent_from_audio(self.current_ref)
        self.language = 'en'
        self.temperature = conf['default_temperature']
        self.out_wavs = []
        self.last_text = ""

        for latent_dir in conf['latents_search_dirs']:
            for path,_,files in os.walk(latent_dir):
                for file in files:
                    file = os.path.join(path, file)
                    if file.endswith('.latents'):
                        self.load_latent(file)

    def preview_audio(self, audio_data):
        audio_data = (audio_data * 32767).astype(np.int16)
        # 16 bit PCM, 24000 hz
        audio_segment = AudioSegment(audio_data.tobytes(),
            frame_rate=XTTS_V2_SAMPLE_RATE, 
            sample_width = audio_data.dtype.itemsize,
            channels=1)
        play(audio_segment)

    def save_audio(self, audio_data):
        audio_data = (audio_data * 32767).astype(np.int16)
        wav_label = (self.model_label+" "+
            self.last_text[:WAV_TRUNCATE_LENGTH])
        wav_label_pre = slugify(wav_label)
        wav_label = wav_label_pre + '.wav'
        i = 1
        while os.path.exists(wav_label):
            wav_label = wav_label_pre + str(i) + '.wav'
            i += 1
        print("Saving audio to "+wav_label)
        sf.write(os.path.join(OUT_DIR,wav_label),
            audio_data, XTTS_V2_SAMPLE_RATE)
        return os.path.join(OUT_DIR,wav_label)

    def repl(self):
        r = Repple()

        def select_reference_audios():
            self.current_latent = self.load_latent_from_audio(
                Repple.selector(self.files['reference_audios'],
                    select_str = "Select reference audios: "))

        def select_ckpt():
            self.load_model_from_paths(
                Repple.selector(self.files['ckpts'],
                select_str = "Select model checkpoint: ",
                is_unary=True)[0]
            )

        def refresh_files():
            self.files = self.find_files(self.conf)

        def set_temperature(temperature):
            self.temperature = temperature

        def select_language():
            self.language = Repple.selector(VALID_LANGUAGES,
                select_str = "Select language code: ",
                is_unary=True)[0]

        def show_state():
            print(f"Temperature: {self.temperature}")
            print(f"Language: {self.language}")
            print(f"Reference audios: {self.current_ref}")

        def preview_audio_by_idx(idx):
            if idx < len(self.out_wavs):
                self.preview_audio(self.out_wavs[idx])
            else:
                print(f"Request audio {idx} out of array bounds")

        def select_preview_audio():
            print("Audio previews from last text "+self.last_text)
            audio = Repple.selector(self.out_wavs,
                select_str = "Select audio to preview: ")
            if len(audio):
                self.preview_audio(audio[0])

        def save_audio_by_idx(*indices):
            for i in indices:
                if not i < len(self.out_wavs):
                    print(f"Request audio {i} out of array bounds")
                    continue
                self.save_audio(self.out_wavs[i])

        def select_save_audio():
            print("Audio from last text "+self.last_text)
            audio = Repple.selector(self.out_wavs,
                select_str = "Select audios to save: ")
            if not len(audio):
                return
            for aud in audio:
                self.save_audio(aud)

        def repeat_tts():
            self.tts(self.last_text)

        r['M'] = (select_ckpt, "Select model checkpoint and load model")
        r['R'] = (select_reference_audios, "Select reference audios")
        r['refr'] = (refresh_files, "Refresh files")
        r['temp'] = (set_temperature, "Set temperature")
        r.add_string_func('T', self.tts, desc="TTS")
        r['t'] = (repeat_tts, "Repeat last TTS request")
        r['lang'] = (select_language, "Select language")
        r['s'] = (show_state, "Show state (temp, lang, ref audio)")
        r['p'] = (preview_audio_by_idx, "Preview audio by index")
        r['pre'] = (select_preview_audio, "Select audio for preview")
        r['S'] = (save_audio_by_idx, "Save audio by index")
        r['Save'] = (select_save_audio, "Select audio to save")
        
        r.main()
        OmegaConf.save(config=self.conf, f=CONF_PATH)

if __name__ == '__main__':
    xtts_repl = XTTS_REPL()
    xtts_repl.repl()