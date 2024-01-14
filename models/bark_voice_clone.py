import torch
import torchaudio
import numpy as np
from bark_extras import (HuBERTManager, CustomHubert, CustomTokenizer, load_codec_model,
                         generate_text_semantic, preload_models, codec_decode, generate_coarse, generate_fine)
from encodec.utils import convert_audio

class BarkVoiceClone:
    def __init__(self, input_audio_file):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_audio_file = input_audio_file
        self.model = self.load_codec_model()
        self.hubert_manager = self.load_hubert_manager()
        self.hubert_model = self.load_hubert_model()
        self.tokenizer = self.load_tokenizer()

    def load_codec_model(self):
        return load_codec_model(use_gpu=True if self.device == 'cuda' else False)

    def load_hubert_manager(self):
        hubert_manager = HuBERTManager()
        hubert_manager.make_sure_hubert_installed()
        hubert_manager.make_sure_tokenizer_installed()
        return hubert_manager

    def load_hubert_model(self):
        return CustomHubert(checkpoint_path='data/models/hubert/hubert.pt').to(self.device)

    def load_tokenizer(self):
        return CustomTokenizer.load_from_checkpoint('data/models/hubert/tokenizer.pth').to(self.device)

    def process_audio(self):
        wav, sr = torchaudio.load(self.input_audio_file)
        wav = convert_audio(wav, sr, self.model.sample_rate, self.model.channels)
        wav = wav.to(self.device)
        return wav

    def generate_semantic_tokens(self, wav):
        semantic_vectors = self.hubert_model.forward(wav, input_sample_hz=self.model.sample_rate)
        semantic_tokens = self.tokenizer.get_token(semantic_vectors)
        return semantic_tokens

    def encode(self, wav):
        with torch.no_grad():
            encoded_frames = self.model.encode(wav.unsqueeze(0))
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()
        return codes

    def clone_voice(self, prompt, voice_name):
        processed_audio = self.process_audio()

        semantic_tokens = self.generate_semantic_tokens(processed_audio)
        codes = self.encode(processed_audio)

        output_path = 'assets/prompts/' + voice_name + '.npz'
        np.savez(output_path, fine_prompt=codes.cpu(), coarse_prompt=codes[:2, :].cpu(), semantic_prompt=semantic_tokens.cpu())

        preload_models(
            text_use_gpu=True,
            text_use_small=False,
            coarse_use_gpu=True,
            coarse_use_small=False,
            fine_use_gpu=True,
            fine_use_small=False,
            codec_use_gpu=True,
            force_reload=False,
            path="models"
        )

        # Generation with more control
        x_semantic = generate_text_semantic(prompt, history_prompt=voice_name, temp=0.7, top_k=50, top_p=0.95)
        x_coarse_gen = generate_coarse(x_semantic, history_prompt=voice_name, temp=0.7, top_k=50, top_p=0.95)
        x_fine_gen = generate_fine(x_coarse_gen, history_prompt=voice_name, temp=0.5)

        audio_array = codec_decode(x_fine_gen)
        return audio_array


# # Example usage:
# input_audio_file = 'path/to/your/audio/file.wav'
# bark_voice_clone = BarkVoiceClone(input_audio_file)
# audio_array = bark_voice_clone.clone_voice(prompt="Your prompt here", voice_name="output_voice1")
