from coding import HuBERTManager, CustomHubert, CustomTokenizer, load_codec_model, generate_text_semantic, preload_models, codec_decode, generate_coarse, generate_fine
from encodec.utils import convert_audio
import torchaudio
import torch
import numpy as np


class ModelLoader:
   def _init_(self, device):
       self.device = device
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
   

class AudioProcessor:
  def _init_(self, filepath, model, device):
      self.filepath = filepath
      self.model = model
      self.device = device
      self.wav, self.sr = torchaudio.load(self.filepath)

  def process_audio(self):
      self.wav = convert_audio(self.wav, self.sr, self.model.sample_rate, self.model.channels)
      self.wav = self.wav.to(self.device)
      return self.wav


class SemanticGenerator:
   def _init_(self, hubert_model, tokenizer, wav, model):
       self.hubert_model = hubert_model
       self.tokenizer = tokenizer
       self.wav = wav
       self.model = model

   def generate_semantic_tokens(self):
       semantic_vectors = self.hubert_model.forward(self.wav, input_sample_hz=self.model.sample_rate)
       semantic_tokens = self.tokenizer.get_token(semantic_vectors)
       return semantic_tokens

class Encoder:
   def _init_(self, model, wav):
       self.model = model
       self.wav = wav

   def encode(self):
       with torch.no_grad():
           encoded_frames = self.model.encode(self.wav.unsqueeze(0))
       codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()
       return codes


class AudioGenerator:
  def _init_(self, text_prompt, voice_name):
      self.text_prompt = text_prompt
      self.voice_name = voice_name

  def generate_audio(self):
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
      x_semantic = generate_text_semantic(
          self.text_prompt,
          history_prompt=self.voice_name,
          temp=0.7,
          top_k=50,
          top_p=0.95,
      )

      x_coarse_gen = generate_coarse(
          x_semantic,
          history_prompt=self.voice_name,
          temp=0.7,
          top_k=50,
          top_p=0.95,
      )

      x_fine_gen = generate_fine(
          x_coarse_gen,
          history_prompt=self.voice_name,
          temp=0.5,
      )

      audio_array = codec_decode(x_fine_gen)

      return audio_array


def main(prompt, voice_name, input_audio_file):
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_loader = ModelLoader(device)

    # Process audio
    audio_processor = AudioProcessor(input_audio_file, model_loader.model, device)
    processed_audio = audio_processor.process_audio()

    # Generate semantic tokens
    semantic_generator = SemanticGenerator(model_loader.hubert_model, model_loader.tokenizer, processed_audio, model_loader.model)
    semantic_tokens = semantic_generator.generate_semantic_tokens()

    # Encode audio
    encoder = Encoder(model_loader.model, processed_audio)
    codes = encoder.encode()

    # Save prompts
    output_path = voice_name + '.npz'
    np.savez(output_path, fine_prompt=codes.cpu(), coarse_prompt=codes[:2, :].cpu(), semantic_prompt=semantic_tokens.cpu())

    # Generate audio
    audio_generator = AudioGenerator(prompt, voice_name)
    audio_array = audio_generator.generate_audio()

    return audio_array

# if __name__ == "__main__":
#     output_audio_array = main(prompt, voice_name, input_audio_file)