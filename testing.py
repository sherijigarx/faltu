import voice_cloning_complex as VoiceCloningComplex
import scipy

prompt = "Hello, I am Eva. People say that I talk too much [laughs]"
voice_name = 'output_voice1'
input_audio_file = 'female.mp3'

audio_array = VoiceCloningComplex.main(prompt, voice_name, input_audio_file)

# Display audio
# Audio(audio_array, rate=SAMPLE_RATE)
scipy.io.wavfile.write(f"voice_clong_gen.wav", rate=24000, data=audio_array)