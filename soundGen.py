import sys
import numpy as np
import soundfile as sf
from kokoro import KPipeline


def text_to_speech(text, output_file="sound.wav"):
    # Initialize Kokoro pipeline for American English
    pipeline = KPipeline(lang_code="a")

    # Generate audio chunks
    audio_chunks = []
    for _, _, audio in pipeline(text, voice="af_heart"):
        audio_chunks.append(audio)

    # Concatenate all audio chunks
    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        # Save as WAV file
        sf.write(output_file, full_audio, 24000)
        print(f"TTS audio saved to {output_file}")
    else:
        print("Error: No audio generated.")

if __name__ == "__main__":
    inputFile: str = sys.argv[1]
    readText: str
    with open(inputFile) as f:
        readText = f.read()

    text_to_speech(text=readText, output_file="second.wav")
