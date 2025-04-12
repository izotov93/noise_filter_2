import os
import wave
import numpy as np

def add_noise_to_speech(input_file, out_dir='noise_audio', noise_level=0.01, snr_level=5):
    """
    Adds noise to an audio file.

        :param input_file: Path to the input WAV file (clean speech).
        :param out_dir: Directory for the output file (with noise).
        :param noise_level: Level of noise (0.0 - no noise, 1.0 - strong noise).
        :param snr_level: level NOIZEUS database
        :return: The path to the output WAV file that contains the noisy audio.
    """

    with wave.open(input_file, 'rb') as wf:
        params = wf.getparams()
        print(f'Parameters wav file - {params}')

        n_channels, sampwidth, framerate, n_frames = params[:4]
        frames = wf.readframes(n_frames)
        audio = np.frombuffer(frames, dtype=np.int16)

    # White noise generation
    noise = np.random.normal(0, 1, len(audio))
    noise = noise_level * np.max(audio) * noise

    # # Adding noise to a signal
    noisy_audio = audio + noise
    noisy_audio = np.clip(noisy_audio, -32768, 32767).astype(np.int16)

    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f'{os.path.splitext(os.path.basename(input_file))[0]}_'
                                        f'white_noise_{noise_level}_sn{snr_level}.wav')
    with wave.open(output_file, 'wb') as wf_out:
        wf_out.setnchannels(n_channels)
        wf_out.setsampwidth(sampwidth)
        wf_out.setframerate(framerate)
        wf_out.writeframes(noisy_audio.tobytes())

    print(f"Noisy speech saved as {output_file}")

    return output_file


if __name__ == '__main__':

    dir_contained_sounds = 'NOIZEUS'
    snr_level = 5
    clean_dir = os.path.join(dir_contained_sounds, 'clean')
    clean_files = sorted([f for f in os.listdir(clean_dir) if f.startswith("sp") and f.endswith(".wav")])

    for noise_level in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]:
        for item in clean_files:
            add_noise_to_speech(input_file=os.path.join(clean_dir, item),
                                out_dir=os.path.join(os.path.dirname(clean_dir),
                                                     f'white_noise_{noise_level}',
                                                     f'{snr_level}dB'),
                                noise_level=noise_level,
                                snr_level=snr_level)

