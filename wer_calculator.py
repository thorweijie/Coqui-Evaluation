"""
This script imports audio data and the corresponding transcript saved 
in a csv file located in an audio folder. The audio data is converted 
to 16kHz sampling rate and 16-bit bit depth before being sent to CoquiSTT
speech-to-text engine. The corresponding output is then used with the 
transcript to calculate word error rate (WER), which can be used to evaluate
the effectiveness of the model used for CoquiSTT.

The audio data is provided as one of the following:

 1) Audio files located in the audio folder
 2) WAV blob data saved in the csv file

The csv file requires the following columns of data:

- filename: Name of the audio file (only required if 
  audio data is provided in the form of audio files)
- transcript: Transcription of the audio data
- contains_product: Y if the transcript contains
  products and services provided by the organisation,
  otherwise N
- HEX(audio): WAV blob data saved as a hexadecimal string
  (only required if audio data is provided in the form of WAV blob)

The csv file should be named "clips.csv" or "blob.csv" based on the type
of audio data provided (audio files/WAV blob).
"""

import stt

from csv_processor import CsvProcessor


def main():
    # Start CoquiSTT
    model_path = "coqui-stt-0.9.3-models.tflite"
    scorer_path = "coqui-stt-0.9.3-models.scorer"
    model = stt.Model(model_path)
    model.enableExternalScorer(scorer_path)

    file_paths = ["audio/clips.csv", "audio/blob.csv"]

    for file in file_paths:
        try:
            processor = CsvProcessor(file, model)
            processor.process_csv()
            break
        except FileNotFoundError:
            continue
    else:
        raise FileNotFoundError("clips.csv/blob.csv does not exist in audio folder")


if __name__ == "__main__":
    main()
