# Coqui-Evaluation

This directory processes a csv file containing transcripts of audio clips and the audio data,
generating a csv file that contains the original transcript, the transcript produced by
passing the audio data to CoquiSTT, and the word error rate (WER). Sample audio data and
csv file are provided.

## Usage

### Using the script

1. Git clone the repository
2. Download and install [ffmpeg] (https://www.ffmpeg.org/download.html)
3. Download the [acoustic model] (https://github.com/coqui-ai/STT-models/releases/download/english/coqui/v0.9.3/model.tflite) and [language model] (https://github.com/coqui-ai/STT-models/releases/download/english/coqui/v0.9.3/coqui-stt-0.9.3-models.scorer) files for CoquiSTT and place it in the cloned repository
4. Create a venv using python -m venv venv
5. Enter venv using venv\scripts\activate (Windows) or source venv/bin/activate (Linux)
6. Run pip install -r requirements.txt
7. Run wer_calculator.py

### Preparing audio data and csv file

The audio data is provided as one of the following:

 1. Audio files located in the audio folder
 2. WAV blob data saved in the csv file

The csv file requires the following columns of data:

1. filename: Name of the audio file (only required if 
  audio data is provided in the form of audio files)
2. transcript: Transcription of the audio data
3. contains_product: Y if the transcript contains
  products and services provided by the organisation,
  otherwise N (optional)
4. HEX(audio): WAV blob data saved as a hexadecimal string
  (only required if audio data is provided in the form of WAV blob)

The csv file should be named "clips.csv" or "blob.csv" based on the type
of audio data provided (audio files/WAV blob).