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

import csv
from io import BytesIO
from os import path
import sys
import wave

import ffmpeg
import jiwer
import numpy as np
import stt

# Start CoquiSTT
model_path = "coqui-stt-0.9.3-models.tflite"
scorer_path = "coqui-stt-0.9.3-models.scorer"
model = stt.Model(model_path)
model.enableExternalScorer(scorer_path)

# Workaroud for csv field larger than field size error
MAX_INT = sys.maxsize
while True:
    # Decrease the MAX_INT value by factor 10 as long as the OverflowError occurs
    try:
        csv.field_size_limit(MAX_INT)
        break
    except OverflowError:
        MAX_INT = int(MAX_INT / 10)

# Apply text pre-processing and calculate WER
def calculate_wer(ground_truth, hypothesis):
    # Text pre-processing (remove punctuation, lowercase everything and convert
    # sentences to words) before evaluating WER
    transformation = jiwer.Compose(
        [jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.SentencesToListOfWords()]
    )
    result = jiwer.wer(
        ground_truth,
        hypothesis,
        truth_transoform=transformation,
        hypothesis_transform=transformation,
    )
    return result


# Convert audio to 16kHz sampling rate and 16-bit bit depth
def normalize_audio(audio):
    out, err = (
        ffmpeg.input("pipe:0")
        .output(
            "pipe:1",
            f="WAV",
            acodec="pcm_s16le",
            ac=1,
            ar="16k",
            loglevel="error",
            hide_banner=None,
        )
        .run(input=audio, capture_stdout=True, capture_stderr=True)
    )
    if err:
        raise Exception(err)
    return out


# Extract raw audio data to be fed into CoquiSTT
def process_audio(audio):
    audio = normalize_audio(audio)
    audio = BytesIO(audio)
    with wave.open(audio) as wav:
        processed_audio = np.frombuffer(wav.readframes(wav.getnframes()), np.int16)
    return processed_audio


def row_count(dir):
    with open(dir, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        total_rows = sum(1 for row in reader)
    return total_rows


# Create output csv file, read input csv file and write to output csv file
def process_csv(dir):
    if not path.isfile(dir):
        raise FileNotFoundError

    ground_truth_products = []
    ground_truth_non_products = []
    hypothesis_products = []
    hypothesis_non_products = []
    wer_products = None
    wer_non_products = None

    results_file = "results_blob.csv" if "blob" in dir else "results_clips.csv"

    with open(f"audio/{results_file}", "w", newline="") as results_csv:
        field_names = [
            "transcript",
            "coqui",
            "contains_product",
            "WER (products)",
            "WER (non-products)",
        ]
        writer = csv.DictWriter(results_csv, fieldnames=field_names)
        writer.writeheader()

        with open(dir, newline="") as clips_csv:
            reader = csv.DictReader(clips_csv)
            total_rows = row_count(dir)
            processed_counter = 0
            for row in reader:
                transcript = row["transcript"]
                contains_product = True if row["contains_product"] == "Y" else False
                ground_truth_products.append(
                    transcript
                ) if contains_product else ground_truth_non_products.append(transcript)
                if "blob" in dir:
                    hex_string = row["HEX(audio)"]
                    blob = bytearray.fromhex(hex_string.lstrip("0x"))
                    audio = process_audio(blob)
                else:
                    filename = row["filename"]
                    with open(f"audio/{filename}", "rb") as file:
                        audio = file.read()
                        audio = process_audio(audio)
                text = model.stt(audio)
                writer.writerow(
                    {
                        "transcript": f"{transcript}",
                        "coqui": f"{text}",
                        "contains_product": "Y" if contains_product else "N",
                    }
                )
                hypothesis_products.append(
                    text
                ) if contains_product else hypothesis_non_products.append(text)
                processed_counter += 1
                print(f"Clip {processed_counter} of {total_rows} processed")

        if ground_truth_products and hypothesis_products:
            wer_products = calculate_wer(ground_truth_products, hypothesis_products)
            print(f"WER (products): {wer_products}")
        if ground_truth_non_products and hypothesis_non_products:
            wer_non_products = calculate_wer(
                ground_truth_non_products, hypothesis_non_products
            )
            print(f"WER (non-products): {wer_non_products}")
        writer.writerow(
            {
                "WER (products)": f"{wer_products}" if wer_products else None,
                "WER (non-products)": f"{wer_non_products}"
                if wer_non_products
                else None,
            }
        )


def main():
    file_paths = ["audio/clips.csv", "audio/blob.csv"]

    for file in file_paths:
        try:
            process_csv(file)
            break
        except FileNotFoundError:
            continue
    else:
        raise FileNotFoundError("clips.csv/blob.csv does not exist in audio folder")


if __name__ == "__main__":
    main()
