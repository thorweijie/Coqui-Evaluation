import csv
from io import BytesIO
import wave

import ffmpeg
import jiwer
import numpy as np
import stt

ground_truth_products = []
ground_truth_non_products = []
hypothesis_products = []
hypothesis_non_products = []


# Text pre-processing (remove punctuation, lowercase everything and convert sentences to words) before evaluating WER
transformation = jiwer.Compose(
    [jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.SentencesToListOfWords()]
)

# Start CoquiSTT
model_path = "coqui-stt-0.9.3-models.tflite"
scorer_path = "coqui-stt-0.9.3-models.scorer"
model = stt.Model(model_path)
model.enableExternalScorer(scorer_path)


# Convert audio to 16kHz sampling rate and 16-bit depth
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


# Create output csv file, read input csv file and write to output csv file
with open("audio/results.csv", "w", newline="") as results_csv:
    field_names = ["filename", "transcript", "coqui", "contains_product"]
    writer = csv.DictWriter(results_csv, fieldnames=field_names)
    writer.writeheader()
    with open("audio/clips.csv", newline="") as clips_csv:
        reader = csv.DictReader(clips_csv)
        for row in reader:
            transcript = row["transcript"]
            contains_product = True if row["contains_product"] == "Y" else False
            ground_truth_products.append(
                transcript
            ) if contains_product else ground_truth_non_products.append(transcript)
            filename = row["filename"]
            with open(f"audio/{filename}", "rb") as file:
                audio = file.read()
                audio = normalize_audio(audio)
                audio = BytesIO(audio)
                with wave.open(audio) as wav:
                    audio = np.frombuffer(wav.readframes(wav.getnframes()), np.int16)
            text = model.stt(audio)
            writer.writerow(
                {
                    "filename": f"{filename}",
                    "transcript": f"{transcript}",
                    "coqui": f"{text}",
                    "contains_product": "Y" if contains_product else "N",
                }
            )
            hypothesis_products.append(
                text
            ) if contains_product else hypothesis_non_products.append(text)

# Apply text pre-processing and calculate WER
if ground_truth_products and hypothesis_products:
    result_products = jiwer.wer(
        ground_truth_products,
        hypothesis_products,
        truth_transoform=transformation,
        hypothesis_transform=transformation,
    )
    print(f"WER (products): {result_products}")


if ground_truth_non_products and hypothesis_non_products:
    result_non_products = jiwer.wer(
        ground_truth_non_products,
        hypothesis_non_products,
        truth_transoform=transformation,
        hypothesis_transform=transformation,
    )
    print(f"WER (non-products): {result_non_products}")


# print(f"Ground truth: {ground_truth}")
# print(f"Hypothesis: {hypothesis}")
