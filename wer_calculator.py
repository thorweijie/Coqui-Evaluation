import csv
from io import BytesIO
from os import path
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


# Apply text pre-processing and calculate WER
def calculate_wer(ground_truth, hypothesis, is_product):

    # Text pre-processing (remove punctuation, lowercase everything and convert sentences to words) before evaluating WER
    transformation = jiwer.Compose(
        [jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.SentencesToListOfWords()]
    )
    result = jiwer.wer(
        ground_truth,
        hypothesis,
        truth_transoform=transformation,
        hypothesis_transform=transformation,
    )
    category = "products" if is_product else "non-products"
    print(f"WER ({category}): {result}")


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


# Create output csv file, read input csv file and write to output csv file
def process_csv(dir):
    if not path.isfile(dir):
        raise FileNotFoundError

    results_file = "results_blob.csv" if "blob" in dir else "results_clips.csv"

    ground_truth_products = []
    ground_truth_non_products = []
    hypothesis_products = []
    hypothesis_non_products = []

    with open(f"audio/{results_file}", "w", newline="") as results_csv:
        field_names = ["filename", "transcript", "coqui", "contains_product"]
        writer = csv.DictWriter(results_csv, fieldnames=field_names)
        writer.writeheader()

        with open(dir, newline="") as clips_csv:
            reader = csv.DictReader(clips_csv)
            for row in reader:
                transcript = row["transcript"]
                contains_product = True if row["contains_product"] == "Y" else False
                ground_truth_products.append(
                    transcript
                ) if contains_product else ground_truth_non_products.append(transcript)
                if "blob" in dir:
                    hex_string = row["HEX(audio)"]
                    blob = BytesIO(bytearray.fromhex(hex_string.lstrip("0x")))
                    audio = process_audio(blob)
                else:
                    filename = row["filename"]
                    with open(f"audio/{filename}", "rb") as file:
                        audio = file.read()
                        audio = process_audio(audio)
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

    if ground_truth_products and hypothesis_products:
        calculate_wer(ground_truth_products, hypothesis_products, is_product=True)
    if ground_truth_non_products and hypothesis_non_products:
        calculate_wer(
            ground_truth_non_products, hypothesis_non_products, is_product=False
        )


def main():
    file_paths = ["audio/clips.csv", "audio/blob.csv"]

    for path in file_paths:
        try:
            process_csv(path)
            break
        except:
            continue
    else:
        raise FileNotFoundError("clips.csv/blob.csv does not exist in audio folder")


if __name__ == "__main__":
    main()
