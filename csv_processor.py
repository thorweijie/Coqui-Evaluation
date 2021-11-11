import csv
from io import BytesIO
from os import path
import sys
import wave

import ffmpeg
import jiwer
import numpy as np


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


class CsvProcessor:
    def __init__(self, dir, model):
        if not path.isfile(dir):
            raise FileNotFoundError
        self.dir = dir
        self.model = model
        self.ground_truth_products = []
        self.ground_truth_non_products = []
        self.hypothesis_products = []
        self.hypothesis_non_products = []
        self.wer_products = None
        self.wer_non_products = None

        # Workaroud for csv field larger than field size error
        MAX_INT = sys.maxsize
        while True:
            # Decrease the MAX_INT value by factor 10 as long as the OverflowError occurs
            try:
                csv.field_size_limit(MAX_INT)
                break
            except OverflowError:
                MAX_INT = int(MAX_INT / 10)

    # Create output csv file, read input csv file and write to output csv file
    def process_csv(self):
        results_file = "results_blob.csv" if "blob" in self.dir else "results_clips.csv"

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

            with open(self.dir, newline="") as clips_csv:
                total_rows = row_count(self.dir)
                reader = csv.DictReader(clips_csv)
                processed_counter = 0
                for row in reader:
                    transcript = row["transcript"]
                    contains_product = True if row["contains_product"] == "Y" else False
                    self.ground_truth_products.append(
                        transcript
                    ) if contains_product else self.ground_truth_non_products.append(
                        transcript
                    )
                    if "blob" in self.dir:
                        hex_string = row["HEX(audio)"]
                        blob = bytearray.fromhex(hex_string.lstrip("0x"))
                        audio = process_audio(blob)
                    else:
                        filename = row["filename"]
                        with open(f"audio/{filename}", "rb") as file:
                            audio = file.read()
                            audio = process_audio(audio)
                    text = self.model.stt(audio)
                    writer.writerow(
                        {
                            "transcript": f"{transcript}",
                            "coqui": f"{text}",
                            "contains_product": "Y" if contains_product else "N",
                        }
                    )
                    self.hypothesis_products.append(
                        text
                    ) if contains_product else self.hypothesis_non_products.append(text)
                    processed_counter += 1
                    print(f"Clip {processed_counter} of {total_rows} processed")

            if self.ground_truth_products and self.hypothesis_products:
                self.wer_products = calculate_wer(
                    self.ground_truth_products, self.hypothesis_products
                )
                print(f"WER (products): {self.wer_products}")
            if self.ground_truth_non_products and self.hypothesis_non_products:
                self.wer_non_products = calculate_wer(
                    self.ground_truth_non_products, self.hypothesis_non_products
                )
                print(f"WER (non-products): {self.wer_non_products}")
            writer.writerow(
                {
                    "WER (products)": f"{self.wer_products}"
                    if self.wer_products
                    else None,
                    "WER (non-products)": f"{self.wer_non_products}"
                    if self.wer_non_products
                    else None,
                }
            )
