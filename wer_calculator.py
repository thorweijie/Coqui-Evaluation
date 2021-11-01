import csv
import wave

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
            with wave.open(f"audio/{filename}", "rb") as wav:
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
result_products = jiwer.wer(
    ground_truth_products,
    hypothesis_products,
    truth_transoform=transformation,
    hypothesis_transform=transformation,
)

result_non_products = jiwer.wer(
    ground_truth_non_products,
    hypothesis_non_products,
    truth_transoform=transformation,
    hypothesis_transform=transformation,
)

# print(f"Ground truth: {ground_truth}")
# print(f"Hypothesis: {hypothesis}")
print(f"WER (products): {result_products}")
print(f"WER (non-products): {result_non_products}")
