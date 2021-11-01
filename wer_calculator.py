import csv
import wave

import jiwer
import numpy as np
import stt

ground_truth = []
hypothesis = []


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
    field_names = ["filename", "transcript", "coqui"]
    writer = csv.DictWriter(results_csv, fieldnames=field_names)
    writer.writeheader()
    with open("audio/clips.csv", newline="") as clips_csv:
        reader = csv.DictReader(clips_csv)
        for row in reader:
            transcript = row["transcript"]
            ground_truth.append(transcript)
            filename = row["filename"]
            with wave.open(f"audio/{filename}", "rb") as wav:
                audio = np.frombuffer(wav.readframes(wav.getnframes()), np.int16)
            text = model.stt(audio)
            writer.writerow(
                {
                    "filename": f"{filename}",
                    "transcript": f"{transcript}",
                    "coqui": f"{text}",
                }
            )
            hypothesis.append(text)

# Apply text pre-processing and calculate WER
result = jiwer.wer(
    ground_truth,
    hypothesis,
    truth_transoform=transformation,
    hypothesis_transform=transformation,
)

# print(f"Ground truth: {ground_truth}")
# print(f"Hypothesis: {hypothesis}")
print(f"WER: {result}")
