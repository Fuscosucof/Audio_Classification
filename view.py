from datasets import load_dataset, Audio
from transformers import pipeline
from huggingface_hub import login
#login()

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
sampling_rate = dataset.features["audio"].sampling_rate
audio_file = dataset[1]["audio"]["path"]

classifier = pipeline("audio-classification", model="Fuscosucof/fusco_mind_model", chunk_length_s=1.0, sampling_rate=sampling_rate)
results = classifier(audio_file)

from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("stevhliu/my_awesome_minds_model")
inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

from transformers import AutoModelForAudioClassification
import torch

model = AutoModelForAudioClassification.from_pretrained("stevhliu/my_awesome_minds_model")
with torch.no_grad():
    logits = model(**inputs).logits


predicted_class_ids = torch.argmax(logits).item()
predicted_label = model.config.id2label[predicted_class_ids]
print(predicted_label)
# cash_deposit
