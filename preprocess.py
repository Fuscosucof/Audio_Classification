from datasets import load_dataset, Audio
from huggingface_hub import login
#login()

#minds = load_dataset("PolyAI/minds14", name='en-US', split="train", trust_remote_code=True)

#minds = minds.train_test_split(test_size=0.2)
''' dataset contains a lot of useful information, like lang_id and english_transcription, you will focus on the audio and intent_class in this guide. Remove the other columns with the remove_columns method:'''
#minds = minds.remove_columns(["path", "lang_id", "english_transcription", "transcription"])
'''audio: a 1-dimensional array of the speech signal that must be called to load and resample the audio file.
   intent_class: represents the class id of the speaker’s intent.'''

# ...existing code...
# After processing, save the dataset
#minds.save_to_disk("./processed_minds_dataset")

# In future runs, you can load the processed dataset directly:
from datasets import load_from_disk
minds = load_from_disk("./processed_minds_dataset")

labels = minds["train"].features["intent_class"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

#print(id2label["2"])

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

minds = minds.cast_column("audio", Audio(sampling_rate=16000))

'''Calls the audio column to load, and if necessary, resample the audio file.
Checks if the sampling rate of the audio file matches the sampling rate of the audio data a model was pretrained with. You can find this information in the Wav2Vec2 model card.
Set a maximum input length to batch longer inputs without truncating them.'''
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
    )
    return inputs

encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)
encoded_minds = encoded_minds.rename_column("intent_class", "label")


