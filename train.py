from preprocess import minds, feature_extractor, encoded_minds, id2label, label2id
import evaluate
import numpy as np
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from accelerate import Accelerator
accelerator = Accelerator()

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)



num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
)

training_args = TrainingArguments(
    output_dir="fusco_mind_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_minds["train"],
    eval_dataset=encoded_minds["test"],
    processing_class=feature_extractor,
    compute_metrics=compute_metrics,)

def main():
    trainer.train()

if __name__ == "__main__":
    # Train the model
    main()
    trainer.push_to_hub(commit_message="Training complete")