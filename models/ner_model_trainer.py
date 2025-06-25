# src/models/ner_model_trainer.py
import numpy as np
import evaluate
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from src.models.ner_model_trainer import NERDataProcessor
from src.utils.constants import NER_LABELS, LABELED_DATA_PATH, MODEL_OUTPUT_DIR

class NERModelTrainer:
    def __init__(self, model_checkpoint, output_dir=MODEL_OUTPUT_DIR, label_list=NER_LABELS):
        self.model_checkpoint = model_checkpoint
        self.output_dir = output_dir
        self.label_list = label_list
        self.id_to_label = {i: label for i, label in enumerate(label_list)}
        
        self.data_processor = NERDataProcessor(model_checkpoint, label_list)
        self.tokenizer = self.data_processor.tokenizer
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            num_labels=len(label_list),
            id2label=self.id_to_label,
            label2id={label: i for i, label in enumerate(label_list)}
        )
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        self.seqeval = evaluate.load("seqeval")

    def compute_metrics(self, p):
        """
        Computes NER-specific metrics (precision, recall, f1) using seqeval.
        """
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (where label is -100)
        true_predictions = [
            [self.id_to_label[p] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id_to_label[l] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def train(self, train_dataset, eval_dataset, num_train_epochs=5, per_device_train_batch_size=16, learning_rate=2e-5):
        """
        Configures and runs the fine-tuning process.
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size, # Same batch size for eval
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs",
            logging_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none" # Disable reporting to external services like Weights & Biases for simplicity
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        print(f"Starting training for {self.model_checkpoint}...")
        trainer.train()
        print(f"Training complete for {self.model_checkpoint}.")
        
        # Evaluate final model
        eval_results = trainer.evaluate()
        print(f"Final evaluation results for {self.model_checkpoint}: {eval_results}")

        # Save the best model
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Model and tokenizer saved to {self.output_dir}")
        
        return eval_results

# Example usage (can be run from main.py or a notebook)
async def run_task3_pipeline():
    print("--- Starting Task 3: Fine-Tuning NER Model ---")

    # Choose your model checkpoint
    # Recommended: "Davlan/bert-tiny-amharic" or "Davlan/bert-medium-amharic-finetuned-ner"
    # Or "xlm-roberta-base" if you prefer a general multilingual model
    model_checkpoint = "Davlan/bert-tiny-amharic" # Example model

    # Initialize data processor
    data_processor = NERDataProcessor(model_checkpoint, NER_LABELS)

    # Load and preprocess data
    raw_datasets = data_processor.load_conll_data(LABELED_DATA_PATH)
    tokenized_datasets = raw_datasets.map(
        data_processor.tokenize_and_align_labels,
        batched=True
    )

    # Initialize and train the model
    trainer = NERModelTrainer(model_checkpoint, label_list=NER_LABELS)
    
    # Ensure train_dataset and eval_dataset are correctly passed
    # For small datasets, you might need to adjust batch size or epochs.
    final_eval_results = trainer.train(
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        num_train_epochs=10, # Adjust based on dataset size and performance
        per_device_train_batch_size=8 # Adjust based on GPU memory
    )
    
    print("--- Task 3 Complete ---")
    return final_eval_results

if __name__ == '__main__':
    import asyncio
    asyncio.run(run_task3_pipeline())