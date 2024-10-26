import torch
from transformers import GenerationConfig
import evaluate
import numpy as np
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.toxicity_evaluator = evaluate.load(
            "toxicity",
            "facebook/roberta-hate-speech-dynabench-r4-target",
            module_type="measurement"
        )

    def evaluate_toxicity(self, dataset, num_samples):
        """Evaluate model toxicity on a dataset"""
        toxicities = []
        
        for i, sample in tqdm(enumerate(dataset)):
            if i >= num_samples:
                break
                
            score = self._evaluate_sample(sample)
            toxicities.extend(score["toxicity"])
        
        return np.mean(toxicities), np.std(toxicities)

    def _evaluate_sample(self, sample):
        """Evaluate a single sample"""
        input_text = sample["query"]
        input_ids = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True
        ).input_ids
        
        response_ids = self.model.generate(
            input_ids=input_ids,
            generation_config=self._get_generation_config()
        )
        
        generated_text = self.tokenizer.decode(
            response_ids[0],
            skip_special_tokens=True
        )
        
        return self.toxicity_evaluator.compute(
            predictions=[input_text + " " + generated_text]
        )

    def _get_generation_config(self):
        """Get generation configuration"""
        return GenerationConfig(
            max_new_tokens=100,
            top_k=0.0,
            top_p=1.0,
            do_sample=True
        )
