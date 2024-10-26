from datasets import load_dataset
from transformers import AutoTokenizer

class DataHandler:
    def __init__(self, model_name, dataset_name):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    def build_dataset(self, input_min_length, input_max_length):
        """Build and preprocess the dataset"""
        dataset = load_dataset(self.dataset_name, split="train")
        
        # Filter by length
        dataset = dataset.filter(
            lambda x: input_min_length < len(x["dialogue"]) <= input_max_length
        )

        # Process dataset
        dataset = dataset.map(self._tokenize, batched=False)
        dataset.set_format(type="torch")
        
        # Split dataset
        return dataset.train_test_split(test_size=0.2, shuffle=False, seed=42)

    def _tokenize(self, sample):
        """Tokenize a single sample"""
        prompt = f"""
        Summarize the following conversation.
        
        {sample["dialogue"]}
        
        Summary:
        """
        sample["input_ids"] = self.tokenizer.encode(prompt)
        sample["query"] = self.tokenizer.decode(sample["input_ids"])
        return sample
