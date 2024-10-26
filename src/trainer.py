import torch
from trl import PPOTrainer, PPOConfig
from trl.core import LengthSampler
from transformers import pipeline
from tqdm import tqdm

class DetoxificationTrainer:
    def __init__(self, ppo_model, ref_model, tokenizer, dataset, learning_rate=1.41e-5):
        self.config = PPOConfig(
            model_name="google/flan-t5-base",
            learning_rate=learning_rate,
            ppo_epochs=1,
            mini_batch_size=4,
            batch_size=16
        )
        
        self.trainer = PPOTrainer(
            config=self.config,
            model=ppo_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=dataset
        )
        
        self.sentiment_pipe = self._setup_reward_model()
        self.output_length_sampler = LengthSampler(100, 400)

    def _setup_reward_model(self):
        """Setup the reward model"""
        return pipeline(
            "sentiment-analysis",
            model="facebook/roberta-hate-speech-dynabench-r4-target",
            device=0 if torch.cuda.is_available() else "cpu"
        )

    def train(self, max_steps=10):
        """Train the model using PPO"""
        generation_kwargs = {
            "min_length": 5,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True
        }
        
        reward_kwargs = {
            "top_k": None,
            "function_to_apply": "none",
            "batch_size": 16
        }
        
        for step, batch in tqdm(enumerate(self.trainer.dataloader)):
            if step >= max_steps:
                break
                
            stats = self._training_step(
                batch, 
                generation_kwargs, 
                reward_kwargs
            )
            self.trainer.log_stats(stats, batch, reward_tensors)

    def _training_step(self, batch, generation_kwargs, reward_kwargs):
        """Perform a single training step"""
        prompt_tensors = batch["input_ids"]
        summary_tensors = self._generate_responses(
            prompt_tensors, 
            generation_kwargs
        )
        
        batch["response"] = [
            self.trainer.tokenizer.decode(r.squeeze()) 
            for r in summary_tensors
        ]
        
        reward_tensors = self._calculate_rewards(
            batch, 
            reward_kwargs
        )
        
        return self.trainer.step(
            prompt_tensors, 
            summary_tensors, 
            reward_tensors
        )
