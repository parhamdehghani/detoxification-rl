from src.data_handler import DataHandler
from src.model_setup import ModelSetup
from src.trainer import DetoxificationTrainer
from src.evaluator import Evaluator
from config import Config

def main():
    # Initialize data handler and load dataset
    data_handler = DataHandler(Config.MODEL_NAME, Config.DATASET_NAME)
    dataset = data_handler.build_dataset(
        Config.INPUT_MIN_LENGTH,
        Config.INPUT_MAX_LENGTH
    )
    
    # Setup models
    model_setup = ModelSetup(Config.MODEL_NAME, Config.PEFT_MODEL_PATH)
    ppo_model, ref_model, tokenizer = model_setup.setup_models()
    
    # Initialize trainer and train
    trainer = DetoxificationTrainer(
        ppo_model,
        ref_model,
        tokenizer,
        dataset["train"],
        Config.LEARNING_RATE
    )
    trainer.train(Config.MAX_STEPS)
    
    # Evaluate results
    evaluator = Evaluator(ppo_model, tokenizer)
    mean, std = evaluator.evaluate_toxicity(
        dataset["test"],
        Config.EVAL_SAMPLES
    )
    
    print(f"Final toxicity score: {mean:.3f} ± {std:.3f}")

if __name__ == "__main__":
    main()
