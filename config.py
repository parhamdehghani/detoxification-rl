class Config:
    # Model configurations
    MODEL_NAME = "google/flan-t5-base"
    DATASET_NAME = "knkarthick/dialogsum"
    PEFT_MODEL_PATH = "./peft-model-checkpoint"
    
    # Training configurations
    INPUT_MIN_LENGTH = 200
    INPUT_MAX_LENGTH = 1000
    MAX_STEPS = 10
    LEARNING_RATE = 1.41e-5
    
    # Evaluation configurations
    EVAL_SAMPLES = 10
