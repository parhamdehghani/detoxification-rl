# Language Model Detoxification using Reinforcement Learning

A PyTorch implementation of Language Model detoxification using Reinforcement Learning. This project uses Proximal Policy Optimization (PPO) with Meta AI's hate speech classifier as a reward model to fine-tune FLAN-T5 for generating less toxic content.

## Overview

This project implements:
- FLAN-T5 base model with Parameter-Efficient Fine-Tuning (PEFT/LoRA)
- Proximal Policy Optimization (PPO) for reinforcement learning
- Meta AI's hate speech classifier for toxicity measurement
- DialogSum dataset for training and evaluation

## Results
- 17.41% reduction in mean toxicity scores
- Final toxicity score: 0.029 ± 0.035 
- Improved content generation while maintaining semantic meaning

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/detoxification-rl.git
cd detoxification-rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure
```
detoxification-rl/
├── src/
│   ├── data_handler.py    # Dataset processing and management
│   ├── model_setup.py     # Model initialization and configuration
│   ├── trainer.py         # PPO training implementation
│   ├── evaluator.py       # Toxicity evaluation metrics
│   └── utils.py           # Helper functions
├── scripts/
│   └── train.py          # Main training script
├── requirements.txt
└── config.py
```

## Dependencies
```txt
torch==1.13.1
torchdata==0.5.1
transformers==4.27.2
evaluate==0.4.0
peft==0.3.0
trl @ git+https://github.com/lvwerra/trl.git@25fa1bd
datasets==2.17.0
numpy
pandas
tqdm
```

## Usage

### Basic Training
```bash
python scripts/train.py
```

### Custom Training
```python
from src.data_handler import DataHandler
from src.model_setup import ModelSetup
from src.trainer import DetoxificationTrainer

# Initialize
data_handler = DataHandler(model_name, dataset_name)
dataset = data_handler.build_dataset(min_length, max_length)

# Setup models
model_setup = ModelSetup(model_name, peft_path)
ppo_model, ref_model, tokenizer = model_setup.setup_models()

# Train
trainer = DetoxificationTrainer(ppo_model, ref_model, tokenizer, dataset)
trainer.train(max_steps)
```

## Configuration
Modify `config.py` for custom settings:
```python
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
```

## Components

### Data Handler
- Dataset loading and preprocessing
- Tokenization and formatting
- Train/test split functionality

### Model Setup
- FLAN-T5 initialization with PEFT
- PPO model configuration
- LoRA implementation

### Trainer
- PPO training loop
- Reward calculation
- Batch processing
- Learning rate scheduling

### Evaluator
- Toxicity evaluation
- Statistical analysis
- Performance comparison

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


