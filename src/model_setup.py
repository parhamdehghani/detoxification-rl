import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType
from trl import AutoModelForSeq2SeqLMWithValueHead, create_reference_model

class ModelSetup:
    def __init__(self, model_name, peft_model_path=None):
        self.model_name = model_name
        self.peft_model_path = peft_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_models(self):
        """Initialize all required models"""
        lora_config = self._get_lora_config()
        base_model = self._load_base_model()
        peft_model = self._add_peft_adapter(base_model)
        ppo_model = self._setup_ppo_model(peft_model)
        ref_model = create_reference_model(ppo_model)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        return ppo_model, ref_model, tokenizer

    def _get_lora_config(self):
        """Get LoRA configuration"""
        return LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

    def _load_base_model(self):
        """Load the base model"""
        return AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16
        )

    def _add_peft_adapter(self, base_model):
        """Add PEFT adapter to base model"""
        if self.peft_model_path:
            return PeftModel.from_pretrained(
                base_model,
                self.peft_model_path,
                self._get_lora_config(),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                is_trainable=True
            )
        return base_model

    def _setup_ppo_model(self, model):
        """Setup PPO model"""
        return AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            is_trainable=True
        )
