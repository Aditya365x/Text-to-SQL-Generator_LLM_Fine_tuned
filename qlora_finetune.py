"""
QLoRA Fine-Tuning Script
Supports any HuggingFace causal LM model with configurable dataset, hyperparameters, and saving.
"""

import os, gc, time, torch, sys, json
from dataclasses import dataclass, field, asdict
from typing import Optional
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainerCallback, HfArgumentParser,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ============================================================================
# ARGUMENTS
# ============================================================================

@dataclass
class ModelArgs:
    model_name: str = field(default="google/gemma-3-1b-it")
    adapter_name: str = field(default="my-qlora-adapter")
    hf_token: Optional[str] = field(default=None)

@dataclass
class DataArgs:
    dataset_name: str = field(default="b-mc2/sql-create-context")
    dataset_split: str = field(default="train")
    num_samples: int = field(default=200)
    text_column: str = field(default="text")
    max_seq_length: int = field(default=256)
    dataset_shuffle: bool = field(default=True)
    shuffle_seed: int = field(default=42)

@dataclass
class LoraArgs:
    r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    target_modules: str = field(default="q_proj,v_proj")

@dataclass
class TrainArgs:
    output_dir: str = field(default="./output")
    num_train_epochs: int = field(default=1)
    per_device_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-4)
    warmup_steps: int = field(default=5)
    logging_steps: int = field(default=1)
    save_strategy: str = field(default="epoch")
    lr_scheduler_type: str = field(default="cosine")
    max_grad_norm: float = field(default=0.3)
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)

@dataclass
class SaveArgs:
    save_local_path: str = field(default="./saved_model")
    save_to_hub: bool = field(default=False)
    hub_repo_id: Optional[str] = field(default=None)


# ============================================================================
# FORMAT FUNCTION (customise per task)
# ============================================================================

def format_sql_prompt(example):
    text = f"""<bos><start_of_turn>system
You are an expert SQL assistant. Write only the SQL query, no explanation.<end_of_turn>
<start_of_turn>user
Schema: {example['context']}
Question: {example['question']}<end_of_turn>
<start_of_turn>model
{example['answer']}<end_of_turn><eos>"""
    return {"text": text}


# ============================================================================
# PROGRESS BAR CALLBACK
# ============================================================================

class ProgressCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.max_steps > 0:
            pct = state.global_step / state.max_steps * 100
            filled = int(20 * state.global_step // state.max_steps)
            bar = "█" * filled + "░" * (20 - filled)
            print(f"\rStep {state.global_step}/{state.max_steps} |{bar}| {pct:.0f}%", end="")
            sys.stdout.flush()
    def on_train_end(self, args, state, control, **kwargs):
        print("\n✅ Training Complete!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = HfArgumentParser((ModelArgs, DataArgs, LoraArgs, TrainArgs, SaveArgs))
    model_args, data_args, lora_args, train_args, save_args = parser.parse_args_into_dataclasses()

    # --- Clear memory ---
    gc.collect()
    torch.cuda.empty_cache()

    # --- Load dataset ---
    print(f"\n📦 Loading dataset: {data_args.dataset_name}")
    dataset = load_dataset(data_args.dataset_name, split=data_args.dataset_split)
    if data_args.dataset_shuffle:
        dataset = dataset.shuffle(seed=data_args.shuffle_seed)
    if data_args.num_samples:
        dataset = dataset.select(range(min(data_args.num_samples, len(dataset))))
    print(f"   -> {len(dataset)} samples")

    # --- Tokenizer ---
    print(f"\n🔤 Loading tokenizer: {model_args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        token=model_args.hf_token,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # --- Format dataset ---
    dataset = dataset.map(format_sql_prompt, remove_columns=dataset.column_names)

    # --- BitsAndBytes 4-bit config ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # --- Load base model ---
    print(f"\n🤖 Loading base model: {model_args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=model_args.hf_token,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # --- LoRA config ---
    target_modules = [m.strip() for m in lora_args.target_modules.split(",")]
    peft_config = LoraConfig(
        r=lora_args.r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # --- Training arguments ---
    training_args = SFTConfig(
        output_dir=train_args.output_dir,
        num_train_epochs=train_args.num_train_epochs,
        per_device_train_batch_size=train_args.per_device_batch_size,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        logging_steps=train_args.logging_steps,
        logging_strategy="steps",
        learning_rate=train_args.learning_rate,
        fp16=train_args.fp16,
        bf16=train_args.bf16,
        max_grad_norm=train_args.max_grad_norm,
        warmup_steps=train_args.warmup_steps,
        lr_scheduler_type=train_args.lr_scheduler_type,
        save_strategy=train_args.save_strategy,
        report_to="none",
        max_seq_length=data_args.max_seq_length,
        packing=False,
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )
    trainer.add_callback(ProgressCallback())

    total_steps = trainer.state.max_steps or (
        (len(dataset) // (train_args.per_device_batch_size * train_args.gradient_accumulation_steps))
        * train_args.num_train_epochs
    )
    print(f"\n{'='*60}")
    print(f"TRAINING")
    print(f"   Model:    {model_args.model_name}")
    print(f"   Samples:  {len(dataset)}")
    print(f"   Epochs:   {train_args.num_train_epochs}")
    print(f"   Batch:    {train_args.per_device_batch_size}")
    print(f"   Grad Acc: {train_args.gradient_accumulation_steps}")
    print(f"   Steps:    ~{total_steps}")
    print(f"   LoRA r:   {lora_args.r}, alpha: {lora_args.lora_alpha}")
    print(f"   Targets:  {target_modules}")
    print(f"{'='*60}")

    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"\n⏱️  Training time: {elapsed:.1f}s")

    # --- Save locally ---
    os.makedirs(save_args.save_local_path, exist_ok=True)
    trainer.model.save_pretrained(save_args.save_local_path)
    tokenizer.save_pretrained(save_args.save_local_path)
    print(f"\n💾 Model saved to: {save_args.save_local_path}")

    for fname in os.listdir(save_args.save_local_path):
        fpath = os.path.join(save_args.save_local_path, fname)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"   📄 {fname} ({size_kb:.1f} KB)")

    # --- Save to HuggingFace Hub ---
    if save_args.save_to_hub and save_args.hub_repo_id:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=save_args.hub_repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=save_args.save_local_path,
            repo_id=save_args.hub_repo_id,
            repo_type="model",
        )
        print(f"☁️  Model pushed to Hub: {save_args.hub_repo_id}")


if __name__ == "__main__":
    main()
