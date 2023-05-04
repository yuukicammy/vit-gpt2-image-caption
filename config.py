from dataclasses import dataclass


@dataclass
class Config:
    # For modal
    project_name = "vit-gpt2-image-caption"
    shared_vol = "caption-vol"
    log_dir = "ViT-GPT2-Image-Caption"

    seed = 42

    ignore_pad_token_for_loss = True
    metric = "rouge"

    # For Hugging Face Dataset
    dataset_path = "yuukicammy/red_caps"
    download_retries = 1
    dataload_num_workers = 4  # [1,4]

    num_val_examples = 320

    # For Hugging Face Model
    image_processor_pretrained = "nlpconnect/vit-gpt2-image-captioning"
    tokenizer_pretrained = "nlpconnect/vit-gpt2-image-captioning"
    encoder_decoder_pretrained = "nlpconnect/vit-gpt2-image-captioning"
    max_target_length = 128

    train_args = {
        "predict_with_generate": True,
        "evaluation_strategy": "steps",
        "per_device_train_batch_size": 30,
        "per_device_eval_batch_size": 30,
        "eval_steps": 50,
        "save_steps": 50,
        "logging_steps": 50,
        "learning_rate": 5e-3,
        "logging_first_step": False,
        "seed": seed,
        "save_total_limit": 3,
        "max_steps": 1000000,
        "disable_tqdm": True,
        "resume_from_checkpoint": True,
        "remove_unused_columns": True,
        "lr_scheduler_type": "cosine_with_restarts",
        "optim": "adamw_torch",
        "push_to_hub": False,
        # "log_level": "debug",
        "dataloader_num_workers": dataload_num_workers,
        # "hub_model_id": "yuukicammy/ViT-GPT2-Image-Caption",
        # "hub_private_repo": True,
        # "hub_strategy": "end",
    }
