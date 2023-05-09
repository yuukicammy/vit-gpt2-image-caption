from dataclasses import dataclass


@dataclass
class Config:
    # For modal
    project_name = "vit-gpt2-image-caption"
    shared_vol = "image-caption-vol"
    output_dir = "image-caption"

    seed = 42

    ignore_pad_token_for_loss = True
    metric = "rouge"

    # For Dataset
    dataset_path = "coco"  # ["red-caps-5k-01", "coco"]
    val_examples = 200  # only COCO
    log_examples = 10

    # For Hugging Face Model
    image_processor_pretrained = "nlpconnect/vit-gpt2-image-captioning"
    tokenizer_pretrained = "nlpconnect/vit-gpt2-image-captioning"
    encoder_decoder_pretrained = "nlpconnect/vit-gpt2-image-captioning"
    max_target_length = 80
    num_beams = 8

    train_args = {
        "predict_with_generate": True,
        "evaluation_strategy": "steps",
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 64,
        "eval_steps": 10,
        "save_steps": 500,
        "logging_steps": 10,
        "learning_rate": 3e-5,
        "logging_first_step": True,
        "generation_max_length": max_target_length,
        "generation_num_beams": num_beams,
        "seed": seed,
        "save_total_limit": 3,
        "max_steps": 2000,
        "disable_tqdm": False,
        "resume_from_checkpoint": True,
        "remove_unused_columns": True,
        "lr_scheduler_type": "cosine_with_restarts",
        "optim": "adamw_torch",
        "load_best_model_at_end": True,
        # "log_level": "debug",
        "dataloader_num_workers": 12,
        "push_to_hub": False,
        "hub_model_id": "yuukicammy/vit-gpt2-image-caption",
        "hub_private_repo": True,
        "hub_strategy": "every_save",
        "metric_for_best_model": "rouge2",
        "greater_is_better": True,
    }
