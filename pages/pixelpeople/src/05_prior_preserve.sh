export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./data/people_sara"
export CLASS_DIR="./data/portraits"
export OUTPUT_DIR="./finetune/prior_sara"


accelerate launch ../diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=0.3 \
  --instance_prompt="a sara person, photograph" \
  --class_prompt="a person, photograph" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --num_class_images=69 \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
#  --train_text_encoder \
