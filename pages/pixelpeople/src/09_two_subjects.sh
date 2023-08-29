# export PYTHONPATH="${PYTHONPATH}:/home/skurs/pixelpeople"

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export MODEL_NAME="/home/skurs/pixelpeople/finetune/people_felix"
export OUTPUT_DIR="./finetune/two_subjects_sara_ege/"

# Subject 1
export INSTANCE_DIR_1="./data/people_sara/"
export INSTANCE_PROMPT_1="a photo of person Sara"
export CLASS_DIR_1="./data/portraits/"
export CLASS_PROMPT_1="a photo of a person"

# Subject 2
export INSTANCE_DIR_2="./data/people_ege_extended/"
export INSTANCE_PROMPT_2="a photo of person Ege"
export CLASS_DIR_2="./data/portraits/"
export CLASS_PROMPT_2="a photo of a person"

accelerate launch /home/skurs/diffusers/examples/research_projects/multi_subject_dreambooth/train_multi_subject_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir="$INSTANCE_DIR_1,$INSTANCE_DIR_2" \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --instance_prompt="$INSTANCE_PROMPT_1,$INSTANCE_PROMPT_2" \
  --class_data_dir="$CLASS_DIR_1,$CLASS_DIR_2" \
  --class_prompt="$CLASS_PROMPT_1,$CLASS_PROMPT_2"\
  --num_class_images=50 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1500






