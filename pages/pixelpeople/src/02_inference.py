from diffusers import DiffusionPipeline
import torch

data_name = 'bob'
prompt = 'A blue cat playing with a red ball on the street.'
model_id = "./finetune/" + data_name + "/"

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")





for idx in range(10):
	print(idx)
	image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]


	image.save(data_name + '_output_{}.png'.format(idx))
