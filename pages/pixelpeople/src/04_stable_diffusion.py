from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
from compel import Compel

data_name = 'kernel_method'
prompt = "A fancy visualization of machine learning using kernel methods."

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to('cuda')

compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
prompt_embeds = compel_proc(prompt)




pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# generator = torch.Generator(device="gpu").manual_seed(33)


for idx in range(10):
	print(idx)	
	
	image = pipe(prompt_embeds=prompt_embeds, num_inference_steps=20).images[0]
	# image = pipe(prompt, num_inference_steps=20).images[0]
	# image = pipe(prompt, generator=generator, num_inference_steps=20).images[0]
	
	image.save(data_name + '_output_{}.png'.format(idx))



