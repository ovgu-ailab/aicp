from diffusers import DiffusionPipeline
import torch
import argparse
import datetime

# prompt e.g.: 
# model e.g.:


parser = argparse.ArgumentParser(
                    prog='myPromt',
                    description='simple frontend')
parser.add_argument('prompt')
parser.add_argument('--model','-m')

args = parser.parse_args()

data_name = args.model
model_id = "./finetune/" + data_name + "/"
pipe = DiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch.float16).to("cuda")


for idx in range(5):
	print(idx)
	image = pipe(args.prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

	image.save("outputs/" + data_name + "{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now()) +  "_output_{}.png".format(idx))
