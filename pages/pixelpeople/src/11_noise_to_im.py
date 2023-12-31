from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

from PIL import Image
import torch
import numpy as np
import argparse
import datetime


def save_tensor(latents, filename):
	# scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(filename)

basename = "outputs/noise/"  + "{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())

parser = argparse.ArgumentParser(
                    prog='noise',
                    description='adds noise')

parser.add_argument('prompt')
parser.add_argument('filename')

args = parser.parse_args()

img = Image.open(args.filename)

# if img.mode == "RGB":
#     a_channel = Image.new('L', img.size, 255)   # 'L' 8-bit pixels, black and white
#     img.putalpha(a_channel)


vec = np.array(img).astype(float)
for i in range(vec.shape[0]):
	for j in range(vec.shape[1]):
		if vec[i][j][3] == 0:
			vec[i][j] = 0.5*255


print(vec.shape)
vec = vec/255.0
vec = vec - 0.5
vec = 2*vec
vec = np.transpose(vec,[2,0,1])
vec = np.array([vec])

print(vec.dtype)
img_tensor = torch.from_numpy(vec).to('cuda')
img_tensor = img_tensor.to(torch.float32)


# 1. Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# 2. Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")


from diffusers import LMSDiscreteScheduler

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

prompt = [args.prompt]

height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion

num_inference_steps = 100           # Number of denoising steps

guidance_scale = 7.5                # Scale for classifier-free guidance

generator = torch.manual_seed(0)    # Seed generator to create the inital latent noise

batch_size = len(prompt)


text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]


max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]   


text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

torch_noise = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
)
torch_noise = torch_noise.to(torch_device)
img_tensor = img_tensor.to(torch_device)

print(vec)
alpha = 0.4


latents = (1-alpha)*torch_noise + alpha*img_tensor
latents = latents.to(torch_device)

scheduler.set_timesteps(num_inference_steps)

latents = latents * scheduler.init_noise_sigma

from tqdm.auto import tqdm

scheduler.set_timesteps(num_inference_steps)

for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample

    #save_tensor(latents,basename + "_{}.png".format(t))

save_tensor(latents,basename+".png")

