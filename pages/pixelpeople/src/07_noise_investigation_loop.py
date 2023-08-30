from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler
from PIL import Image
import torch
import numpy as np

torch.manual_seed(1)

# scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")
scheduler.set_timesteps(200)

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size)).to("cuda")
noise2 = torch.zeros((1, 3, sample_size, sample_size)).to("cuda")

# print(noise.shape)
# print(torch.mean(noise))

for idx_ratio, ratio in enumerate(np.linspace(0, 1, 100000000)):
    print(idx_ratio)
    input = noise + ratio * noise2
    
    for t in scheduler.timesteps:
    
        with torch.no_grad():
            noisy_residual = model(input, t).sample
            prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
    
            input = prev_noisy_sample
    
    # save image
    image = (input / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = Image.fromarray((image * 255).round().astype("uint8"))
    image.save('noise_{}.png'.format(idx_ratio))
    



