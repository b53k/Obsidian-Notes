import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('Jiraya.jpg')
img = img.resize(size = (128,128))
current_img = np.asarray(img)/255.0

def forward_diffusion(previous_img, beta, t):
    beta_t = beta[t]
    mean = previous_img * np.sqrt(1.0 - beta_t)
    sigma = np.sqrt(beta_t) # variance = beta
    # Generates samples from N(0,1) of prev img size and scales it to new distribution.
    xt = mean + sigma * np.random.randn(*previous_img.shape) 
    return xt

time_steps = 100
beta_start = 0.0001
beta_end = 0.05
beta = np.linspace(beta_start, beta_end, time_steps)

samples = []

for i in range(time_steps):
    current_img = forward_diffusion(previous_img = current_img, beta = beta, t = i)
    if i%20 == 0 or i == time_steps - 1:
        sample = (current_img.clip(0,1)*255.0).astype(np.uint8) # convert to integer for display
        samples.append(sample)

plt.figure(figsize = (12,5))
for i in range(len(samples)):
    
    plt.subplot(1, len(samples), i+1)
    plt.imshow(samples[i])
    plt.title(f'Timestep: {i*20}')
    plt.axis('off')

plt.show()