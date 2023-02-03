<center> <h1> Diffusion Models </h1></center>
<center><span style="color: red">🗓️Date: </span><span style="color: blue">12-30-2022</span></center> 
<center><span style="color: red">🕐Time: </span><span style="color: blue">20:39</span></center><right><span style="color: orange">Bipin Koirala
</span></right><hr>

```toc
	style: number
```
<hr>

## Preliminary 🔍

### Discriminant Models Vs Generative Models

Most of the time we are concerned with predicting a label given an instance of a dataset. Statistical models that operate on this notion are #discriminant models. Unlike these models there exists a different paradigm where we want to learn the joint probability distribution between the data and its label (holds true even if the data has no label). These models are called #generative models and can generate new data instances. 

For a set of data instances $X$ and set of labels $Y$, we have the following:

$$\text{Discriminant Model}: \mathbb{P}(label|data) = \mathbb{P}(Y|X) $$
	$$\text{Generative Model}: \mathbb{P}(data,label) = \mathbb{P}(X,Y)\ \  \text{or}\ \ \mathbb{P}(X)\ \ \text{if no labels}$$
For example, a discriminant model could tell a picture of a bird from a horse but generative model could generate a new pictures of animals that look like real animals. 
For data $\large x \in \mathcal{D}$, discriminant model aims to draw a boundary in $\mathcal{D}$ whereas generative model aims to model how a data is placed throughout the space $\mathcal{D}$.

![[Pasted image 20221231151138.png]]

Some types of generative models are:
* Gaussian Mixture Model (GMM)
* Bayesian Network (Naive Bayes, Auto-regressive models)
* Boltzmann Machine
* Generative Adversarial Network (GAN)
* Variational Auto-encoder (VAE)
* Diffusion Models
* Energy Based Models (EBM) etc.

Below are summary of few Generative-Models[1].

#### GANs

These are primarily used to replicate real-world contents such as images, languages, and musics. Two agents, #generator and #discriminator play a min-max game to attain equilibrium. It is difficult to train GAN because of training instability and failure to converge. #Wasserstein GAN (WGAN) provides improved results over traditional GAN.

#### VAE

#Auto-encoder is a neural network which attempts to reconstruct a given data via compressing the input in the process so as to discover a latent/sparse representation. This latent representation of data can later be used in various downstream tasks.

![[Pasted image 20221231155054.png]]

The latent space in auto-encoders are primarily discrete and does not allow for an easy interpolation. The generative part or the decoder of the auto-encoder works by randomly sampling points from the latent space and it can be challenging if the latent space is itself discontinuous or has gaps.

#Variational-Auto-Encoder (VAE) solves this issue because its latent space is continuous in nature which makes VAE powerful in generating new data instances. Instead of generating a latent vector $\large\mathcal{z} \in \mathbb{R}^N$, VAE generates two vectors i.e. mean ($\mu$) vector and standard deviation ($\sigma$) vector followed by decoder sampling from this distribution. 

![[Pasted image 20221231164732.png]]

________________
**Some Remarks**:  
> $\color{red}\mathcal{p}(z) \tag{1}$: <span style="color: orange">Prior</span>
> $\color{red}\mathcal{p}(x|z)$: <span style="color: orange">Likelihood (Generation)</span>
> $\color{red}\mathcal{p}(x,z) = \mathcal{p}(x|z)\mathcal{p}(z)$: <span style="color: orange">Joint Distribution</span>
> $\color{red}\mathcal{p}(x)$: <span style="color: orange">Marginal</span>
> $\color{red}\mathcal{p}(z|x)$: <span style="color: orange">Posterior (Inference)</span>
> 
> For Generation: sample $\color{red}\large z \sim p(z)$ then sample $\color{red}\large x \sim p_\theta(x|z)$
> For Inference: sample $\color{red}\large x \sim p(x)$ then sample $\color{red}\large z \sim q_\phi(z|x)$ which acts as a proxy for $\color{red}\large\mathcal{p}_\theta(z|x)$
_____________

Ideally we would like to get; $$\large \mathcal{p}_\theta(x) = \int \mathcal{p}_\theta(x|z)\ \mathcal{p}_\theta(z)\ dz$$However the integral is intractable because it needs to be evaluate over all possible values of $\large z \sim \mathcal{p}_\theta(z)$. These kinds of latent variable models are often trained with Maximum Likelihood Estimation (M.L.E)$$\theta_{MLE} = \arg \max_{\theta} \sum_{i=1}^N log\ \mathcal{p}_\theta(x_i)$$Since, $\large \mathcal{p}_\theta(x)$ does not have a closed form solution we find its lower bound called the #Variational-Lower-Bound(V.L.E) or #Evidence-Lower-Bound (E.L.B.O)$$\large\begin{align}
log\ p_\theta(x) &= \mathbb{E}_{z \sim q_\theta(z|x)} [\ \text{log}\ p_\theta(x)\ ]\\
&= \mathbb{E}_{z \sim q_\theta(z|x)} \Big[\ \text{log}\ \frac{p_\theta(x,z)}{p_\theta(z|x)}\ \Big] \\
&= \mathbb{E}_{z \sim q_\theta(z|x)} \Big[\ \text{log}\ \frac{p_\theta(x,z)}{q_\phi(z|x)}\ \frac{q_\phi(z|x)}{p_\theta(z|x)}\ \Big] \\
&= \mathbb{E}_{z \sim q_\theta(z|x)} \Big[\ \text{log}\ \frac{p_\theta(x,z)}{q_\phi(z|x)}\ \Big] + \mathbb{E}_{z \sim q_\theta(z|x)} \Big[\ \text{log}\ \frac{q_\phi(z|x)}{p_\theta(z|x)}\ \Big]\\
&= L_{\theta,\phi}(x) + D_{KL}\Big(q_\phi(z|x)\ \ ||\ \ p_\theta(z|x)\Big)
\end{align}$$Since $D_{KL}\Big(q_\phi(z|x)\ \ ||\ \ p_\theta(z|x)\Big) \geq 0$;$$\large\text{log}\ p_\theta(x) \geq L_{\theta,\phi}(x)$$ Therefore, we optimize (maximize) over this #Variational-Lower-Bound.

##### Optimizing Variational Lower Bound

One possibility is to sample $\large x_i$ and get the best $\large q_\phi(z|x_i)$ using multiple gradient steps in $\large \phi$. Then gradient ascend in $\large \theta$. However, this is an expensive inference process because we would need to learn individual distribution for each $\large x_i$.

Instead we #Amortize the inference costs by learning an inference neural network that presumes $\large q_\phi(z|x) \sim \mathcal{N}(z\ ;\ \mu(x), I\sigma(x))$.

👉 Stochastic Gradient Optimization of #Variational-Lower-Bound
The want to solve the following optimization problem$$\large\max_{\theta,\ \phi} \sum_{x_i \ \in \ \mathcal{D}}L_{\theta, \phi}(x_i)$$ Computing $\large \nabla_{\theta, \phi}\ L_{\theta, \phi}(x_i)$ is intractable as we would need to sample across entire $\large q_{\phi}(z|x)$ and take its gradient. However, there are unbiased estimators for this. 

Recall, $\large q_\phi(z|x) \sim \mathcal{N}(z\ ;\ \mu(x), I\sigma(x)) = \mu(x) + \sigma(x)\epsilon$ where $\large \epsilon \sim \mathcal{N}(0, I)$

Now,$$\begin{align}
\large L_{\theta, \phi}(x) &= \mathbb{E}_{z \sim q_\phi(z|x)}\big[log\ p_\theta(x,z) - log\ q_\phi(z|x)\big] \\
\large \hat{L}_{\theta, \phi}(x) &= \mathbb{E}_{\epsilon \sim p(\epsilon)}\big[log\ p_\theta(x,z) - log\ q_\phi(z|x)\big] \tag{unbiased estimator}
\end{align}$$
🔥Note: $\large \mathbb{E}_{\epsilon \sim p(\epsilon)}\hat{L}_{\theta, \phi}(x) = L_{\theta, \phi}(x)$ so now need to compute $\large \nabla_{\phi}\hat{L}_{\theta, \phi}(x)$

Neural Network architecture of VAE showing how decoder samples from latent vectors.

![[Pasted image 20221231164816.png]]


#### Flow-Models

At its core, #Flow-Models make use of **Normalizing Flow (NF)**, a technique used to build a complex probability distributions by transforming simple distributions.
Let, $\large{z}$ ~ $\mathbb{P}_\theta(\large z)$ and $\large z \in Z$ be a probability distribution, generally taken something simple like $\mathcal{N}(\large z;\mu,\sigma)$. The key idea here is to transform this simple distribution to a complex distribution $\large x = \large f (\large z)$, where $\large f$ is a bijective map. We formulate $\large f$ as a composition of sequence of invertible transformations. $$\large x = \large f_K \circ \large f_{K-1} \circ \ ...\ \large f_2 \circ \large f_1(\large z)$$
Now, $$\large \int p_\theta(x) dx = \large \int p_\theta(z)dz = 1$$$$\large p_\theta(x)dx = \large p_\theta(f^{-1}(x))dz$$
$$\large p_\theta(x) = \large p_\theta(f^{-1}(x)) \Big|\frac{dz}{dx}\Big| = \large p_\theta(f^{-1}(x)) \Big|\frac{df^{-1}}{dx}\Big|$$
Multivariable formulation of the above expression gives us;$$\large p_\theta(\textbf{x}) = \large p_\theta(f^{-1}(\textbf{x})) \Big| det \Big(\frac{df^{-1}}{d\textbf{x}}\Big)\Big|$$![[Pasted image 20230101224929.png]]

## Diffusion Models

Non-equilibrium thermodynamics deals with the study of time-dependent thermodynamic systems, irreversible transformations and open systems. #Diffusion-Models are heavily inspired by non-equilibrium thermodynamics. They define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise. Unlike VAE or Flow-Models, diffusion models are learned with a fixed procedure and the latent space has high dimensionality (same as the original data)[2].

>Much of the notes in the followings sections on the #Diffusion-Model are based on <span style="color: magenta"><i>Deep Unsupervised Learning using Non-equilibrium Thermodynamics</i></span>. 

#Diffusion-Model allows us to rapidly learn, sample from, and evaluate probabilities in deep generative models with thousands of layers/time-steps, as well as to compute conditional and posterior probabilities under the learned model. Diffusion process exists for any smooth target distribution, this method can capture data distributions of arbitrary form.

### Algorithm

The goal is to define a forward (or inference) diffusion process which converts any complex data distribution into a simple, tractable, distribution. Then learn a finite-time reversal of this diffusion process which defines the generative model distribution. 

#### Forward Trajectory

👉 Initial data distribution: $\large q(x^{(0)})$

👉 This is gradually converted to a well-behaved (analytically tractable) distribution $\pi(y)$ by repeated application of a Markov diffusion kernel $T_{\pi}(y\ |\ y^{'};\beta)$ where $\beta$ is diffusion rate
$$\begin{align}
\large \pi(y) &= \int T_\pi(y\ |\ y^{'};\beta)\ \ \pi(y^{'})\ \ dy^{'} \tag{1}\\
\large q\big(x^{t}|x^{(t-1)}\big) &= T_\pi\Big( x^{(t)}\ |\ x^{(t-1)};  \beta_t\Big) \tag{2}
\end{align}$$
Here; forward diffusion kernel is $T_\pi\Big( x^{(t)}\ |\ x^{(t-1)};  \beta_t\Big) = \mathcal{N}\Big(x^{(t)}; x^{(t-1)}\sqrt{1-\beta_t}, \ I\beta_t \Big)$ i.e. Gaussian but can be other distribution as well for example a Binomial distribution.

The forward trajectory, corresponding to starting at the initial data distribution and performing $T$ steps of diffusion is given by,$$\large q\Big( x^{(0....T)}\Big) = q\big(x^{(0)}\big) \prod_{t=1}^T q\Big(x^{(t)}\ |\ x^{(t-1)}\Big) \tag{3}$$
The above process allows for the sampling of $\large x_t$ at any arbitrary time step $\large t$ in a closed form using reparameterization trick. $$\large x^{(t)} = \sqrt{1-\beta_t}\ \  \large x^{(t-1)} \ + \ \sqrt{\beta_t} \ \mathcal{N}(0,I)$$
```python
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
	# Generate sample from N(0,1) of prev img size and scale to new distribution.
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
		# convert to integer for display
		sample = (current_img.clip(0,1)*255.0).astype(np.uint8) 
		samples.append(sample)

  
plt.figure(figsize = (12,5))
for i in range(len(samples)):
	plt.subplot(1, len(samples), i+1)
	plt.imshow(samples[i])
	plt.title(f'Timestep: {i*20}')
	plt.axis('off')

plt.show()
```
![[Pasted image 20230116163111.png]]

#### Reverse Trajectory

Generative distribution is trained to describe the above (forward) trajectory, but in the reverse direction,$$\large \begin{align} 
\mathcal{p}\big(x^{(T)}\big) &= \pi\big(x^{(T)}\big) \tag{4} \\
\mathcal{p}\big(x^{(0...T)}\big) &= \mathcal{p}\big(x^{(T)}\big)\ \prod_{t=1}^T\mathcal{p}\Big(x^{(t-1)}\ |\  x^{(t)} \Big) \tag{5}
\end{align}$$
If we take $\large \beta_t$ to be small enough then, $q\big(x^{(t-1)}|x^{(t)}\big)$ will also be a Gaussian distribution. The longer the trajectory the smaller the diffusion rate $\large \beta$ can be made. During learning only the mean and the covariance for a Gaussian diffusion kernel need to be estimated. The functions defining the mean and the covariance of the reverse process are:$$\large f_{\mu}(x^{(t)},t) \ \ \ \& \ \ \ f_\Sigma(x^{(t)},t)$$
Practically speaking, we don't know $\large q\big(x^{t-1}|x^{(t)}\big)$ as it is intractable since the statistical estimate requires computations involving the data distribution. Therefore, we approximate $\large q\big(x^{t-1}|x^{(t)}\big)$ with a parameterized model $\large p_\theta$ (e.g. a neural network). With the parameterized model, we have $$\large \mathcal{p}_\theta\big(x^{(0...T)}\big) = \mathcal{p}_\theta\big(x^{(T)}\big)\ \prod_{t=1}^T\mathcal{p}_\theta\Big(x^{(t-1)}\ |\  x^{(t)} \Big)$$i.e. starting with the pure Gaussian noise, the model learns the joint distribution $\large \mathcal{p}_\theta(x^{(0...T)})$. Conditioning the model on time-step $\large t$, it will learn to predict the Gaussian parameters, $\large f_{\mu}(x^{(t)},t)$ and $\large f_\Sigma(x^{(t)},t)$, for each time-step.

Here;$$\large{p_\theta(x^{(t-1)}|x^{(t)})} = \mathcal{N}\Big(x^{(t-1)}; \ \ \mu_\theta(x^{(t)},t),\ \ \Sigma_\theta(x^{(t)},t)\Big)$$

### Training

The probability the generative model assigns to the data is
$$\large \mathcal{p}(x^{(0)}) = \int p(x^{(0...T)}) \ dx^{(1...T)} \tag{6}$$
This tells us that if we were to calculate $\large p(x^{(0)})$ we need to marginalize over all the possible trajectories to arrive at the initial distribution starting from the noise sample...which is intractable in practice. However, we can maximize a lower bound.

In #Diffusion-Model, the forward process is fixed and only reverse process needs to be trained i.e. only a single network is trained unlike #Variational-Auto-Encoder. 

#Diffusion-Models are trained by finding the reverse Markov transitions that maximize the likelihood of the training data. Similar to #Variational-Auto-Encoder training is based on minimizing the #Variational-Lower-Bound. Therefore, we optimize the negative log-likelihood.$$\large \begin{align}
-\text{log}\ p_\theta(x_0) &\leq -\text{log}\ p_\theta(x_0) + D_{KL}\Big(\ q(x_{1:T}|x_0)\ \ ||\ \ p_\theta(x_{1:T}|x_0)\ \Big) \\
&= -\text{log}\ p_\theta(x_0) + \mathbb{E}_{x_{1:T}\ \sim\ q(x_{1:T}|x_0)}\Big[ \text{log}\ \frac{q(x_{1:T}|x_0)}{p_\theta(x_{1:T}|x_0)} \Big] \\
&= -\text{log}\ p_\theta(x_0) + \mathbb{E}_{x_{1:T}\ \sim\ q(x_{1:T}|x_0)}\Bigg[ \text{log}\ \frac{q(x_{1:T}|x_0)}{\frac{p_\theta(x_{1:T})\ p_\theta(x_0|x_{1:T})}{p_\theta(x_0)}} \Bigg] \\
&= \mathbb{E}_{x_{1:T}\ \sim\ q(x_{1:T}|x_0)}\Big[ \text{log}\ \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:1:T})} \Big]
\end{align}$$
Here, #Variational-Lower-Bound is $\large \mathbb{E}_{x_{1:T}\ \sim \  q(x_{1:T}|x_0)}$ and $$\large -\text{log}\ p_\theta(x_0) \leq \large \mathbb{E}_{x_{1:T}\ \sim \  q(x_{1:T}|x_0)}\Big[ \text{log}\ \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:1:T})} \Big] \tag{7}$$
Now,$$\large \begin{align}
\text{log}\ \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})} &= \text{log}\ \frac{\prod_{t=1}^Tq\ (x_t|x_{t-1})}{p(x_T)\prod_{t=1}^T\ p_\theta(x_{t-1}|x_t)} \\
&= -\text{log}\ (p(x_T)) + \text{log}\ \frac{\prod_{t=1}^Tq\ (x_t|x_{t-1})}{\prod_{t=1}^T\ p_\theta(x_{t-1}|x_t)}\\
&= -\text{log}\ (p(x_T)) + \sum_{t=1}^T\text{log}\ \frac{q(x_t|x_{t-1})}{p_\theta(x_{t-1}|x_t)}\\
&= -\text{log}\ (p(x_T)) + \sum_{t=2}^T\text{log}\ \frac{q(x_t|x_{t-1})}{p_\theta(x_{t-1}|x_t)} + \text{log}\ \frac{q(x_1|x_0)}{p_\theta(x_0|x_1)}\\
&= -\text{log}\ (p(x_T)) + \sum_{t=2}^T\text{log}\ \frac{q(x_t|x_0)\ q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)\ q(x_{t-1}|x_0)} + \text{log}\ \frac{q(x_1|x_0)}{p_\theta(x_0|x_1)}\\
&\because \text{Baye's Rule and conditoning on }x_0\text{ to avoid high variance}\\
&= -\text{log}\ (p(x_T)) + \sum_{t=2}^T\text{log}\ \frac{q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)} + \sum_{t=2}^T\text{log}\ \frac{q(x_t|x_0)}{q(x_{t-1}|x_0)}{}+\text{log}\ \frac{q(x_1|x_0)}{p_\theta(x_0|x_1)}\\
&= -\text{log}\ (p(x_T)) + \sum_{t=2}^T\text{log}\ \frac{q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)}\ +\ \text{log}\ \frac{q(x_T|x_0)}{q(x_1|x_0)} + \text{log}\ \frac{q(x_1|x_0)}{p_\theta(x_0|x_1)}\\
&= \text{log}\ \frac{q(x_T|x_0)}{p(x_T)} + \sum_{t=2}^T\text{log}\ \frac{q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)}\ - \text{log}\ p_\theta(x_0|x_1)\\
&\because\ \text{Firt terms has no learnable parameter --- drop it}\\
\end{align}$$From $(7)$;$$\large \begin{align}
-\text{log}\ p_\theta(x_0) &\leq \large \mathbb{E}_{x_{1:T}\ \sim \  q(x_{1:T}|x_0)}\Big[ \text{log}\ \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:1:T})} \Big] \\
&= \large \mathbb{E}_{x_{1:T}\ \sim \  q(x_{1:T}|x_0)}\Big[ \sum_{t=2}^T\text{log}\ \frac{q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)}\ - \text{log}\ p_\theta(x_0|x_1) \Big]\\
&= \large \mathbb{E}_{x_{1:T}\ \sim \  q(x_{1:T}|x_0)}\Big[ \sum_{t=2}^T\text{log}\ \frac{q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)}\ - \text{log}\ p_\theta(x_0|x_1) \Big]\\
&= \sum_{t=2}^TD_{KL}\Big(q(x_{t-1}|x_t,x_0)\ ||\ p_\theta(x_{t-1}|x_t)\Big) -\text{log}\ p_\theta(x_0|x_1)
\end{align}$$
We can further simplify the first term above because;$$\large
p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t), \beta_t I) \ \ \ \&\ \ \ q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1}; \ \tilde\mu_t(x_t,x_0), \tilde \beta_t I)$$
And,$$\large \begin{align}
\tilde \mu_t(x_t,x_0) &:= \frac{\sqrt{\overline{\alpha}_{t-1}}\ \beta_t}{1 - \overline\alpha_t}\ x_0\ + \ \frac{\sqrt{\alpha_t}(1-\overline{\alpha}_{t-1})}{1-\overline\alpha_t}\ x_t \\
\tilde{\beta_t} &:= \frac{1-\overline{\alpha}_{t-1}}{1-\overline{\alpha}_t}\beta_t
\end{align}$$where $\large \alpha_t = 1 - \beta_t$ and $\large \overline{\alpha}_t = \prod_{t=s}^t \alpha_s$ and further simplification as shown in <span style="color: magenta"><i>Denoising Diffusion Probabilistic Models [2020]</i></span> yields the loss function$$\large \color{red}L_t \sim || \epsilon - \epsilon_\theta(x_t, t)||_2^2$$where, $\large \epsilon := \mathcal{N}(0, I)$ and $\large (x_t,t)  = \big(\sqrt{\overline{\alpha}_t}\ x_0 + \sqrt{1 - \overline{\alpha}_t}\ \epsilon,\ t\big)$

## Notes 📝

## Sources 📚
[1]. [Generative Models](https://lilianweng.github.io/posts/2018-10-13-flow-models/#types-of-generative-models)
[2]. [Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)