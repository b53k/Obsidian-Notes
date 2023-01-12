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


## Notes 📝

## Sources 📚
[1]. [Generative Models](https://lilianweng.github.io/posts/2018-10-13-flow-models/#types-of-generative-models)
[2]. [Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)