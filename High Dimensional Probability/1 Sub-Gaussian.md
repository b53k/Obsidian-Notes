<center> <h1> Sub-Gaussian </h1></center>
<center><span style="color: red">🗓️Date: </span><span style="color: blue">01-12-2023</span></center> 
<center><span style="color: red">🕐Time: </span><span style="color: blue">14:06</span></center><right><span style="color: orange">Bipin Koirala
</span></right><hr>

```toc
	style: number
```
<hr>

## Sub Gaussian Distribution

In probability theory, a #Sub-Gaussian distribution is a probability distribution with strong tail decay. Informally, the tails of a #Sub-Gaussian distribution are dominated by (i.e. decay at least as fast as) the tails of a #Gaussian. This property gives #Sub-Gaussian distribution their name. 

![[Pasted image 20230112142842.png|450]]

A random variable $X \in \mathbb{R}$ is said to be #Sub-Gaussian with variance proxy $\sigma^2$ if:
* $\mathbb{E}[X] = 0$ (centered Gaussian) 
* $\mathbb{E}[\mathcal{e}^{t X}] \leq \text{exp}(\large \frac{t^2 \sigma^2}{2})\ \ \forall \ t \in \mathbb{R}$        i.e. M.G.F of $X \leq$ M.G.F of $\mathcal{N}(0, \sigma^2)$ 

> 👉 Note: $\mathbb{E}[X^n] = \large \frac{d^n}{dt^n}M_X(t)|_{t=0}$  where $M_X(t)$ is M.G.F

Equivalently, $X$ is called a #Sub-Gaussian if there are any positive constant $c$ such that$$\mathbb{P}(|X|\geq t)\leq 2 \text{exp}\Big(\frac{-t^2}{c^2}\Big)\ \ \ \ \forall \ \ t \geq 0$$
### Properties:

* $X \sim SG(\sigma^2) \implies \mathbb{E}[X]=0$
* $X \sim SG(\sigma^2); c\in\mathbb{R} \implies cX\sim SG(c^2 \sigma^2)$
* $X \sim SG(\sigma^2) \implies \begin{cases} \mathbb{P}(X\geq t)\leq \text{exp}(\frac{-t^2}{2\sigma^2})\\ \mathbb{P}(X\geq t)\leq 2 \text{exp}(\frac{-t^2}{2\sigma^2})\end{cases}$
*Proof:*
$$\begin{align}
\mathbb{P}(X \geq t) &= \mathbb{P}(e^{\lambda X} \geq e^{\lambda t}) \ \ \ \ \ \text{Monotone Nature of }e\\
&\leq \frac{\mathbb{E}(e^{\lambda X})}{e^{\lambda t}} \ \ \ \ \ \ \ \ \ \ \text{Markov's Inequality}\\
&\leq \large \frac{e^{\frac{\lambda^2 \sigma^2}{2}}}{e^{\lambda t}} \ \ \ \ \ \ \ \ \ \ \ \small X \sim SG(\sigma^2)\\
&= \text{exp}(-\lambda t + \frac{\lambda^2 \sigma^2}{2}) \\
&= \text{exp}\Big(-\sup_{\lambda > 0}\big[\lambda t - \frac{\lambda^2 \sigma^2}{2}\big]\Big)\\
\therefore \ \ \ \mathbb{P}(X \geq t) \ \ \ &\leq \text{exp}\Big(\frac{-t^2}{2\sigma^2}\Big) 

\end{align}$$

### Examples:
👉 (1) $X \sim \mathcal{N}(0, \sigma^2) \implies X \sim SG(\sigma^2)$

👉 (2) $X = \begin{cases} +1, & \text{w.p. } \frac{1}{2}\\ -1, & \text{w.p. } \frac{1}{2} \end{cases}$      *Rademacher R.V. or symmetric Bernoulli R.V* $\implies X \sim SG(1)$
Proof:
$$\begin{align}
\mathbb{E}[\mathcal{e}^{tX}] &= \frac{e^t+e^{-t}}{2} = cosh(t)\\
&= \frac{1}{2}(1 + \frac{t}{1!} + \frac{t^2}{2!}+.....+ 1-\frac{t}{1!}+\frac{t^2}{2!}+...)\\
&= 1 + \frac{t^2}{2!} + \frac{t^4}{4!} + ....\\
&\leq \large e^{\frac{t^2}{2}}
\end{align}$$
👉 (3) $a \leq X \leq b$; $\mathbb{E}[X] = 0 \implies X\sim SG\big(\frac{(b-a)^2}{4}\big)$  Hoeffding

👉 (4) $X\sim SG(\sigma^2) \implies ||X||_{LP} := (\mathbb{E}|X|^p)^{1/p} = \mathbb{E}^{1/p}|X|^p \lesssim \sigma \sqrt{p}; \ \ \ \ p \geq 1$

👉 (5) $X_1, X_2, ......, X_n$ are independent R.V. and $X_i \sim SG(\sigma_i^2) \implies \sum_{i=1}^n X_i \sim SG\Big(\sum_{i=1}^n(\sigma_i^2)\Big)$ 
Proof:
$$\begin{align}
\mathbb{E}[e^{\lambda(X_1 + X_2 + ....+X_n)}] &= \prod_{i=1}^n \mathbb{E}[e^{\lambda X_i}]\\
&\leq \prod_{i=1}^n \text{exp}\Big(\frac{\lambda^2 \sigma^2}{2}\Big)\\
&= \text{exp}\Big(\frac{\lambda^2}{2} \sum_{i=1}^n \sigma_i^2 \Big)\\
\therefore \ \ \ \mathbb{P}(X_1+X_2+...+X_n \geq t) &\leq \text{exp}\Big(\frac{-t^2}{2(\sigma_1^2 + \sigma_2^2+....+\sigma_n^2)  }\Big)
\end{align}$$
## Hoeffding Inequality

Central Limit Theorem (C.L.T) proves concentration inequality directly comparing to $\mathcal{N}(0,1)$. However, #Hoeffding inequality does so directly without comparing to $\mathcal{N}(0,1)$.

Let $X_1, X_2, ...., X_N$ be independent Symmetric Bernoulli R.V. i.e.$\mathbb{P}(X_i = 1) = \mathbb{P}(X_i = -1) = \frac{1}{2}$.  Then;$$\mathbb{P}\Big(\frac{1}{\sqrt N}\sum_{i=1}^NX_i \geq t\Big) \leq e^{-t^2/2}$$
Proof:
>First take $\lambda \geq 0$ multiply it to both sides of probability and take exponential followed by Markov's inequality.
>$$\begin{align}
\mathbb{P}\Big(\sum_{i=1}^NX_i \geq t\sqrt N\Big) &= \mathbb{P}\Big(\lambda \sum_{i=1}^NX_i \geq \lambda t\sqrt N\Big) \\
&= \mathbb{P}\Big(\text{exp}(\lambda \sum_{i=1}^NX_i) \geq \text{exp}(\lambda t\sqrt N)\Big) \\
&\leq \frac{\mathbb{E}[\text{exp}(\lambda \sum_{i=1}^NX_i)]}{\text{exp}(\lambda t\sqrt N)} \\
&= e^{-\lambda t \sqrt N} \ \  \mathbb{E}\Big[\prod_{i=1}^N e^{\lambda X_i}\Big] \\
&= e^{-\lambda t \sqrt N}\ \ \prod_{i=1}^N \mathbb{E}[e^{\lambda X_i}] \\
&= e^{-\lambda t \sqrt N}\ \ \prod_{i=1}^N \mathbb{E}[e^{\lambda X}] \\
&= e^{-\lambda t \sqrt N}\ \  \big(\mathbb{E}[e^{\lambda X}]\big)^N = e^{-\lambda t \sqrt N}\ \  \big(cosh(\lambda))^N \\
&\leq e^{-\lambda t \sqrt N}\ \  \big(e^{\lambda^2/2}\big)^N \\
&= e^{-\lambda t \sqrt N + \lambda^2 N/2} \ \ \ \ \big(\text{let; } -\lambda \sqrt N = \mu\big)\\
&= e^{\large-t\mu + \mu^2/2} \\
&= e^{-t^2/2} \ \ \big(\text{supremum over }\mu \big)
\end{align}$$

In a general case, let $\large X_1, X_2, ..., X_n$ be independent R.V.s and $\large a_j \leq X_j \leq b_j \ ;\ \ \forall j = 1,...,n$ and let $\large S_n :=\sum_{j}^n X_j$ then $$\large \mathbb{P}(|S_n - \mathbb{E}[S_n]| \geq t) \leq \text{exp }\bigg(\frac{-2t^2}{\sum_{j=1}^n(b_j - a_j)^2}\bigg)$$
$\rightarrow X_j - \mathbb{E}[X_j] \sim SG(\frac{(b_j-a_j)^2}{4})$ 

## Hoeffding Lemma


