<center> <h1> Cyclical Monotonicity and Kantorovich Problem </h1></center>
<center><span style="color: red">🗓️Date: </span><span style="color: blue">01-11-2023</span></center> 
<center><span style="color: red">🕐Time: </span><span style="color: blue">14:21</span></center><right><span style="color: orange">Bipin Koirala
</span></right><hr>

```toc
	style: number
```
<hr>

## Continuous Problem in Higher Dimension

Similar to Monge's Formulation in 1-D, in $\mathbb{R}^N$ we have;

Goal: Find $T(x)$ for $$\min \frac{1}{2} \int_\mathbb{R} (x - T(x))^2 \ f(x) dx\ \ \ \ \ \text{s.t} \ \ \ \int_{T^{-1}(A)}f(x)dx = \int_A g(y)\ dy\ \ \forall \ A \subset \mathbb{R}^N$$with the change of variables; $y = T(x)$: 
$$\int_{T^{-1}(A)}f(x)dx = \int_{T^{-1}(A)} g(T(x))\ \big|\nabla T(x)\big|\ dx$$
Here; $|\nabla T(x)|$ is the determinant of the #Jacobian. 
$$\therefore f(x) = g(T(x)) \ \big|\nabla T(x)\big|$$
At this stage, assume we have an optimal map $T$ already. Choose, some $\large x_1, x_2, ..., x_N \in X$ and let $\large y_i = T(x_i)$. Let, $E_i$ be a ball centered at $\large x_i$ s.t. $\large \int_{E_i} f(x) \ dx = \large \epsilon$ and let $F_i = T(E_i)$. Below is a visualization of this scenario.

![[Pasted image 20230111164752.png]]

Let's create a new map, $\widetilde T(x)$ that is measure-preserving and we want the following:$$\begin{align}
\widetilde T(x_i) &= y_{i+1} \\
\widetilde T(E_i) &= F_{i+1} \\
\widetilde T(x) = T(x) \ \ &\text{if}\ \ x \notin \cup_{i=1}^N E_i
\end{align}$$Note: the above transformation is cyclic i.e. $E_1 \to F_2$ and $E_3 \to F_1$.

Now, if $T$ is optimal, we have:$$\begin{align}
\frac{1}{2} \int_{\mathbb{R}} (x-T(x))^2\ f(x) dx &\leq \frac{1}{2} \int_{\mathbb{R}} (x-\widetilde{T}(x))^2\ f(x) dx \\
\Rightarrow \frac{1}{\epsilon} \sum_{i=1}^N \int_{E_i} x(\widetilde T(x) &- T(x))\ f(x)\ dx \leq 0
\end{align}$$Above expression is similar to 1-D case. 
As $\epsilon \to 0$: $x \to x_i, \widetilde T(x_i) \to y_{i+1}, T(x) \to y_i$ and integrating $f(x)$ over $E_i$ gives $\large \epsilon$, so$$\sum_{i=1}^N x_i(y_{i+1} - y_i) \leq 0$$This condition is called #Cyclical-Monotonicity.

### Example:
When $N = 2$, from #Cyclical-Monotonicity we have:$$\begin{align}
x_1(y_2 - y_1) + x_2(y_1 -y_2) &\leq 0 \\
(x_2 - x_1)\cdot(y_2 - y_1) &\geq 0
\end{align}$$In other words, upon transformation the vectors when super-imposed, make an acute angle.

![[Pasted image 20230111173141.png]]

This condition holds true for all $N$ and restricts our ability to 'twist' mass. This is a stronger condition than being irrotational.

## Gradient of Convex Function

>**💡Theorem (Rockafellar):**
   A cyclically monotone map can be expressed as the gradient of a convex function.

From the optimality condition; we can write $T(x) = \nabla u(x)$ where $u$ is a convex function. Now, by the conservation of mass we have already established that $f(x) = g(T(x))  \big|\nabla T(x)\big|$ 
Now, $$\begin{align}
\big|\nabla T(x)\big| &= \frac{f(x)}{g(T(x))} \\
\big| \nabla^2u(x)\big| &= \frac{f(x)}{g(T(x))}
\end{align}$$ Above equation is called the #Monge-Ampere equation and $|\nabla^2u(x)|$ is the determinant of a Hessian.

## Kantorovich Problem

We have already seen the objective function in #Kantorovich formulation.
$$\inf \int_{X \times Y} c(x,y)\ d\pi(x,y) \big| \pi \in \Pi(\mu, \nu)$$
Here, $\Pi(\mu, \nu)$ is the set of measures whose marginals on $X$ and $Y$ are $\mu$ and $\nu$ respectively.

#Kantorovich formulation is feasible under the following conditions:
* $\Pi(\mu, \nu)$ is feasible if there is a mass balance
* As long as the cost function is bounded below and $X$, $Y$ are bounded then there exists an infimum that is finite.

### Is there a minimizer?

We need a compactness argument.
>**💡Theorem (Weierstrass):**
> If $f:U \to \mathbb{R}$ is continuous and $U$ is compact then $f$ attains a minimum on $U$. 

So we need to show that $\large c(x,y)\ d\pi(x,y)$ is continuous and $\Pi(\mu, \nu)$ is compact in order for the #Kantorovich formula to contain infimum.

>💡**Theorem**
> Suppose $X,Y \subseteq \mathbb{R}^N$ are compact and that $\large c(x,y)$ is continuous. Under this assumptions #Kantorovich problem contains minimum.

**Proof:**
Before we talk about compactness we need a notion of convergence. We identify $U = \Pi(\mu, \nu)$ and without loss of generality these are probability measures. Furthermore, a sequence of measures converges $\large \gamma_n \to \gamma$ if $$\int_{X \times Y} g(x,y)d\gamma_n(x,y) \to \int_{X\times Y}g(x,y)d\gamma \ \ \ \ \forall \ g \in C^0(X \times Y)$$
>Is the set $\large \Pi(\mu,\nu)$ compact?
>Choose any sequence $\large \pi_n \in \Pi(\mu,\nu)$. We need to extract a convergent sub-sequence.
>Since, $\large \pi_n$ are probability measures, we can extract a sub-sequence.
>$\large \pi_{n_k} \to \pi$, which is also a probability measure.

🔔 Need to check: Does $\pi \in \Pi(\mu, \nu)$ ?
`Check marginals` Choose any $g \in C(X)$ and $$\begin{align}
\int_{X \times Y} g(x)\ d\pi(x,y) &= \lim_{k \to \infty} \int_{X \times Y}g(x)\ d\pi_{n_k}(x,y) \ \ \ \ \because \ \pi_{n_k} \to \pi \\
&= \int_X g(x)\ d\mu(x)\ \ \ \ \ \because \ \pi_{n_k} \in \Pi(\mu,\nu), \text{ so is the marginal over }X\text{ is }\mu
\end{align}$$With the same argument, the marginal of $\pi$ over $Y$ is $\nu$.$$\therefore \ \ \pi \in \Pi(\mu, \nu) \ \ \ \ \ \text{ i.e. }\Pi\text{ is compact}$$
> Is the value of $f$ continuous?
> Let, $\large f(\pi) = \int_{X \times Y}c(x,y)\ d\pi(x,y)\ \ \ \ \forall \pi \in U$
> Choose any $\pi_n \in \Pi(\mu, \nu)$ s.t. $\pi_n \to \pi \in \Pi(\mu,\nu)$. > 

🔔 Need to show: $$f(\pi_n) \to f(\pi)$$
Here;$$\begin{align}
f(\pi_n) &= \int_{X \times Y} c(x,y)\ d\pi_n(x,y) \\
&= \int_{X \times Y}c(x,y)\ d\pi(x,y) \ \ \ \because \pi_n \to \pi \ \ \&\ \ c\ \text{is continuous} \\
&= f(\pi)
\end{align}$$ $\therefore \ f$ is continuous. 

Therefore, from Weierstrass theorem we can say that the #Kantorovich problem has a minimizer.

<hr>
📝 #Kantorovich<span style="color: red"> problem seeks to minimize a real-valued, continuous function over a compact set. It admits a minimizer.</span><hr>

