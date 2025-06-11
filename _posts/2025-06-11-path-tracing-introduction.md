# Path Tracing Global Illumination - Introduction
## Introduction

When we talk about photorealistic rendering, there's one technique that consistently stands out − path tracing. It’s elegant in theory, simple at its core, and yet incredibly powerful. Unlike rasterization, which fakes the appearance of realism, path tracing simulates the actual physics of light: how it bounces, scatters, reflects, and refracts through the world.

But that physical accuracy comes at a cost - noise, performance, and complexity. So why has path tracing become the backbone of modern offline rendering, and why are real-time engines slowly shifting toward it?

In this post, we’ll dive into the fundamentals of path tracing: starting from outlining the problem and finishing with the solution that is used in the entire industry. Whether you're building your own renderer, studying graphics, or just curious about how light turns into pixels − this is the place to start.

---
## The problem

As every other technique or algorithm, path tracing also solves some problem. In order to smoothly dive into the topic, let's firstly describe the it.
### Rendering: The Light Transport Equation

At the heart of photorealistic rendering lies a deceptively simple question:
How much light reaches the camera from a given direction?

To answer this, we need to simulate the transport of light throughout a scene - how it bounces, scatters, and is absorbed or emitted. This process is governed by the **Light Transport Equation** (LTE), first formalized by James Kajiya in 1986:

$$
L_o(x, \omega_o) = L_e(x, \omega_o) + \int_{\Omega} f_r(x, \omega_i, \omega_o) \, L_i(x, \omega_i) \, (\omega_i \cdot n) \, d\omega_i
$$

Let's briefly walk through all of its terms:
- $$L_e(x, \omega_o)$$ - emitted radiance from a point
- $f_r(x, \omega_i, \omega_o)$ - BRDF for a given point
- $L_i(x, \omega_i)$ - incoming radiance
- $\omega_i \cdot n$ - dot product between a surface normal and incoming direction

Seems simple enough, right?

It would be simple and PBR rendering could be much easier topic, if not this: $\int_{\Omega}$ . What does it mean for us, is that all radiance leaving a point equals to emitted radiance **and** an **integral** over all directions on the hemisphere above the point.

So, now it doesn't seem to be that easy to solve, especially that we literally cannot solve it analytically - thus, we can't get the precise result of that equation for most of the cases.

Yet, there's another problem

--- 
### The recursiveness of LTE

If you look at the LTE once more, you may notice an equation for outgoing radiance at a point has an incoming radiance term. The thing is that incoming radiance is outgoing radiance from another point - **the equation is recursive**. It makes things even harder, especially that beams of light may bounce practically infinitely.

---
### Summary

Let's wrap up everything we have so far:

- Rendering is essentially solving a Light Transport Equation
- The equation is comparably simple
- Yet it is effectively impossible to solve analytically due to integral
- Furthermore - the equation is recursive, making it even more complex to solve

Having all of that, let's see what we can do to solve it!

---
## Possible solution
### Riemann sum

The first thing that may come to readers' mind is using a numerical solution called **Riemann Sum**. Let's briefly talk about it and how it may help us.

Instead of solving the integral analytically − so taking into account practically infinite amount of possible directions on hemisphere, we could subdivide said sphere into a set of discrete pieces and solving each of them independently, making solving of the equation possible.

In the case of the **light transport equation**, the integral over the hemisphere can be written as:
$$\int_{\Omega} f_r(x, \omega_i, \omega_o) \, L_i(x, \omega_i) \, (\omega_i \cdot n) \, d\omega_i$$
To approximate this integral using a **Riemann sum**, we would discretize the hemisphere  $\Omega$  into $N$ uniformly spaced directions $\omega_i^{(k)}$, and compute:

$$
\int_{\Omega} f_r(x, \omega_i, \omega_o) \, L_i(x, \omega_i) \, (\omega_i \cdot n) \, d\omega_i \approx \sum_{k=1}^{N} f_r(x, \omega_i^{(k)}, \omega_o) \, L_i(x, \omega_i^{(k)}) \, (\omega_i^{(k)} \cdot n) \, \Delta \omega
$$

Where:
- $\Delta \omega = \frac{2\pi}{N}$ if directions are uniformly distributed over the hemisphere,
- $N$ is the number of discrete sample directions,
- $\omega_i^{(k)}$ are the direction samples.

In essence, we **sample the hemisphere deterministically**, evaluate the function at each sample, and multiply by the solid angle element $\Delta \omega$.

---
### Why this may not work

While this method works for low-dimensional, well-behaved integrals, it **does not scale well** for rendering. There are several reasons:

- **High dimensionality**: Each additional bounce adds another integral - over the hemisphere at the new surface point.
- **No adaptivity**: Riemann sums treat all directions equally, even though most contribute very little to the final radiance.
- **Exponential cost**: For $k$ bounces and $N$ directions, the total number of evaluations becomes $N^k$, which grows exponentially.

This is why rendering engines rely on **stochastic methods**, which allows us to approximate the integral using just a few random but carefully chosen directions.

---
## Monte-Carlo Path Tracing
### Introduction

Before we dive into Monte Carlo rendering, we need to take a quick - and very useful - detour into the world of **probability and statistics**. Don’t worry, we’ll build this up slowly, and you’ll see exactly how it connects to rendering in a moment.

Let’s begin with a fundamental concept in probability: the **expected value**. You can think of it as the "average outcome" of a random experiment. If you flip a fair coin, you expect heads 50% of the time − that’s an expected value. But in math, we can make this more general and powerful.

Imagine you have a function $f(x)$ defined over some domain $\mathcal{D}$ − for example, over all directions on a hemisphere above a surface point. This function might represent how much light comes from direction $x$, or how bright a pixel would be if a ray went in that direction.

Now, suppose you're able to randomly sample points $x$ from the domain according to a **probability density function (PDF)** $p(x)$, which tells you how likely it is to pick any particular $x$. The expected value (i.e., the average result you'd get over many trials) of $f(x)$ is given by:

$$
\mathbb{E}[f(X)] = \int_{\mathcal{D}} f(x) \, p(x) \, dx
$$

This equation tells us: “If we sample $x$ randomly from the distribution $p(x)$, then on average, $f(x)$ will behave like this.”

In the special case where $p(x)$ is **uniform** − meaning every point in the domain is equally likely − the expected value becomes a simple average. But in most practical scenarios, especially in rendering, the distribution isn’t uniform at all. Some directions may contribute far more to the final lighting than others, and sampling them more often is beneficial − more on that later.

In rendering terms, $f(x)$ might represent something like:

- the amount of light arriving from direction $x$ (incoming radiance),
- the reflected light due to a specific BRDF,
- or the entire contribution to a pixel's brightness.

And our goal is to compute its average over the hemisphere of directions − i.e., **to integrate it**. That’s where Monte Carlo comes in.

---
### Monte Carlo Integration

**Now, let's talk about the key aspect and the foundation of the path tracing - Monte Carlo Integration.**

Let’s now suppose you’re faced with the problem of computing an integral like this:

$$
I = \int_{\mathcal{D}} f(x) \, dx
$$

This type of integral appears everywhere in physically based rendering − especially when we want to compute how much light is reflected at a point by summing over all incoming directions.

But here’s the challenge: in high-dimensional spaces, or in complex scenes, evaluating such an integral **exactly** is either impossibly slow or outright intractable. Traditional numerical techniques like Riemann sums or trapezoidal rules require dividing the domain into many small parts − which leads to an **exponential explosion** of work as the dimension increases. This is commonly referred to as the **curse of dimensionality**.

So how can we approximate this integral more cleverly?

Here’s the key idea: instead of trying to evaluate the function $f(x)$ everywhere, we choose a few points $x_i$ **randomly**, and weigh their contributions appropriately. If we can sample from a probability distribution $p(x)$ over the same domain $\mathcal{D}$, we can rewrite the integral as:

$$
I = \int_{\mathcal{D}} \frac{f(x)}{p(x)} \, p(x) \, dx = \mathbb{E}\left[\frac{f(X)}{p(X)}\right]
$$

Now, we’re back in expected value territory! This means we can estimate the integral by **sampling**:

$$
I \approx \frac{1}{N} \sum_{i=1}^{N} \frac{f(x_i)}{p(x_i)}
$$

Each term in the sum represents one random sample: we evaluate the function $f$ at that point, and divide it by how likely we were to pick it (the PDF value). This gives us an **unbiased estimate** of the integral: noisy, yes, but statistically correct − it will converge to the true value as the number of samples $N$ increases.

This is the foundation of **Monte Carlo integration**: an incredibly general and powerful method that works even when the function $f(x)$ is complicated, discontinuous, or defined over a high-dimensional domain.

And now here comes the best part: this is exactly what we need in rendering. When a pixel’s color depends on complex combinations of light paths, materials, occlusion, and geometry, we don’t want to brute-force everything. We want a method that can sample light paths randomly and still give us meaningful results.

There’s one important concept we haven’t touched on yet − **variance**. In Monte Carlo integration, each random sample gives us an unbiased estimate of the result, but those samples may vary wildly depending on the function we’re evaluating and how well we’re sampling it.

**Variance** is a statistical measure of how much our samples “jump around” from the expected value. If our samples are consistent, variance is low. If they’re chaotic, variance is high − and this shows up as **visual noise** in rendered images.

In fact, the grainy appearance in a low-sample path-traced image is nothing but a direct visual manifestation of high variance in our Monte Carlo estimates. The more variance we have, the noisier the image looks.

Reducing variance − without introducing bias − is one of the central challenges in physically based rendering. We’ll revisit this concept again when we talk about importance sampling and other variance reduction strategies.

In the next section, we’ll apply this technique directly to the light transport equation and see how it gives birth to **path tracing**.

---
### Solving LTE using Monte Carlo Integration

Now that we understand how Monte Carlo integration can approximate complicated integrals, let’s return to our core problem: the **light transport equation**.

Recall the rendering equation at a single surface point $x$:

$$
L_o(x, \omega_o) = L_e(x, \omega_o) + \int_{\Omega} f_r(x, \omega_i, \omega_o) \, L_i(x, \omega_i) \, (\omega_i \cdot n) \, d\omega_i
$$

This equation tells us that the **outgoing radiance** $L_o$ in direction $\omega_o$ (e.g., toward the camera) is the sum of:
- $L_e$: the **emitted radiance** from the surface itself, and
- the **reflected radiance**, which is an integral over all incoming directions $\omega_i$ on the hemisphere $\Omega$.

Let’s focus on the reflected part of the equation, which we want to approximate using Monte Carlo. For convenience, define the integrand as:

$$
f(\omega_i) = f_r(x, \omega_i, \omega_o) \, L_i(x, \omega_i) \, (\omega_i \cdot n)
$$

This $f(\omega_i)$ represents the amount of incoming light from direction $\omega_i$ that gets reflected toward $\omega_o$ according to the surface’s BRDF.

To evaluate the integral:

$$
\int_{\Omega} f(\omega_i) \, d\omega_i
$$

we apply Monte Carlo integration. We sample $N$ directions $\omega_i^{(k)}$ from some PDF $p(\omega_i)$ over the hemisphere $\Omega$, and compute:

$$
\int_{\Omega} f(\omega_i) \, d\omega_i \approx \frac{1}{N} \sum_{k=1}^{N} \frac{f(\omega_i^{(k)})}{p(\omega_i^{(k)})}
$$

Substituting the original integrand back in, we get the Monte Carlo estimate of the full rendering equation:

$$
L_o(x, \omega_o) \approx L_e(x, \omega_o) + \frac{1}{N} \sum_{k=1}^{N} \frac{f_r(x, \omega_i^{(k)}, \omega_o) \, L_i(x, \omega_i^{(k)}) \, (\omega_i^{(k)} \cdot n)}{p(\omega_i^{(k)})}
$$

This is the **Monte Carlo form of the rendering equation**.

At this point, the idea is simple: at every surface point we hit, we sample $N$ incoming directions, estimate how much light comes from those directions, weigh them appropriately, and add them up. If we do this for every pixel, we get an image. Sounds good in theory… but there’s a catch.

---
### Monte Carlo in Path Tracing

Here’s the problem: the term $L_i(x, \omega_i)$ − the incoming radiance − isn’t something we know in advance. In fact, **it’s defined by the same rendering equation** − just at a different surface point, in a different direction.

This means that to evaluate $L_i$, we need to trace a new ray from point $x$ in direction $\omega_i$, find out where it hits, and repeat the process. That’s right: we’re solving the same equation again. **Recursively.**

This is exactly what **path tracing** does.

At a high level, path tracing works like this:

1. We shoot a ray from the camera into the scene.
2. It hits a surface at point $x$.
3. We evaluate emitted light $L_e$ at that point.
4. Then, we pick a random direction $\omega_i$, trace a new ray into the scene, and compute $L_i$ recursively.
5. We multiply that result by the BRDF and geometry term, divide by the PDF, and return the weighted contribution.
6. We repeat this process for multiple paths, and average their contributions to compute the final color.

Each time a ray bounces off a surface, it spawns a new ray in a randomly chosen direction − forming a **path** through the scene. Some paths hit lights and contribute useful data; others hit black walls or disappear into the void. But if we trace enough of them, we eventually converge to the correct solution.

In pseudocode, a single path might look like this:

```vb
function TracePath(ray):  
	radiance = 0  
	throughput = 1  
	for bounce in range(MAX_DEPTH):  
		hit = IntersectScene(ray)  
		if not hit:  
			break  
		radiance += throughput * EmittedLight(hit)  
		direction = SampleDirectionFromBRDF(hit)  
		pdf = DirectionPDF(hit, direction)  
		throughput *= BRDF(hit) * GeometryTerm / pdf  
		ray = NewRayFrom(hit, direction)  
	return radiance
```


Each path is just one random estimate of the light coming into the camera through a pixel. The more paths you trace per pixel, the better your estimate becomes.

So, in summary:
- **Path tracing** is a recursive application of **Monte Carlo integration** to the **light transport equation**.
- It handles arbitrary materials, complex lighting, and global illumination naturally.
- It’s unbiased: given enough samples, it converges to the ground-truth solution.
- But it’s also noisy: especially in early samples, the image can be very grainy − which is where techniques like **importance sampling**, **MIS**, and **denoising** come into play (we’ll explore in the next chapters).

We now have the mathematical and conceptual foundation for path tracing. In the next chapters, let’s explore how to make it *efficient*.

---
## Conclusion

Path tracing is more than just an algorithm − it's a profound idea rooted in physics, mathematics, and probability. By reinterpreting the rendering problem as a high-dimensional integral, and then tackling it using Monte Carlo methods, we arrive at a technique that is both theoretically sound and visually stunning.

We’ve seen how the seemingly simple rendering equation hides tremendous complexity − from its recursive nature to the sheer dimensionality of light interactions. We explored why traditional approaches like Riemann sums fall short, and how randomness, when applied cleverly, allows us to get meaningful approximations through Monte Carlo integration.

But path tracing is only the beginning.

As elegant as it is, pure path tracing suffers from a major drawback: **variance**, which shows up as noise. And while the algorithm is unbiased and eventually correct, getting to a clean, artifact-free image with a reasonable number of samples requires smarter techniques − like **importance sampling**, **multiple importance sampling (MIS)**, **Russian roulette**, and **denoising**.

These ideas will be the focus of the next chapters. We’ll dive into how we can *guide* our sampling, *balance* different light sources, and *accelerate* convergence − all while preserving physical correctness.

For now, if you’ve followed along this far, congratulations − you’ve just built a solid foundation for understanding one of the most powerful tools in physically based rendering.

Let’s keep tracing.
