# Human Face Image Generation, part #1

## Introduction

While extensive literature exists on human face synthesis using GANs and other established techniques, significantly less attention has been paid to modern face generation models that can be trained and deployed entirely on consumer-grade hardware. This work addresses this gap by presenting a complete face generation pipeline based on latent flow matching, designed specifically for consumer GPU constraints.

This article presents the first component of this system: a variational autoencoder (VAE) for efficient face image compression and reconstruction. The discussion encompasses problem formulation, architectural design decisions, training methodology, and empirical results. The subsequent article in this series will cover the flow matching component and complete pipeline integration.

<hr>

### Problem Statement

State-of-the-art face generation techniques, including diffusion models (Stable Diffusion, DALL-E) and high-fidelity GANs (StyleGAN2/3), achieve remarkable visual quality but impose severe computational requirements that preclude training and deployment on consumer hardware. Diffusion models typically require hundreds of high-end GPUs and weeks of training time. Furthermore, their iterative denoising process necessitates hundreds to thousands of inference steps, making real-time generation infeasible even on capable hardware.

GANs, while potentially faster at inference, present distinct challenges. StyleGAN architectures are notoriously difficult to train, exhibiting training instabilities and requiring extensive hyperparameter tuning. More fundamentally, GANs demonstrate limited expressiveness compared to flow-based and diffusion models, particularly for conditioning on complex multi-modal inputs. Operating directly in pixel space further exacerbates computational costs for high-resolution synthesis.

The constraints imposed by consumer hardware deployment are multifaceted:

- **Memory Limitations**: Consumer GPUs typically offer 6-16GB VRAM, several orders of magnitude below the memory requirements of production-scale generative models
- **Training Duration**: Practical training must complete within hours or days, not weeks or months
- **Inference Latency**: Real-time or near-real-time generation (< 1 second per image) is essential for interactive applications
- **Quality-Efficiency Trade-off**: Achieving acceptable visual quality under these constraints requires careful architectural and algorithmic design

This work demonstrates that a complete generative pipeline - comprising a custom VAE and flow matching model - can be successfully trained and deployed on a single consumer GPU (RTX 5070 Ti, 16GB VRAM) while maintaining competitive reconstruction and generation quality.

<hr>

## Pipeline Overview

The generation system consists of two main components: a variational autoencoder (VAE) that compresses $$256 \times 256$$ RGB face images into a compact latent representation, and a flow matching model that generates new samples in this latent space. This article focuses on the VAE component; the flow matching implementation will be covered in a subsequent post.

The VAE architecture achieves a 1:768 compression ratio, reducing images from 196,608 dimensions ($$256 \times 256 \times 3$$) to a 256-dimensional latent vector. This aggressive compression serves multiple purposes: (1) making flow matching computationally tractable on consumer hardware, (2) forcing the encoder to extract only the most salient facial features and identity information, and (3) encouraging a dense, smooth posterior distribution in latent space that facilitates effective sampling by the downstream generative model. The decoder subsequently reconstructs RGB images from these compact latent representations.

<hr>

## VAE Architecture

### Design Choices

The encoder follows a hierarchical ResNet-based architecture with progressive downsampling. After an initial convolutional stem (3 $$\to$$ 64 channels), the network processes features through four downsampling stages ($$256 \to 128 \to 64 \to 32 \to 16$$ spatial resolution) with channel progression $$64 \to 128 \to 256 \to 512 \to 512$$. Each downsampling stage uses a strided convolution *(stride=2, kernel=4)* followed by two residual blocks that refine features at the current resolution. The final $$16 \times 16$$ feature map undergoes global average pooling, producing a 512-dimensional feature vector that is linearly projected to the *N-dimensional*\* latent space parameters ($$\mu$$ and $$\log \sigma^2$$). 

\* 256 was empirically found to be an optimal value for final latent dimensionality (N)

Key architectural decisions:
- **ResBlock design**: Two $$3 \times 3$$ convolutions with pre-activation (GroupNorm $$\to$$ SiLU $$\to$$ Conv), using 8 groups for normalization
- **Downsampling**: Learned strided convolutions rather than fixed pooling operations
- **Normalization**: Group normalization with 8 groups for stability with small batch sizes (crucial for consumer GPU constraints)
- **Activation**: SiLU (Swish) throughout for smooth gradients
- **Bottleneck**: Global average pooling converts spatial features to a compact vector representation before latent projection

The decoder uses a symmetric architecture but with nearest-neighbor upsampling followed by $$3 \times 3$$ convolutions instead of transposed convolutions. This design choice eliminates checkerboard artifacts common in transposed convolution outputs. The decoder begins by linearly projecting the 256-dimensional latent vector to $$512 \times 16 \times 16$$ spatial features, then progressively upsamples through four stages ($$512 \to 256 \to 128 \to 64 \to 32$$ channels) with two residual blocks after each upsampling. The final layer applies a $$3 \times 3$$ convolution and tanh activation to produce RGB output in [-1, 1] range.

### Latent Space Configuration

The latent space employs a diagonal Gaussian posterior $$q(z\mid x)$$, parameterized by mean $$\mu$$ and log-variance $$\log \sigma^2$$ vectors. The 256-dimensional latent representation provides sufficient capacity to encode facial identity, expression, pose, and lighting information while remaining compact enough for efficient downstream generation.

Empirical evaluation compared standard VAE and $$\beta$$-VAE formulations. The final architecture adopts $$\beta = 0.05$$, significantly lower than typical $$\beta$$-VAE implementations. This choice prioritizes reconstruction fidelity over strict adherence to the Gaussian prior, allowing the posterior to deviate moderately from $$\mathcal{N}(0, I)$$ when necessary to preserve fine-grained facial details. The aggressive compression ratio naturally encourages a continuous and compact latent space even with reduced KL regularization.

A critical design objective is maintaining latent space smoothness while preserving information content sufficient for downstream conditional generation. The flow matching model requires informative latent codes to enable effective sampling. Consequently, the KL divergence term cannot be driven to zero, as perfect alignment with a standard Gaussian prior would eliminate the semantic structure necessary for controlled generation.

<hr>

## Training Procedure

### Dataset and Preprocessing

Training was conducted on the FFHQ-256 dataset, comprising 70,000 high-quality, aligned and cropped face images at $$256 \times 256$$ resolution. The dataset provides diverse coverage of age, ethnicity, gender, accessories, and lighting conditions.

Data augmentation includes horizontal mirroring and subtle random color jitter (±0.1 brightness, ±0.1 contrast, ±0.05 saturation and hue). These augmentations improve generalization while preserving facial identity. Images are normalized to the [-1, 1] range to match the decoder's tanh output activation. 

### Loss Function

The training objective combines reconstruction loss and KL divergence:

$$
\mathcal{L} = \mathcal{L}_{recon} - \beta \cdot D_{KL}(q(z|x) || p(z))
$$

The reconstruction term combines L1 pixel-wise loss with perceptual loss computed from VGG feature representations:

$$
\mathcal{L}_{recon} = \lambda_1 ||x - \hat{x}||_1 + \lambda_{perceptual} \sum_{l \in \mathcal{L}} ||\phi_l(x) - \phi_l(\hat{x})||_2
$$

where $$\phi_l(\cdot)$$ extracts features from layer $$l$$ of a pretrained VGG-19 network. The perceptual loss is computed across multiple intermediate layers (relu1_2, relu2_2, relu3_3, relu4_3) to capture both low-level textures and high-level semantic features. This multi-scale perceptual loss substantially improves reconstruction quality, particularly for high-frequency facial details such as skin texture, hair strands, and fine wrinkles that are poorly captured by pixel-wise metrics alone.

#### Adversarial Fine-tuning

Following convergence of the base VAE training, an additional fine-tuning phase incorporates adversarial loss to further enhance reconstruction quality and reduce artifacts. This two-stage approach prevents early training instabilities that can arise when adversarial objectives are introduced prematurely. During fine-tuning, the total loss becomes:

$$
\mathcal{L}_{total} = \mathcal{L}_{recon} - \beta \cdot D_{KL}(q(z|x) || p(z)) + \lambda_{adv} \mathcal{L}_{adv}
$$

The adversarial loss uses a PatchGAN discriminator architecture, which evaluates local image patches rather than the entire image. This multi-scale approach is particularly effective at enforcing high-frequency detail and texture quality:

$$
\mathcal{L}_{adv} = -\mathbb{E}_{x \sim p_{data}}[\log D(x)] - \mathbb{E}_{z \sim q(z|x)}[\log(1 - D(G(z)))]
$$

where $$D(\cdot)$$ is the PatchGAN discriminator and $$G(z)$$ represents the decoded reconstruction. The discriminator outputs a feature map where each element classifies a local patch, providing spatially-aware feedback that helps preserve fine details like skin texture and hair strands that may be smoothed by the perceptual loss alone.

The adversarial component is activated only after base VAE convergence, as premature introduction destabilizes training and can trigger mode collapse. The adversarial weight $$\lambda_{adv} = 0.1$$ balances adversarial feedback against reconstruction fidelity, preventing the discriminator from overwhelming the reconstruction objectives. The discriminator is trained with label smoothing (real labels = 0.9, fake labels = 0.1) to improve training stability.

<div style="display: flex; justify-content: center; align-items: center; gap: 20px; margin: 20px 0;">
  <img src="https://github.com/user-attachments/assets/0b063883-6110-48e8-8dda-72bfe56d1dcf" alt="VAE reconstruction before adversarial fine-tuning" style="max-width: 256px; height: auto;" />
  <img src="https://github.com/user-attachments/assets/50ee5948-bab7-4612-b1cf-9ac2f9e280ab" alt="VAE reconstruction after adversarial fine-tuning" style="max-width: 256px; height: auto;" />
</div>

<p style="text-align: center;"><i>Results of adversarial fine-tuning. First image representation pure reconstruction + KL loss, while second one is post-fine-tuning result</i></p>

### Training Configuration

The model is trained using the following hyperparameters and infrastructure:

**Hardware and Framework:**
- Single NVIDIA RTX 5070 Ti (16GB VRAM)
- PyTorch 2.0+ with mixed precision (fp16) via automatic mixed precision (AMP)
- Gradient checkpointing disabled (sufficient memory available)

**Optimization:**
- Optimizer: AdamW ($$\beta_1 = 0.9$$, $$\beta_2 = 0.999$$, weight decay = 0.01)
- Base learning rate: 0.0001
- Learning rate schedule: Linear warmup over first 10% of training, followed by cosine annealing to 5% of peak rate
- Batch size: 24 (effective batch size 24, no gradient accumulation required)
- Total epochs: 180 (approximately 50 hours of training time)

**Loss Configuration:**
- KL divergence weight ($$\beta$$): 0.05 (with cosine annealing from 0 over first 30% of training)
- Reconstruction loss weight: 1.0
- Adversarial loss weight (fine-tuning only): 0.1
- Gradient clipping: max norm 1.0 to prevent exploding gradients

**Regularization:**
- KL annealing prevents posterior collapse during early training
- No dropout (normalization provides sufficient regularization)
- Data augmentation as described above

## Results and Analysis

### Reconstruction Quality

The trained VAE demonstrates strong reconstruction fidelity, particularly for salient facial features. Identity, expression, pose, and major attributes are preserved accurately through the encode-decode cycle. The aggressive compression necessitates some loss of fine-grained details - individual hair strands, skin pores, and background textures exhibit smoothing - but overall facial structure and appearance remain intact.

**Quantitative Evaluation** (validation set, 7,000 held-out images):
- **PSNR**: 9.56 dB
- **SSIM**: 0.31
- **LPIPS** (Learned Perceptual Image Patch Similarity): 0.67

These metrics warrant contextualization. First, the 1:768 compression ratio inherently limits reconstruction fidelity compared to models with less aggressive compression. Second, the architecture explicitly prioritizes facial feature reconstruction over background fidelity, as facial regions are the primary target for downstream generation. Background reconstruction artifacts disproportionately impact global metrics like PSNR and SSIM, which weight all pixels equally. In contrast, LPIPS - which correlates better with human perceptual judgments - indicates acceptable perceptual quality, particularly for facial regions.

**Qualitative Assessment:**

Visual comparison of original and reconstructed images demonstrates the model's capability to preserve identity and expression:

<img src="https://github.com/user-attachments/assets/adf094b6-0e1f-4a3a-bb35-f7f1e4da0a7f" alt="Comparison grid showing original images (top row) and VAE reconstructions (bottom row)" style="max-width: 100%; height: auto; display: block;" />

<p style="text-align: center;"><i>Comparison grid: Original images from FFHQ validation set (top row) and corresponding VAE reconstructions after adversarial fine-tuning (bottom row)</i></p>

The reconstruction successfully preserves facial identity, expression, and key attributes while exhibiting expected smoothing in high-frequency details.

### Latent Space Properties

Empirical analysis of the learned latent space reveals several properties conducive to downstream generation:

- **Smoothness and Continuity**: Linear interpolation between two face encodings produces semantically coherent intermediate faces without abrupt transitions or artifacts. This suggests the latent space forms a continuous manifold where nearby points correspond to perceptually similar faces.

- **Partial Disentanglement**: While explicit disentanglement objectives are not employed, informal analysis suggests different latent dimensions capture somewhat independent factors of variation (e.g., pose, lighting, expression). Complete disentanglement is not achieved - as expected without explicit supervision such as $$\beta$$-VAE with very high $$\beta$$ or supervised attribute labels - but the observed partial separation facilitates interpretable latent space navigation.

- **Distribution Coverage**: The posterior distribution $$q(z\mid x)$$ balances adherence to the Gaussian prior $$\mathcal{N}(0, I)$$ with preservation of semantic information. KL divergence per dimension averages 0.32 nats, indicating moderate but non-negligible deviation from the prior. This trade-off enables the flow matching model to leverage a relatively well-behaved prior while retaining sufficient information for conditional generation.

- **Latent Manipulation and Feature Control**: Empirical exploration reveals that targeted modifications to specific dimensions in the 256-dimensional latent vector produce interpretable changes in decoded facial features. Adjusting individual latent values can influence attributes such as facial expression (smile intensity, mouth opening), eye characteristics (size, openness), and other facial features. However, these manipulations demonstrate the inherently entangled nature of the representation: individual latent dimensions do not correspond to single, isolated semantic attributes. Rather, each dimension contributes to a combination of features, with multiple dimensions collectively encoding complex attributes. For instance, modifying a single dimension may simultaneously affect smile intensity, mouth curvature, and subtle changes in cheek positioning. This entanglement is expected given the absence of explicit disentanglement objectives during training and the aggressive compression ratio, which necessitates efficient information encoding where each dimension must contribute to multiple facial attributes.

### Latent Space Interpolation

To empirically validate the smoothness and semantic organization of the learned latent space, interpolation experiments were conducted between pairs of face encodings. Given two face images $$x_1$$ and $$x_2$$ with corresponding latent encodings $$z_1 = \mu_1$$ and $$z_2 = \mu_2$$ (using the posterior mean for deterministic interpolation), intermediate latent codes are generated via spherical linear interpolation (SLERP):

$$
z_t = \frac{\sin((1-t)\theta)}{\sin(\theta)} z_1 + \frac{\sin(t\theta)}{\sin(\theta)} z_2
$$

where $$\theta = \arccos\left(\frac{z_1 \cdot z_2}{\lVert z_1 \rVert \, \lVert z_2 \rVert}\right)$$ and $$t \in [0, 1]$$. SLERP is preferred over linear interpolation to maintain consistent norm throughout the trajectory, which better respects the geometry of the approximately Gaussian latent distribution.

**Spatial Feature Interpolation**: The interpolation results demonstrate smooth transitions across multiple semantic attributes simultaneously. Facial identity morphs continuously between the two endpoint identities without discrete jumps or artifacts. Crucially, spatial features - including pose orientation, face shape, and facial feature positioning - interpolate coherently. For instance, interpolating between a frontal face and a profile view produces intermediate poses at natural angles, while facial features (eyes, nose, mouth) transition smoothly in both position and appearance.

Expression and lighting also exhibit smooth interpolation. Transitioning between different expressions (e.g., neutral to smiling) produces natural intermediate expressions without anatomically implausible configurations. Similarly, lighting direction and intensity change gradually across the interpolation trajectory, suggesting the latent space captures these attributes as continuous rather than discrete factors.

**Interpolation Quality**: While some fine details (hair texture, background elements) may exhibit slight inconsistencies during interpolation - a consequence of the aggressive compression - the overall perceptual quality remains high. The facial region maintains coherent identity and realistic appearance throughout the trajectory. This behavior validates the design choice of prioritizing facial features during training: the attributes most relevant to face generation are precisely those that interpolate most smoothly.

<img src="https://github.com/user-attachments/assets/1878a33c-a107-4665-b7f7-2486f1c8a8ee" alt="Comparison grid showing original images (top row) and VAE reconstructions (bottom row)" style="max-width: 100%; height: auto; display: block;" />
<p style="text-align: center;"><i>Latent space interpolation between two face encodings. Animation demonstrates smooth transitions in identity, pose, expression, and spatial features</i></p>

These interpolation experiments confirm that the learned latent space possesses the geometric properties necessary for effective generative modeling: smoothness, continuity, and semantic organization. The flow matching model can leverage these properties to generate diverse, high-quality samples through latent space sampling and traversal.

## Challenges and Design Insights

### Training Stability and Posterior Collapse

Initial experiments with $$\beta > 0.2$$ consistently resulted in posterior collapse: the model learned to ignore the latent code entirely, relying solely on the decoder's expressiveness to reconstruct images. This manifests as near-zero KL divergence and failure of the latent space to encode meaningful information.

Two interventions successfully mitigated this issue. First, reducing $$\beta$$ to 0.05 permits greater posterior deviation from the prior, reducing pressure toward the uninformative $$\mathcal{N}(0, I)$$ distribution. Second, implementing KL annealing - gradually increasing $$\beta$$ from 0 over the first 30% of training - allows the encoder to first learn meaningful representations before regularization pressure is applied. Additionally, a "free bits" constraint ensures minimum information content per latent dimension, preventing complete collapse.

### Compression-Fidelity Trade-off

The 1:768 compression ratio is aggressive for $$256 \times 256$$ RGB images, particularly compared to typical VAE compression ratios of 1:16 to 1:64. This design choice reflects the project's constraint-driven objectives: minimizing computational requirements for downstream flow matching necessitates compact representations.

The primary limitation is background reconstruction quality. The VAE prioritizes facial features - where most semantic content resides - at the expense of backgrounds, which often exhibit blurring and loss of fine textures. This trade-off is acceptable for the intended application (face generation), but represents a clear area for improvement in future work. Potential solutions include spatial attention mechanisms that explicitly allocate capacity to facial regions, or hierarchical latent representations that dedicate separate encodings to foreground and background.

## Conclusion and Future Work

This work presents a variational autoencoder designed for consumer-hardware-constrained face image generation. The architecture achieves 1:768 compression while preserving facial identity and attributes, enabling downstream generative modeling on a single consumer GPU. Key contributions include:

1. A hierarchical ResNet-based VAE architecture optimized for aggressive compression under memory constraints
2. A two-stage training procedure combining perceptual and adversarial losses for high-quality reconstruction
3. Empirical validation of design choices for latent space configuration and regularization strength
4. Demonstration of feasible training on consumer hardware (RTX 5070 Ti, 50 hours)

The VAE component establishes the foundation for the complete generation pipeline. Part #2 of this series will present the flow matching model, covering:

- **Flow Matching Architecture**: Continuous flow matching operating over latent space
- **Conditioning Mechanisms**: Enabling controlled generation via attribute vectors or other modalities
- **Sampling Procedures**: Efficient ODE solvers for fast, high-quality sampling
- **End-to-End Evaluation**: Complete pipeline integration and generation quality assessment
