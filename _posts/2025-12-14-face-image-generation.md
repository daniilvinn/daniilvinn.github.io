# Human Face Image Generation, part #1

## Introduction
There are many articles on using GANs and other older solution for human face image synthesis. However, there are not so many (if any) articles / posts on human face generation models that can be trained and deployed entirely on a regular user PC, while still using modern techniques for image synthesis. 

In this post, I will cover a project I am currently working on - human face generation using latent flow matching, particularly the problem overview alongside with VAE design and training.

<hr>

### The problem

Modern face generation techniques like diffusion models (Stable Diffusion, DALL-E) produce stunning results, but they come with significant computational requirements and are nowhere near to be considered as a project that one can fully implement and train on a regular user PC. Training these models typically requires multiple (usually hundreds and more) high-end GPUs and enormous amount of training time. Problem does not fanish even at inference time - diffusion models are *stochastic* by definition, and in order to perform well, one should make many steps (usually hundreds, up to a thousand) in order to achieve desired result.

Similarly, while GANs (like StyleGAN2/3) can generate high-quality faces, they present several fundamental limitations. They are notoriously difficult to train, requiring careful hyperparameter tuning and suffering from training instabilities. More importantly, GANs are less expressive than flow-based and diffusion models, particularly when it comes to conditioning on complex inputs such as text descriptions or multi-modal attributes. They also operate directly in pixel space, which is computationally expensive for high-resolution images.

The key challenges for consumer hardware deployment are:
- **Limited VRAM**: Most users have GPUs with 6-16GB of memory which is nowhere near to terabytes of VRAM that are used to train current SOTA models
- **Training time**: Models must train in hours or a few days, not weeks
- **Inference speed**: Generation should be near real-time (under a second per image)
- **Quality trade-off**: Can we achieve competitive quality with these constraints?

Even considering all these constaints, I managed to design, train and deploy full generation pipeline that fully consists of custom models - VAE and flow matching - all on a single home PC's GPU.

<hr>

## Pipeline Overview

The generation system consists of two main components: a variational autoencoder (VAE) that compresses $256 \times 256$ RGB face images into a compact latent representation, and a flow matching model that generates new samples in this latent space. This article focuses on the VAE component; the flow matching implementation will be covered in a subsequent post.

The VAE compresses images from 196,608 dimensions ($256 \times 256 \times 3$) to a 256-dimensional vector in latent space, providing 1:768 compression ratio. This aggressive compression is necessary to make flow matching computationally feasible on consumer hardware while preserving facial features and identity information and forces VAE decoder to form dense, smooth posterior distribution that is further used by a decoder to reconstruct an image back into RGB space.

<hr>

## VAE Architecture

### Design Choices

The encoder follows a hierarchical ResNet-based architecture with progressive downsampling. After an initial convolutional stem (3 $\to$ 64 channels), the network processes features through four downsampling stages ($256 \to 128 \to 64 \to 32 \to 16$ spatial resolution) with channel progression $64 \to 128 \to 256 \to 512 \to 512$. Each downsampling stage uses a strided convolution *(stride=2, kernel=4)* followed by two residual blocks that refine features at the current resolution. The final $16 \times 16$ feature map undergoes global average pooling, producing a 512-dimensional feature vector that is linearly projected to the *N-dimensional*\* latent space parameters ($\mu$ and $\log \sigma^2$). 

\* 256 was empirically found to be an optimal value for final latent dimensionality (N)

Key architectural decisions:
- **ResBlock design**: Two $3 \times 3$ convolutions with pre-activation (GroupNorm $\to$ SiLU $\to$ Conv), using 8 groups for normalization
- **Downsampling**: Learned strided convolutions rather than fixed pooling operations
- **Normalization**: Group normalization with 8 groups for stability with small batch sizes (crucial for consumer GPU constraints)
- **Activation**: SiLU (Swish) throughout for smooth gradients
- **Bottleneck**: Global average pooling converts spatial features to a compact vector representation before latent projection

The decoder uses a symmetric architecture but with nearest-neighbor upsampling followed by $3 \times 3$ convolutions instead of transposed convolutions. This design choice eliminates checkerboard artifacts common in transposed convolution outputs. The decoder begins by linearly projecting the 256-dimensional latent vector to $512 \times 16 \times 16$ spatial features, then progressively upsamples through four stages ($512 \to 256 \to 128 \to 64 \to 32$ channels) with two residual blocks after each upsampling. The final layer applies a $3 \times 3$ convolution and tanh activation to produce RGB output in [-1, 1] range.

### Latent Space Configuration

The latent space uses a diagonal Gaussian posterior $q(z|x)$, parameterized by mean $\mu$ and log-variance $\log \sigma^2$ vectors. The 512-dimensional latent vector provides sufficient capacity to encode facial identity, expression, pose, and lighting while remaining compact enough for efficient downstream generation.

I experimented with both standard VAE and $\beta$-VAE formulations. The final model uses very little $\beta = 0.05$ to balance reconstruction quality against latent space regularization, allowing some deviation from a standard Gaussian prior in favor of better reconstructions. It applies to our setup especially well, considering that latent space is highly compressed, naturally forcing the model to form continuos and compact latent space.

The main task was to keep latent space smooth, yet force it to carry useful information for data generation, since downstream flow matching model may require it for conditional flow matching. All that means that we cannot force KL divergence to zero, aligning posterior distribution with normal Gaussian one.

<hr>

## Training Procedure

### Data

The model was trained on FFHQ-256 dataset, consisting of 70,000 aligned and cropped face images at $256 \times 256$ resolution. Images were preprocessed with.

It was decided to apply augmentations: mirroring, slight random color jitter. 

### Loss Function

The training objective combines reconstruction loss and KL divergence:

$$
\mathcal{L} = \mathcal{L}_{recon} - \beta \cdot D_{KL}(q(z|x) || p(z))
$$

For reconstruction, I use a combination of L1 loss and perceptual loss based on VGG features:

$$
\mathcal{L}_{recon} = \lambda_1 ||x - \hat{x}||_1 + \lambda_{perceptual} ||\phi(x) - \phi(\hat{x})||_2
$$

where $\phi(\cdot)$ extracts features from intermediate VGG layers. The perceptual loss significantly improved reconstruction quality, particularly for high-frequency facial details like skin texture and hair.

#### Adversarial Fine-tuning

After the initial training phase converges, it was decided to run an additional fine-tuning pass with adversarial loss to further improve reconstruction quality and reduce artifacts. During this phase, the total loss becomes:

$$
\mathcal{L}_{total} = \mathcal{L}_{recon} - \beta \cdot D_{KL}(q(z|x) || p(z)) + \lambda_{adv} \mathcal{L}_{adv}
$$

The adversarial loss uses a PatchGAN discriminator architecture, which evaluates local image patches rather than the entire image. This multi-scale approach is particularly effective at enforcing high-frequency detail and texture quality:

$$
\mathcal{L}_{adv} = -\mathbb{E}_{x \sim p_{data}}[\log D(x)] - \mathbb{E}_{z \sim q(z|x)}[\log(1 - D(G(z)))]
$$

where $D(\cdot)$ is the PatchGAN discriminator and $G(z)$ represents the decoded reconstruction. The discriminator outputs a feature map where each element classifies a local patch, providing spatially-aware feedback that helps preserve fine details like skin texture and hair strands that may be smoothed by the perceptual loss alone.

The adversarial component is only enabled after the base VAE has converged, as introducing it too early can destabilize training and lead to mode collapse. During fine-tuning, I use a small weight $\lambda_{adv} = 0.1$ to balance adversarial feedback against reconstruction fidelity.

### Training Configuration

- **Hardware**: Single RTX 5070 Ti with 16GB VRAM
- **Batch size**: 24
- **Optimizer**: AdamW with learning rate 0.0001
- **Learning rate schedule**: Linear warmup first 10% of epochs, cosine annealing to 5% of top LR
- **Training time**: approximately 50 hours
- **Framework**: PyTorch with mixed precision training (fp16)
- **KL-divergence loss weight**: 0.05
- **Reconstruction loss weight**: 1.0
- **Beta annealing** - cosine annealing first 30% of epochs
- **Epoch count** - 180

## Results and Analysis

### Reconstruction Quality

The trained VAE achieves strong reconstruction quality with minimal perceptual loss. Facial identity, expression, and major attributes are preserved accurately through the encode-decode cycle. Some fine details like individual hair strands and skin pores are smoothed due to the compression, but overall structure and appearance remain intact.

Quantitative metrics on validation set:
- **PSNR**: 9.56 dB
- **SSIM**: 0.31
- **LPIPS**: 0.67

Note that these metrics may appear suboptimal in absolute terms, partially due to the aggressive compression ratio, but also because the model prioritizes facial feature reconstruction over background quality. As discussed in the Challenges section, background reconstruction remains imperfect, which negatively impacts global metrics like PSNR and SSIM. However, the facial regions themselves—which are the primary focus for downstream generation—maintain high perceptual quality.

### Latent Space Properties

Analysis of the learned latent space reveals several desirable properties:
- **Smoothness**: Interpolation between two face encodings produces semantically meaningful intermediate faces
- **Disentanglement**: Different dimensions capture somewhat independent factors (though not perfectly disentangled without explicit supervision)
- **Coverage**: The latent distribution attempts to follow the imposed Gaussian prior while preserving important information, facilitating downstream generation

## Challenges and Lessons Learned

### Training Stability

Initial training attempts with $\beta > 0.2$ resulted in posterior collapse, where the model learned to ignore the latent code and rely primarily on the decoder's capacity. Reducing $\beta$ < 0.2 and "free bits" technique to force information preservation resolves this issue.

### Compression vs. Quality Trade-off

The $1:768$ compression is quite aggressive for $256 \times 256$ images. Yet considering the project goal, VAE decoder was able to reconstruct facial features with good quality. The main issue was reconstruction of a background of a generated image, which is still an issue to resolve.

## Next Steps

With the VAE component trained and validated, the next phase involves training the flow matching model in the learned latent space. Part #2 of this series will cover:
- Flow matching architecture and training
- Possible conditioning mechanisms for controlled generation
- Sampling procedures and quality evaluation
- Complete pipeline integration and results