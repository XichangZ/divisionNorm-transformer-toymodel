# My first exploration
After learning the nanoGPT taught by Karpathy,I tried to understand the functions of Norms in my mindset.In this period,I found a kind of Norm in biology.I learned about it,and I found that this Norm is really fit to my mindset--each design should have some meanings in philosophy.
## Conclusion
When mixing ReLU in the hidden layers with DN in the normal layers, the BPC (Bits Per Character) is slightly lower than that of a conventional FFN (by approximately 0.02), while the computational speed remains virtually unchanged.
Applying DN concurrently across both the T (sequence) and C (channel) dimensions in the hidden layers can completely replace non-linear activations (ReLU/GELU). However, this comes at the cost of reduced computational speed.
## Limitation
The main limitations come from two aspects:

1.Lack of Compute and Time: I do not have sufficient computational resources or time to conduct experimental validation at a larger scale. It is possible that such modifications might be meaningless or even a waste of resources at scale; however, the hybrid approach of mixing DN with activation functions like ReLU might also yield better performance with larger parameter scales.

2.VRAM Constraints: The other limitation is GPU memory (VRAM). The current implementation of DN is extremely memory-intensive, which is the primary reason I couldn't push forward with more experiments.The memory overhead is primarily caused by the explicit computation of statistics across both the T and C dimensions, which prevents PyTorch from fusing the operations and creates large intermediate tensors.

Perhaps after learning more related knowledge in the future, I will return to improve this project. Although this finding is relatively trivial, it is my first research exploration project, and it has brought me immense joy and personal growth.
## 0.Talk about LN
What does LN do?In my opinion,LN aggregates all information across the feature (channel) dimension, which introduces a certain degree of noise to some extent, thereby enhancing the model's attention mechanism and generalization capability.But, the network cann't decide which features to be aggregated,that's why I was hooked by DN for almost a month.  
## 1 the formula in biology

$$y_i=\frac{z_i}{\beta_i+\sum_j \gamma_{ij}\,|z_j|^{\alpha}}$$

## 2 the formula I use

$$h=\big(x-\overline{x}_{C}\big)^{\!\top},\qquad e_j=|h_j|^{p},\qquad D=\beta+\Gamma e$$

$$\Gamma e=k\odot e+U\left(V^{\top}e\right),\qquad \gamma_{ij}=\sum_{q=1}^{r}u_{iq}\,v_{jq}$$

$$g_i=\frac{D_i}{D_i^{2}+1},\qquad y_i=h_i\,g_i$$

## 3 Why do I think it useful

After reading my thoughts on LN, you might notice that I actually prefer doing this DN in the feature (Channel) dimension. But funny enough, as a beginner, I messed up using PyTorch's nn.Conv1d at the very beginning, which made me apply DN to the temporal (T) dimension first. Surprisingly, this gave a small but real improvement, and that's why the formula for calculating D includes a term for the T dimension.

Later, to fix this "mistake", I started adding C-dimension processing. But I quickly realized that using the exact same method isn't necessarily optimal because the C dimension is just too large. Thinking back to the CR (Column-Row) decomposition I learned in linear algebra, I figured: not all feature dimensions are equally critical. Especially in hidden layers, column vectors are very likely to be linearly correlated. So, I used CR decomposition to handle this. This lets the model decide by itself which vectors should be correlated and which columns are more important, without dragging a massive C×C matrix into backpropagation. That's the whole story behind the C dimension processing.

As for the smoothing part, that idea actually came from an AI. But I didn't listen to its advice to set epsilon very small. Instead, I made it look like an activation function, using pre-clamping to keep the input D within [−1,1]. The AI also suggested initializing D near 1, but I don't think that's a good idea. It would trap the network in a local optimum right at the extreme point of D=1 too early, making the learnable parameters pretty much useless. The trade-off is giving up the regions outside [−1,1], since most D values won't go there anyway—they'll just get stuck in local optima at the boundaries.

So, regarding this smoothing part, there are actually quite a few experiments I haven't tried yet.

## 4 datas
here are a set of datas based on seed547844,I also did tests on other seeds,the results are similar.And the best BPC even reached the 1.813.I will spend some time on enriching the datas on my holidays.
### 4.1 without DN — s547844, 13,045,944 params, 6001 steps

| step | train | val | BPC |
|---|---|---|---|
| 0 | 7.3381 | 7.4210 | 3.454 |
| 500 | 4.5678 | 4.8485 | 2.257 |
| 1000 | 4.0112 | 4.4568 | 2.074 |
| 1500 | 3.7115 | 4.2659 | 1.985 |
| 2000 | 3.5021 | 4.1530 | 1.933 |
| 2500 | 3.3422 | 4.0772 | 1.898 |
| 3000 | 3.2115 | 4.0351 | 1.878 |
| 3500 | 3.1078 | 4.0069 | 1.865 |
| 4000 | 3.0200 | 3.9548 | 1.841 |
| 4500 | 2.9960 | 3.9493 | 1.838 |
| 5000 | 2.9759 | 3.9445 | 1.836 |
| 5500 | 2.9591 | 3.9432 | 1.835 |
| 6000 | 2.9442 | 3.9410 | 1.834 |

### 4.2 without DN — s547844, 14,817,720 params, 6001 steps

| step | train | val | BPC |
|---|---|---|---|
| 0 | 7.3064 | 7.3896 | 3.439 |
| 500 | 4.5487 | 4.8291 | 2.248 |
| 1000 | 3.9818 | 4.4350 | 2.064 |
| 1500 | 3.6734 | 4.2522 | 1.979 |
| 2000 | 3.4633 | 4.1392 | 1.926 |
| 2500 | 3.3099 | 4.0764 | 1.897 |
| 3000 | 3.1769 | 4.0267 | 1.874 |
| 3500 | 3.0785 | 4.0081 | 1.865 |
| 4000 | 2.9793 | 3.9632 | 1.845 |
| 4500 | 2.9545 | 3.9594 | 1.843 |
| 5000 | 2.9353 | 3.9554 | 1.841 |
| 5500 | 2.9162 | 3.9535 | 1.840 |
| 6000 | 2.8999 | 3.9496 | 1.838 |

### 4.3 have DN just for T — s547844_, rank 8, 13,230,264 params, 5001 steps

| step | train | val | BPC | positive | reverse | drift | β |
|---|---|---|---|---|---|---|---|
| 0 | 7.4032 | 7.4871 | 3.485 | 50.0% | 0.0% | 0.003 | -1.000 |
| 500 | 4.3822 | 4.7021 | 2.188 | 50.0% | 0.0% | 0.296 | -1.023 |
| 1000 | 3.9159 | 4.3622 | 2.030 | 50.0% | 0.0% | 0.426 | -1.010 |
| 1500 | 3.6457 | 4.1937 | 1.952 | 50.0% | 0.0% | 0.511 | -0.999 |
| 2000 | 3.4549 | 4.0861 | 1.902 | 50.1% | 0.1% | 0.577 | -0.988 |
| 2500 | 3.3089 | 4.0249 | 1.873 | 50.3% | 0.3% | 0.628 | -0.973 |
| 3000 | 3.1915 | 3.9812 | 1.853 | 50.7% | 0.7% | 0.670 | -0.958 |
| 3500 | 3.1359 | 3.9480 | 1.837 | 50.7% | 0.7% | 0.673 | -0.955 |
| 4000 | 3.1141 | 3.9408 | 1.834 | 50.7% | 0.7% | 0.676 | -0.950 |
| 4500 | 3.0976 | 3.9343 | 1.831 | 50.8% | 0.8% | 0.679 | -0.947 |
| 5000 | 3.0827 | 3.9296 | 1.829 | 50.7% | 0.7% | 0.682 | -0.943 |

### 4.4 have DN for both T and C— s547844, rank 8, 13,161,144 params, 7001 steps

| step | train | val | BPC | 正核 | 翻转 | 漂移 | β |
|---|---|---|---|---|---|---|---|
| 0 | 7.4650 | 7.5303 | 3.505 | 50.0% | 0.0% | 0.002 | -1.000 |
| 500 | 4.2651 | 4.6060 | 2.144 | 50.0% | 0.0% | 0.130 | -1.018 |
| 1000 | 3.7821 | 4.2844 | 1.994 | 50.0% | 0.0% | 0.189 | -1.010 |
| 1500 | 3.5113 | 4.1196 | 1.917 | 50.0% | 0.0% | 0.229 | -1.005 |
| 2000 | 3.3161 | 4.0265 | 1.874 | 50.1% | 0.1% | 0.259 | -0.999 |
| 2500 | 3.1677 | 3.9839 | 1.854 | 50.1% | 0.1% | 0.282 | -0.994 |
| 3000 | 3.0428 | 3.9464 | 1.837 | 50.2% | 0.2% | 0.301 | -0.988 |
| 3500 | 2.9468 | 3.9420 | 1.835 | 50.3% | 0.3% | 0.318 | -0.983 |
| 4000 | 2.8494 | 3.9349 | 1.831 | 50.5% | 0.5% | 0.333 | -0.978 |
| 4500 | 2.7551 | 3.9050 | 1.817 | 50.6% | 0.6% | 0.340 | -0.976 |
| 5000 | 2.7278 | 3.9034 | 1.817 | 50.6% | 0.6% | 0.341 | -0.975 |
| 5500 | 2.7084 | 3.9103 | 1.820 | 50.6% | 0.6% | 0.342 | -0.974 |
| 6000 | 2.6862 | 3.9071 | 1.818 | 50.6% | 0.6% | 0.343 | -0.973 |
| 6500 | 2.6715 | 3.9098 | 1.820 | 50.6% | 0.6% | 0.345 | -0.972 |
| 7000 | 2.6570 | 3.9102 | 1.820 | 50.6% | 0.6% | 0.346 | -0.971 |
|best  | 2.7278 | 3.9034 | 1.817 | 50.6% | 0.6% | 0.341 | -0.975 |
### 4.5 have DN without RELU— s547844, rank 16, 13,492,920 params, 7001 steps

| step | train | val | BPC | positive | reverse | drift | β |
|---|---|---|---|---|---|---|---|
| 0 | 7.4408 | 7.5033 | 3.492 | 50.0% | 0.0% | 0.006 | -1.000 |
| 500 | 4.3061 | 4.6341 | 2.157 | 50.2% | 0.2% | 0.391 | -1.031 |
| 1000 | 3.8146 | 4.3322 | 2.016 | 50.4% | 0.4% | 0.522 | -1.045 |
| 1500 | 3.5224 | 4.1646 | 1.938 | 50.9% | 0.9% | 0.618 | -1.055 |
| 2000 | 3.3196 | 4.0698 | 1.894 | 51.4% | 1.4% | 0.691 | -1.063 |
| 2500 | 3.1715 | 4.0190 | 1.870 | 51.6% | 1.6% | 0.749 | -1.068 |
| 3000 | 3.0395 | 4.0071 | 1.865 | 51.7% | 1.7% | 0.796 | -1.069 |
| 3500 | 2.9383 | 3.9996 | 1.861 | 51.9% | 1.9% | 0.838 | -1.068 |
| 4000 | 2.8492 | 3.9562 | 1.841 | 51.8% | 1.8% | 0.838 | -1.065 |
| 4500 | 2.8265 | 3.9561 | 1.841 | 51.8% | 1.8% | 0.839 | -1.062 |
| 5000 | 2.8073 | 3.9567 | 1.842 | 51.8% | 1.8% | 0.839 | -1.059 |
| 5500 | 2.8032 | 3.9551 | 1.841 | 51.8% | 1.8% | 0.839 | -1.059 |
| 6000 | 2.8004 | 3.9549 | 1.841 | 51.8% | 1.8% | 0.839 | -1.059 |
| 6500 | 2.7978 | 3.9553 | 1.841 | 51.8% | 1.8% | 0.839 | -1.058 |
| 7000 | 2.7957 | 3.9547 | 1.841 | 51.8% | 1.8% | 0.839 | -1.058 |

## 5 thanks for my AI
Without AI, I wouldn't have had such a joyful time exploring. Most of the code in this repo inherits from my previous project, nanoGPT. The core DN code is handwritten by me, the code for tracking experimental data was written by my AI, and honestly, most of this README's translation was also produced by AI.

Just yesterday, there was an announcement claiming that AI had cracked the Riemann Hypothesis. It made me both excited and a bit sad. It's already hard to imagine what life would be like without AI. Without it, I couldn't have learned this fast, and my life wouldn't be this convenient.

I'll stop writing here. Maybe this repo is just a "garbage project" produced by me and AI, but the joy of free exploration is unforgettable: staring at the loss printing line by line, and sharing the thrill with a friend when the val loss dropped below 3.9 (on another seed).

Trivial, but deeply meaningful to me. The learning journey continues.

