# VLM Embed with AE to 3D Notes

## Why text embeddings sometimes fail to map meaningfully in image space

If your **3D text embeddings don’t line up well with the image embeddings**, even though the original 1536-dim ones do, then something subtle is happening in the autoencoder training.  

Here are the likely reasons:

---

### 1. **Training only on image embeddings biases the AE**

- Your autoencoder is learning to compress the *distribution of image embeddings* into 3D.  
- Text embeddings, even though aligned in the original 1536D space, occupy slightly different *submanifolds*.  
- When you project them with an AE trained **only on images**, it hasn’t learned to preserve those regions, so text points get distorted.

👉 Analogy: Imagine you compress satellite photos of Europe into 3D. The system learns Europe’s geography. Then you drop in Africa (text embeddings): the system never learned how to handle those coordinates, so they get squashed in strange ways.

---

### 2. **The AE is non-linear, not linear**

- PCA would preserve global variance directions of the whole embedding cloud, so text and images would still roughly align.  
- An autoencoder, being non-linear, can warp the space a lot to optimize reconstruction *only for images*.  
- That “warping” is what breaks the alignment for text.

---

### 3. **Objective mismatch**

- Multimodal models are trained on **cosine similarity** (angle preservation).  
- Your AE is trained on **L2 reconstruction loss**, which doesn’t necessarily preserve *cosine similarity structure*.  
- So after encoding, two points that were close by cosine in 1536D may not be close in 3D, unless they follow the same distribution you trained on (images).

---

### 4. **Dimensionality bottleneck (3D is very aggressive)**

- Going from 1536 → 3 is a huge compression.  
- The AE will prioritize the major modes of variance in the **image distribution**, but text variance directions may differ.  
- That means the few dimensions left don’t capture text structure well.

---

✅ **What you can do to fix this:**

1. **Train on both text + image embeddings together.**  
   That way the AE learns a projection that preserves *shared* structure.

2. **Switch to PCA or t-SNE/UMAP for exploration.**  
   These methods don’t warp in modality-specific ways. A PCA to 50D + AE to 3D can sometimes balance things better.

3. **Change the AE loss to preserve similarity.**  
   Instead of only reconstruction, add a **cosine similarity preservation term** or train it as a *metric learning autoencoder*.  
   That way, embeddings that are close in 1536D cosine remain close in 3D.

---

So, in short:  
It doesn’t work well in your case because the AE learned to preserve the geometry of *images only*, and it warped the space in a way that doesn’t generalize to text. The shared space property is preserved only in the original embedding space — your AE disrupted it by training on a subset of modalities.

---
