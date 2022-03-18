# FENeRF: Face Editing in Radiance Fields<br><sub>Official PyTorch implementation</sub>

![Teaser image](./teaser.png)

**FENeRF: Face Editing in Radiance Fields**<br>
Jingxiang Sun, Xuan Wang, Yong Zhang, Xiaoyu Li, Qi Zhang, Yebin Liu and Jue Wang
<br>
https://mrtornado24.github.io/FENeRF/<br>

Abstract: *Previous portrait image generation methods roughly fall into two categories: 2D GANs and 3D-aware GANs. 2D GANs can generate high fidelity portraits but with low view consistency. 3D-aware GAN methods can maintain view consistency but their generated images are not locally editable. To overcome these limitations, we propose FENeRF, a 3D-aware generator that can produce view-consistent and locally-editable portrait images. Our method uses two decoupled latent codes to generate corresponding facial semantics and texture in a spatial aligned 3D volume with shared geometry. Benefiting from such underlying 3D representation, FENeRF can jointly render the boundary-aligned image and semantic mask and use the semantic mask to edit the 3D volume via GAN inversion. We further show such 3D representation can be learned from widely available monocular image and semantic mask pairs. Moreover, we reveal that joint learning semantics and texture helps to generate finer geometry. Our experiments demonstrate that FENeRF outperforms state-of-the-art methods in various face editing tasks.*

## Citation

```
@article{sun2021fenerf,
  title={FENeRF: Face Editing in Neural Radiance Fields},
  author={Sun, Jingxiang and Wang, Xuan and Zhang, Yong and Li, Xiaoyu and Zhang, Qi and Liu, Yebin and Wang, Jue},
  journal={arXiv preprint arXiv:2111.15490},
  year={2021}
}
```
