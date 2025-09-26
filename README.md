
# HyperDiff: Hypergraph Guided Diffusion Model for 3D Human Pose Estimation [MMSP2025]

The PyTorch implementation for ["HyperDiff: Hypergraph Guided Diffusion Model for 3D Human Pose Estimation"](https://arxiv.org/abs/2508.14431) .
<p align="center"><img src="fig/overview.png", width="600" alt="" /></p>
<!-- <p align="center"><img src="fig/demo.gif", width="600"  alt="" /></p> -->


## Dependencies

Make sure you have the following dependencies installed (python):

* pytorch >= 0.4.0
* matplotlib=3.1.0
* einops
* timm
* tensorboard

You should download [MATLAB](https://www.mathworks.com/products/matlab-online.html) if you want to evaluate our model on MPI-INF-3DHP dataset.

## Datasets

Our model is evaluated on [Human3.6M](http://vision.imar.ro/human3.6m) and [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) datasets. 

### Human3.6M

We set up the Human3.6M dataset in the same way as [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md).  You can download the processed data from [here](https://drive.google.com/file/d/1FMgAf_I04GlweHMfgUKzB0CMwglxuwPe/view?usp=sharing).  `data_2d_h36m_gt.npz` is the ground truth of 2D keypoints. `data_2d_h36m_cpn_ft_h36m_dbb.npz` is the 2D keypoints obatined by [CPN](https://github.com/GengDavid/pytorch-cpn).  `data_3d_h36m.npz` is the ground truth of 3D human joints. Put them in the `./data` directory.

### MPI-INF-3DHP

We set up the MPI-INF-3DHP dataset following [P-STMO](https://github.com/paTRICK-swk/P-STMO). However, our training/testing data is different from theirs. They train and evaluate on 3D poses scaled to the height of the universal skeleton used by Human3.6M (officially called "univ_annot3"), while we use the ground truth 3D poses (officially called "annot3"). You can download our processed data from [here](https://drive.google.com/file/d/1zOM_CvLr4Ngv6Cupz1H-tt1A6bQPd_yg/view?usp=share_link). Put them in the `./data` directory. 
 

### Human3.6M

To evaluate our HyperDiff using the 2D keypoints obtained by CPN as inputs, please run:
```bash
python main.py -k cpn_ft_h36m_dbb -c checkpoint/model_h36m -gpu 0,1 --nolog --evaluate best_epoch.bin -num_proposals 1 -sampling_timesteps 1 -b 4
```

### MPI-INF-3DHP
To evaluate our HyperDiff using the ground truth 2D poses as inputs, please run:
```bash
python main_3dhp.py -c checkpoint/model_3dhp -gpu 0,1 --nolog --evaluate best_epoch.bin -num_proposals 1 -sampling_timesteps 1 -b 4
```
After that, the predicted 3D poses are saved in `./checkpoint`. To get the MPJPE, AUC, PCK metrics, you can evaluate the predictions by running a Matlab script `./3dhp_test/test_util/mpii_test_predictions_ori_py.m`.

## Training from scratch
Trained on 2*NVIDIA RTX 4090.
### Human3.6M
To train our model using the 2D keypoints obtained by CPN as inputs, please run:
```bash
python main.py -k cpn_ft_h36m_dbb -c checkpoint/model_h36m -gpu 0,1 --nolog
```

### MPI-INF-3DHP
To train our model using the ground truth 2D poses as inputs, please run:
```bash
python main_3dhp.py -c checkpoint/model_3dhp -gpu 0,1 --nolog
```

### Pretrained Models
[Baidu Netdisk](https://pan.baidu.com/s/1oiI1vFOkjgAL-4FpGbDJ2w?pwd=a7ka)

[Google Drive](https://drive.google.com/file/d/1WlMr0vlXhllDiqNq0eb3FnBXs447YILG/view?usp=drive_link)

## Reference
```
@article{han2025hyperdiff,
  title={HyperDiff: Hypergraph Guided Diffusion Model for 3D Human Pose Estimation},
  author={Han, Bing and Huang, Yuhua and Gao, Pan},
  journal={arXiv e-prints},
  pages={arXiv--2508},
  year={2025}
}
```

## Acknowledgement
Our code refers to the following repositories.

* [D3DP](https://github.com/paTRICK-swk/D3DP)
* [Selective-HCN](https://github.com/CFM-MSG/Code_SelectiveHCN)

We thank the authors for releasing their codes.
>>>>>>> 4421b04 (first commit)
