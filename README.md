# FW-Net
FW-Net for image denoising

Our Configurations: CUDA-8.0, cudnn-v5, Python 2.7, GPU 1080Ti.

I trained these models on windows servers. I will release the DLLs of windows caffe asap.

This code is based on Linux Caffe, including training and testing parts.

Our models are under the folder "pretrained model".

You can check the details in our paper: (https://arxiv.org/abs/1802.10252)

This code will be continually updated.
If you have any question, please feel free to contact me.

E-Mail: sunk@mail.ustc.edu.cn

if it helps your research, please cite our paper:

@misc{1802.10252,
  author = {Ke Sun and Zhangyang Wang and Dong Liu and Runsheng Liu},
  title = {${L}_p$-{N}orm {C}onstrained {C}oding {W}ith {F}rank-{W}olfe {N}etwork},
  year = {2018},
  eprint = {1802.10252},
  note = {arXiv:1802.10252v1}
}

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by Berkeley AI Research ([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning Center (BVLC) and community contributors.

- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }

