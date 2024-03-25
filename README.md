# Fast Design of Plasmonic Metasurfaces Enabled by Deep Learning

## Description
Metasurfaces is an emerging field that enables the manipulation of light by an ultra-thin structure composed of sub-wavelength antennae and fulfills an important requirement for miniaturized optical elements. Finding a new design for a metasurface or optimizing an existing design for a desired functionality is a computationally expensive and time consuming process as it is based on an iterative process of trial and error. We propose a deep learning (DL) architecture dubbed bidirectional autoencoder for nanophotonic metasurface design via a template search methodology. In contrast with the earlier approaches based on DL, our methodology addresses optimization in the space of multiple metasurface topologies instead of just one, in order to tackle the one to many mapping problem of inverse design. We demonstrate the creation of a Geometry and Parameter Space Library (GPSL) of metasurface designs with their corresponding optical response using our DL model. This GPSL acts as a universal design and response space for the optimization. As an example application, we use our methodology to design a multi-band gap-plasmon based half-wave plate metasurface. Through this example, we demonstrate the power of our technique in addressing the non-uniqueness problem of common inverse design. Our network converges aptly to multiple metasurface topologies for the desired optical response with a low mean absolute error between desired optical response and the optical response of topologies searched. Our proposed technique would enable fast and accurate design and optimization of various kinds of metasurfaces with different functionalities.

## Code
This repository contains the template code used in the research article. The code includes : create_dataset.py, models.py and train.py. It is written for one of the metasurface dataset and can be extended to other datasets. The code demonstrates the implementation of autoencoder for the design of plasmonic metasurfaces. The dataset was generated using simulation software, COMSOL.

The program is pure-Python and runs on GPU. Additionally, the following libraries are used

- numpy
- pandas
- pytorch

## Article Information

[Link to Research Article](https://iopscience.iop.org/article/10.1088/1361-6463/abb33c/meta?casa_token=k-IKQCH2uVwAAAAA:EkxEYh0JemFOo218FVkzKru3KJpz4dZLyoP2OczqMxjkU6f_LQBaXYGtxBWHC6rLuGLmu_0vMKnh9YoNjyie1AaoDe9Zxw)

- **Authors:** Abhishek Mall, Abhijeet Patil, Dipesh Tamboli, Amit Sethi, Anshuman Kumar
- **Publication Date:** 2020/10/1
- **Journal:** Journal of Physics D: Applied Physics
- **Volume:** 53
- **Issue:** 49
- **Pages:** 49LT01
- **Publisher:** IOP Publishing Ltd


## Citation

If you use this code or the research in your work, please cite the following:

```plaintext
@article{mall2020fast,
  title={Fast design of plasmonic metasurfaces enabled by deep learning},
  author={Mall, Abhishek and Patil, Abhijeet and Tamboli, Dipesh and Sethi, Amit and Kumar, Anshuman},
  journal={Journal of Physics D: Applied Physics},
  volume={53},
  number={49},
  pages={49LT01},
  year={2020},
  publisher={IOP Publishing}
}


