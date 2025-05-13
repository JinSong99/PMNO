# Physics guided multi-step neural operator predictor for physical systems
Neural operators, which aim to approximate mappings between infinite-dimensional function spaces, have been widely applied in the simulation and prediction of physical systems. However, the limited representational capacity of network architectures, combined with their heavy reliance on large-scale data, often hinder effective training and result in poor extrapolation performance. Inspired by traditional numerical methods, we propose the physics guided multi-step neural operator (PMNO) architecture to address challenges in long-horizon prediction of complex physical systems. Distinct from general operator learning methods, the PMNO framework replaces the single-step input with multi-step historical data in the forward pass and introduces an implicit time-stepping scheme based on the Backward Differentiation Formula (BDF) during backpropagation. This design not only strengthens the model's extrapolation capacity but also facilitates more efficient and stable training with fewer data samples, especially for long-term predictions. Meanwhile, a causal training strategy is employed to circumvent the need for multi-stage training and to ensure efficient end-to-end optimization. The neural operator architecture possesses resolution-invariant properties, enabling the trained model to perform fast extrapolation on arbitrary spatial resolutions. We demonstrate the superior predictive performance of PMNO predictor across a diverse range of physical systems, including 2D linear system, modeling over irregular domain, complex-valued wave dynamics, and reaction-diffusion processes. Depending on the specific problem setting, various neural operator architectures, including FNO, DeepONet, and their variants, can be seamlessly integrated into the PMNO framework.

## Highlights
- The multi-step neural operator structures enhances the temporal dependency across different time steps, thereby improving the network's expressive capacity for capturing complex dynamical behaviors.
- BDF guided training paradigm enhances the model’s extrapolation capability, and facilitates more efficient and stable training with limited data. And the causal training strategy is employed to circumvent the need for multi-stage training and to ensure efficient end-to-end optimization.
- Despite its architectural simplicity, PMNO predictor performs fast extrapolation on arbitrary spatial resolutions, and achieves superior predictive performance across a diverse range of physical systems, including 2D linear system, modeling over irregular domain, complex-valued wave dynamics, and reaction-diffusion processes.

## Datasets
Due to the file size limit, We provide the datasets used in this projects in google drive [link](https://drive.google.com/drive/folders/18ZiM3cimjfpeXvgSD0fHQOjgOnRONRpT?usp=sharing).

## Pretrained Model
We provide a pretrained model for quick evaluation and extrapolation, attached in the [link](https://drive.google.com/drive/folders/18ZiM3cimjfpeXvgSD0fHQOjgOnRONRpT?usp=sharing).
- Filename: `./model/checkpoint.pt`


