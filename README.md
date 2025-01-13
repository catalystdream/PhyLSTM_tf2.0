# PhyLSTM_tf2.0
Physics Informed LSTM for nonlienar structures using tensorflow2.0

This repository provides a TensorFlow 2.0-compatible implementation of the Physics-Informed LSTM (PhyLSTM) model, originally developed by [zhry10](https://github.com/zhry10).The model is designed for metamodeling nonlinear structures using physics-informed machine learning techniques.
## Original work
This implementation is based on the following paper: [Physics-informed multi-LSTM networks for metamodeling of nonlinear structures](https://www.sciencedirect.com/science/article/pii/S0045782520304114), Authors: Ruiyang Zhanga, Yang Liub, Hao Sun

Original code repository: [zhry10/PhyLSTM](https://github.com/zhry10/PhyLSTM)

This work extends their methodology by adapting the code to TensorFlow 2.0, making it compatible with modern frameworks and ensuring better support for current machine learning workflows.

## Key Features
	•	Fully reimplemented PhyLSTM code for TensorFlow 2.0.
	•	Physics-informed loss functions for nonlinear structural dynamics.
	•	Improved compatibility with modern TensorFlow/Keras APIs.
	•	Retains the functionality and principles outlined in the original implementation.
## What’s New in This Version?
	1.	TensorFlow Compatibility: The code is rewritten using TensorFlow 2.x, replacing deprecated TensorFlow 1.x features.
	2.	Eager Execution: Updated to leverage TensorFlow 2.0’s eager execution mode for easier debugging and development.
	3.	Improved Documentation: Added inline comments and detailed documentation to improve usability.
	4.	Enhanced Modular Design: Code refactored to be more modular, facilitating further customization and experimentation.
## Dataset
The model is designed to work with time-series datasets for nonlinear structural systems. The example dataset used in the original implementation is included here for the $PhyLSTM^2$ implementation. For $PhyLSTM^3$ the same dataset is used to make the code run. You may refer to the original repository for preprocessing guidelines or use your dataset.

## Acknowledgment
This repository is heavily based on the original work by [zhry10](https://github.com/zhry10). If you use this code or find it helpful, please consider citing their paper:
<pre>
@article{yang2020physics,
  title={Physics-informed multi-LSTM networks for metamodeling of nonlinear structures},
  author={Yang, Zhuoran and Ma, Xing and Du, Wenyang and Beer, Michael},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={373},
  pages={113500},
  year={2021},
  publisher={Elsevier}
}
</pre>

## License
This project is distributed under the same license as the original repository. Please ensure proper acknowledgment of the original authors in any derivative works.
## Contact
If you have any questions or suggestions, feel free to contact me at: 
GitHub: [catalystdream](https://github.com/catalystdream)
