# Vital Signs Simulation During Anesthesia using convolutional neural networks
Project submited to EMBC 2020
We proide the code and the results obtained on the dataset provided by the article.
Also, we provide the artificial neural networks parameters for testing.

## Abstract:


In the past years, machine learning techniques have been obtaining promising results in attempts at anesthesia automatization. The most recent study by Lee et al. demonstrated that machine learning performed better than target-controlled infusion (TCI) in predicting hypnosis level measured by Bispectral Index (BIS) values. We aimed to reach acceptable accuracy of vital signs prediction during anesthesia using convolutional neural networks and an open vital sign database in the present work. We use BIS values, heart rate (HR), mean blood pressure (MBP), propofol, and remifentanil infusion rates for BIS, HR, and MBP prediction during anesthesia. This work accomplished a good prediction of all three vital signs using a convolutional neural network architecture (CNN). We compare our method to previous studies obtaining similar error metrics,  resulting in the acceptable generation of BIS values, heart rate, and mean blood pressure. In addition, our method is more adaptable for multiple inputs, simultaneously handling many vital signs time-series while maintaining comparable performance on the BIS metric and less parameters.


## Requirements

- Python 3.6
- [Keras 2.4.3](https://pypi.org/project/Keras/)


## Usage

First, install prerequisites

```
$ pip install keras==2.4.3
```




	

If you re-use this work, please cite:

```
@article{Felizardo2020,
  Title                    = {Vital Signs Simulation During Anesthesia: a Deep Learning Approach},
  Author                   = {Felizardo, Leonardo},
  journal                  = {},
  Year                     = {2020},
  volume                   = {},
  number                   = {},
  pages                    = {},
  url                      = {https://github.com/leokan92/anesthesia}
}
```







