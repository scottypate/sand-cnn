# Image Segmentation with TensorFlow

Convolutional Neural Network built using OpenCV and Tensorflow for [this](https://www.kaggle.com/c/tgs-salt-identification-challenge) Kaggle competition. 

This is an implementation of "U-Net" described in [this](https://arxiv.org/abs/1505.04597) paper. The name "U-Net" comes from the "U" shape of the convolution and upsampling layers in the network. The "left" side of the "U" are convolution layers in which image shrinkage occurs. The "right" side of the "U" are transposed convolutional layers in which image dilation occurs.


### Net.py
The `Net` class of this application is responsible for the structure and the flow of data through the neural network. This class allows training of a neural network with the `train` method and access to the saved results of the training session with the `predict` method.

### Convolution.py
The `Convolution` class of this application is responsible for the operations that occur on the data within the structure defined in the `Net.py` class. This includes the definition of activation functions and the 


### Convolution animations

![](https://media.giphy.com/media/i4NjAwytgIRDW/giphy.gif)

_N.B.: Blue maps are inputs, and cyan maps are outputs._

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="150px" src="gif/no_padding_no_strides.gif"></td>
    <td><img width="150px" src="gif/arbitrary_padding_no_strides.gif"></td>
    <td><img width="150px" src="gif/same_padding_no_strides.gif"></td>
    <td><img width="150px" src="gif/full_padding_no_strides.gif"></td>
  </tr>
  <tr>
    <td>No padding, no strides</td>
    <td>Arbitrary padding, no strides</td>
    <td>Half padding, no strides</td>
    <td>Full padding, no strides</td>
  </tr>
  <tr>
    <td><img width="150px" src="gif/no_padding_strides.gif"></td>
    <td><img width="150px" src="gif/padding_strides.gif"></td>
    <td><img width="150px" src="gif/padding_strides_odd.gif"></td>
    <td></td>
  </tr>
  <tr>
    <td>No padding, strides</td>
    <td>Padding, strides</td>
    <td>Padding, strides (odd)</td>
    <td></td>
  </tr>
</table>

### Transposed convolution animations

_N.B.: Blue maps are inputs, and cyan maps are outputs._

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="150px" src="gif/no_padding_no_strides_transposed.gif"></td>
    <td><img width="150px" src="gif/arbitrary_padding_no_strides_transposed.gif"></td>
    <td><img width="150px" src="gif/same_padding_no_strides_transposed.gif"></td>
    <td><img width="150px" src="gif/full_padding_no_strides_transposed.gif"></td>
  </tr>
  <tr>
    <td>No padding, no strides, transposed</td>
    <td>Arbitrary padding, no strides, transposed</td>
    <td>Half padding, no strides, transposed</td>
    <td>Full padding, no strides, transposed</td>
  </tr>
  <tr>
    <td><img width="150px" src="gif/no_padding_strides_transposed.gif"></td>
    <td><img width="150px" src="gif/padding_strides_transposed.gif"></td>
    <td><img width="150px" src="gif/padding_strides_odd_transposed.gif"></td>
    <td></td>
  </tr>
  <tr>
    <td>No padding, strides, transposed</td>
    <td>Padding, strides, transposed</td>
    <td>Padding, strides, transposed (odd)</td>
    <td></td>
  </tr>
</table>

### Acknowledgements

Convolution animations used from the following source
* \[1\] Vincent Dumoulin, Francesco Visin - [A guide to convolution arithmetic
  for deep learning](https://arxiv.org/abs/1603.07285)
  ([BibTeX](https://gist.github.com/fvisin/165ca9935392fa9600a6c94664a01214))