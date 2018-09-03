import tensorflow as tf

# Class for convolutional layers. This is an implementation of the
# U-net architecture. Left and right refer to corresponding "sides" of the
# process. The left side of the process will cause the image to shrink. 
# The right side of the process uses transpositional convolution to counter
# the shrinkage (also called deconvolution).
class Convolution():
    
    def __init__(self, filter_height, filter_width, in_channels, out_channels):
        '''
        Return a tensor of random normally distributed values to use as image filter
        Args:
            filter_height: 
            filter_width:
            in_channels:
            out_channels:
        Returns:
            A filter/kernel tensor for use in convolution.
        '''
        self.filter = tf.Variable(
            tf.random_normal(
                shape=[filter_height, filter_width, in_channels, out_channels],
                mean=0.0,
                stddev=0.05,
                dtype=tf.float32
            )
        )

    def relu(self, features):
        '''
        Computes rectified linear unit activations
        Args:
            features: tensor
        Returns:
            A tensor of same type as features
        '''
        return tf.nn.relu(features=features)

    def tensor_to_float(self, features): 
        '''
        Cast the tensor data that is greater than 0 to float
        Args:
            features: tensor
        Returns:
            tensor with non-zero values cast to float
        '''
        return tf.cast(tf.greater(features,0),dtype=tf.float32)

    def tf_softmax(self, logits):
        '''
        Computes the softmax activations
        Args:
            logits: A non-empty tensor.
            axis: The dimension softmax would be performed on.
                The default is -1 which indicates the last dimension.
            name: A name for the operation (optional).
        Returns:
            A tensor with same shape as logits.
        ''' 
        return tf.nn.softmax(
            logits=logits,
            axis=axis,
            name=name
        )

    def left_feed_forward(self, inputs, stride=1, dilate=1, data_format='NHWC'):
        '''
        The process to convolve the image and generate output for the layer
        Args:
            input: A tensor that represents the image
            stride: 1-D tensor of length 4. The amount to slide the filter overlay on the image.
                The order of the dimensions is determined by data_format.
            dilate: The dilation factor for each dimension of input.
            data_format: 'NHWC' by default which is [batch, height, width, channels]
        Returns:
            A tensor with activations
        '''
        self.inputs = inputs
        self.layer = tf.nn.conv2d(
            input=inputs,
            filter=self.filter,
            strides=[1, stride, stride, 1],
            padding='SAME',
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            dilations=[1, dilate, dilate, 1]
        )

        return self.relu(self.layer)

    def right_feed_forward(self, inputs, stride=2, data_format='NHWC'):
        '''
        The process to convolve the image and generate output for the layer
        Args:
            inputs: A tensor that represents the image
            stride: 1-D tensor of length 4. The amount to slide the filter overlay on the image.
                The order of the dimensions is determined by data_format.
            dilate: The dilation factor for each dimension of input.
            data_format: 'NHWC' by default which is [batch, height, width, channels]
            output: Shape of the transpositional convolution operation
        Returns:
            A tensor with activations
        '''
        current_shape = inputs.shape
        batch_size = tf.shape(inputs)[0]
        output_height = int(current_shape[1].value * 2)
        output_width = int(current_shape[2].value * 2)
        output_channels = int(current_shape[3].value / 2)
        output_shape = [batch_size, output_height, output_width, output_channels]
        self.layer = tf.nn.conv2d_transpose(
            value=inputs,
            filter=self.filter,
            output_shape=output_shape,
            strides=[1, stride, stride, 1],
            padding='SAME'
        )

        return self.relu(self.layer)