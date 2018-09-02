import tensorflow as tf

from Convolution import Convolution

class Net:

    def define_structure(self):
        # Construct the layers of the network. The pixel growth and shrinkage
        # is dependent on both the size of the input image and the number of
        # convolution layers. 
        self.conv_1_1 = Convolution(3, 3, 1, 3)
        self.conv_1_2 = Convolution(3, 3, 3, 3)
        self.conv_1_3 = Convolution(3, 3, 3, 3)

        self.conv_2_1 = Convolution(3, 3, 3, 6)
        self.conv_2_2 = Convolution(3, 3, 6, 6)
        self.conv_2_3 = Convolution(3, 3, 6, 6)

        self.conv_3_1 = Convolution(3, 3, 6, 12)
        self.conv_3_2 = Convolution(3, 3, 12, 12)
        self.conv_3_3 = Convolution(3, 3, 12, 12)

        self.conv_4_1 = Convolution(3, 3, 12, 24)
        self.conv_4_2 = Convolution(3, 3, 24, 24)
        self.conv_4_3 = Convolution(3, 3, 24, 24)

        self.conv_5_1 = Convolution(3, 3, 24, 48)
        self.conv_5_2 = Convolution(3, 3, 48, 48)
        self.conv_5_3 = Convolution(3, 3, 48, 24)

        self.conv_6_1 = Convolution(3, 3, 24, 48)
        self.conv_6_2 = Convolution(3, 3, 24, 24)
        self.conv_6_3 = Convolution(3, 3, 24, 12)

        self.conv_7_1 = Convolution(3, 3, 12, 24)
        self.conv_7_2 = Convolution(3, 3, 12, 12)
        self.conv_7_3 = Convolution(3, 3, 12, 6)

        self.conv_8_1 = Convolution(3, 3, 6, 12)
        self.conv_8_2 = Convolution(3, 3, 6, 6)
        self.conv_8_3 = Convolution(3, 3, 6, 3)

        self.conv_9_1 = Convolution(3, 3, 3, 6)
        self.conv_9_2 = Convolution(3, 3, 3, 3)
        self.conv_9_3 = Convolution(3, 3, 3, 3)

        self.conv_10_out = Convolution(3, 3, 3, 1)

    def placeholders(self, batch_size):
        self.features = tf.placeholder(shape=[batch_size, 256, 256, 1],dtype=tf.float32)
        self.labels = tf.placeholder(shape=[batch_size, 256, 256, 1],dtype=tf.float32)

    def connect_layers(self, features):
        # Connect the layers to each other 
        layer_1_1 = self.conv_1_1.left_feed_forward(self.features)
        layer_1_2 = self.conv_1_2.left_feed_forward(layer_1_1)
        layer_1_3 = self.conv_1_3.left_feed_forward(layer_1_2)

        layer_2_Input = tf.nn.max_pool(
            value=layer_1_3,
            ksize=[1,2,2,1],
            strides=[1,2,2,1],
            padding='VALID'
        )
        layer_2_1 = self.conv_2_1.left_feed_forward(layer_2_Input)
        layer_2_2 = self.conv_2_2.left_feed_forward(layer_2_1)
        layer_2_3 = self.conv_2_3.left_feed_forward(layer_2_2)

        layer_3_Input = tf.nn.max_pool(
            value=layer_2_3,
            ksize=[1,2,2,1],
            strides=[1,2,2,1],
            padding='VALID'
        )
        layer_3_1 = self.conv_3_1.left_feed_forward(layer_3_Input)
        layer_3_2 = self.conv_3_2.left_feed_forward(layer_3_1)
        layer_3_3 = self.conv_3_3.left_feed_forward(layer_3_2)

        layer_4_Input = tf.nn.max_pool(
            value=layer_3_3,
            ksize=[1,2,2,1],
            strides=[1,2,2,1],
            padding='VALID'
        )
        layer_4_1 = self.conv_4_1.left_feed_forward(layer_4_Input)
        layer_4_2 = self.conv_4_2.left_feed_forward(layer_4_1)
        layer_4_3 = self.conv_4_3.left_feed_forward(layer_4_2)

        layer_5_Input = tf.nn.max_pool(
            value=layer_4_3,
            ksize=[1,2,2,1],
            strides=[1,2,2,1],
            padding='VALID'
        )
        layer_5_1 = self.conv_5_1.left_feed_forward(layer_5_Input)
        layer_5_2 = self.conv_5_2.left_feed_forward(layer_5_1)
        layer_5_3 = self.conv_5_3.left_feed_forward(layer_5_2)

        layer_6_Input = tf.concat(
            values=[layer_5_3,layer_5_Input],
            axis=3
        )
        layer_6_1 = self.conv_6_1.right_feed_forward(layer_6_Input)
        layer_6_2 = self.conv_6_2.left_feed_forward(layer_6_1)
        layer_6_3 = self.conv_6_3.left_feed_forward(layer_6_2)

        layer_7_Input = tf.concat(
            values=[layer_6_3,layer_4_Input],
            axis=3
        )
        layer_7_1 = self.conv_7_1.right_feed_forward(layer_7_Input)
        layer_7_2 = self.conv_7_2.left_feed_forward(layer_7_1)
        layer_7_3 = self.conv_7_3.left_feed_forward(layer_7_2)

        layer_8_Input = tf.concat(
            values=[layer_7_3,layer_3_Input],
            axis=3
        )
        layer_8_1 = self.conv_8_1.right_feed_forward(layer_8_Input)
        layer_8_2 = self.conv_8_2.left_feed_forward(layer_8_1)
        layer_8_3 = self.conv_8_3.left_feed_forward(layer_8_2)

        layer_9_Input = tf.concat(
            values=[layer_8_3,layer_2_Input],
            axis=3
        )
        layer_9_1 = self.conv_9_1.right_feed_forward(layer_9_Input)
        layer_9_2 = self.conv_9_2.left_feed_forward(layer_9_1)
        layer_9_3 = self.conv_9_3.left_feed_forward(layer_9_2)

        self.layer_10 = self.conv_10_out.left_feed_forward(layer_9_3)

    def execute(self, batch_size, features, labels, n_epochs):

        self.define_structure()
        self.placeholders(batch_size=batch_size)
        self.connect_layers(features=self.features)

        loss = tf.reduce_mean(tf.square(self.layer_10 - self.labels))
        training_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for iter in range(n_epochs):
                for current_batch_index in range(0,len(features),batch_size):
                    current_batch = features[current_batch_index:current_batch_index+batch_size]
                    current_label = labels[current_batch_index:current_batch_index+batch_size]
                    sess_results = sess.run([loss, training_optimizer], feed_dict={self.features:current_batch,self.labels:current_label})
                
        return sess_results