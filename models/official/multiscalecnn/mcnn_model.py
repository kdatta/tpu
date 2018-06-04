import model
import tensorflow as tf

def create_mcnn_model():
  return MCNNModel('NCHW', 8, 0.001)

class MCNNModel(model.Model):
    """MCNN model."""

    def __init__(self, data_format, batchsize, learning_rate):
        super(MCNNModel, self).__init__('mcnn', 1024*1280, batchsize, learning_rate, 6)
        self.data_format = data_format
        self.conv_counter = 0
        self.preconv_counter = 0
        self.pool_counter = 0
        self.par_conv_pool_counter = 0
        self.ip_counter = 0
        self.batchsize = batchsize
        self.lr = learning_rate

    def print_activations(self, t):
        print(t.op.name, ' ', t.get_shape().as_list())

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

    def preconv_(self, prefix, inp, kH, kW, dH, dW):
        self.preconv_counter += 1
        name = prefix + '_preConv_' + str(self.preconv_counter)
        with tf.name_scope(name) as scope:
            if self.data_format == 'NCHW':
                ksize = [1, 1, kH, kW]
                strides = [1, 1, dH, dW]
            else:
                ksize = [1, kH, kW, 1]
                strides = [1, dH, dW, 1]

            return tf.nn.max_pool(inp,
                                  ksize=ksize,
                                  strides=strides,
                                  padding='VALID',
                                  data_format=self.data_format,
                                  name=name)

    def conv_(self, name_prefix, inp, nIn, nOut, kH, kW, dH, dW, padType, padding, pV):
        self.conv_counter += 1
        name = name_prefix + '_conv_' + str(self.conv_counter)
        #with tf.name_scope(name) as scope:
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if self.data_format == 'NCHW':
                ksize = [1, 1, kH, kW]
                strides = [1, 1, dH, dW]
            else:
                ksize = [1, kH, kW, 1]
                strides = [1, dH, dW, 1]
            kernel_shape = [kH, kW, nIn, nOut]
            kernel = tf.get_variable(name,
                                     shape=kernel_shape, 
                                     initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.Variable(tf.zeros(shape=[nOut], dtype=tf.float32),
                                 name='biases',
                                 trainable=True)
            if padding == True:
                if self.data_format == 'NCHW':
                    padded_input = tf.pad(inp, [[0, 0], [0, 0], [pV, pV], [pV, pV]], "CONSTANT")
                else:
                    #NHWC
                    padded_input = tf.pad(inp, [[0, 0], [pV, pV], [pV, pV], [0, 0]], "CONSTANT")
            conv = tf.nn.conv2d(inp,
                                kernel,
                                strides,
                                padding=padType, 
                                use_cudnn_on_gpu=False,
                                data_format=self.data_format)
            bias = tf.nn.bias_add(conv,
                                  biases,
                                  data_format=self.data_format)
            conv1 = tf.nn.relu(bias)
            #conv_bn = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=True)
            #conv1 = tf.nn.dropout(conv1, 0.1)
            #self.variable_summaries(kernel)
            return conv1

    def max_pool_(self, prefix, inp, kH, kW, dH, dW):
        name = prefix + '_pool_' + str(self.pool_counter)
        self.pool_counter += 1
        with tf.name_scope(name) as scope:
            if self.data_format == 'NCHW':
                ksize = [1, 1, kH, kW]
                strides = [1, 1, dH, dW]
            else:
                ksize = [1, kH, kW, 1]
                strides = [1, dH, dW, 1]
            return tf.nn.max_pool(inp,
                                  ksize=ksize,
                                  strides=strides,
                                  padding='SAME',
                                  data_format=self.data_format,
                                  name=name)

    def _inner_product(self, prefix, inp, nIn, nOut): 
        name = prefix + '_ip_' + str(self.ip_counter) 
        self.ip_counter += 1 
        #with tf.name_scope(name) as scope:  
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:  
            #  kernel_shape = [kH, kW, nIn, nOut]
            # kernel = xavier_initializer(kernel_shape, nIn, nOut, factor=2.0, mode='FAN_AVG',
            #                                                           dtype=dtypes.float32)
                
            # kernel = tf.get_variable(tf.truncated_normal([nIn, nOut], 
            #                                                       dtype=tf.float32, 
            #                                                       stddev=1e-2), name='weights') 
            kernel_shape = [nIn, nOut]
            kernel = tf.get_variable(name,
                                     shape=kernel_shape, 
                                     initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.Variable(tf.zeros(shape=[nOut], dtype=tf.float32),
                                 name='biases',
                                 trainable=True)
            fc = tf.matmul(inp, kernel) + biases
            #fc = tf.matmul(inp, kernel, transpose_a=False, transpose_b=True) + biases 
            # fc_bn = tf.contrib.layers.batch_norm(fc, center=True, scale=True, is_training=True)
            #self.variable_summaries(kernel)
            return fc

    def pc_par_conv_pool(self, prefix, inp, kHpc, kWpc, dHpc, dWpc, nIn,
                                             nOut, kH, kW, dH, dW, kHp, kWp, dHp, dWp, padType, padding, pV):

        self.par_conv_pool_counter +=1
        #Preconv pooling
        preConv_pool = self.preconv_(prefix, inp, kHpc, kWpc, dHpc, dWpc)
        #self.print_activations(preConv_pool)
        #conv1 + relu1
        conv1 = self.conv_(prefix+'_conv1_', preConv_pool, nIn, nOut, kH, kW, dH, dW, padType, padding, pV)
        #self.print_activations(conv1)
        #conv1-1 + relu1-1
        conv2 = self.conv_(prefix+'_conv2_', conv1, nOut, nOut, kH, kW, dH, dW, padType, padding, pV)
        #self.print_activations(conv2)
        #conv1-2 + relu1-2
        conv3 = self.conv_(prefix+'_conv3_', conv2, nOut, nOut, kH, kW, dH, dW, padType, padding, pV)
        #self.print_activations(conv3)

        #Pooling
        pool_ = self.max_pool_(prefix, conv3, kHp, kWp, dHp, dWp)
        #self.print_activations(pool_)

        return pool_

    def conv_pool(self, inp, nIn, nOut, kH, kW, dH, dW, kHp, kWp, dHp, dWp, padType, padding, pV):

        name='highres'
        with tf.name_scope(name) as scope:
            #conv1 + relu1
            conv1 = self.conv_(name, inp, nIn, nOut, kH, kW, dH, dW, padType, padding, pV)
            #self.print_activations(conv1)
            #conv1-1 + relu1-1
            conv2 = self.conv_(name, conv1, nOut, nOut, kH, kW, dH, dW, padType, padding, pV)
            #self.print_activations(conv2)
            #conv1-2 + relu1-2
            conv3 = self.conv_(name, conv2, nOut, nOut, kH, kW, dH, dW, padType, padding, pV)
            #self.print_activations(conv3)

            #Pooling
            pool_ = self.max_pool_(name, conv3, kHp, kWp, dHp, dWp)
            #self.print_activations(pool_)

        return pool_

    #def inference(images):
    def add_inference(self, batchsize, images):
        print "******************************"
        print "******************************"
        print "inside add_inference"
        print "******************************"
        print "******************************"
        if self.data_format == 'NCHW':
            images = tf.reshape(images, shape=[-1,  3, 1024 , 1280])
        else:
            images = tf.reshape(images, shape=[-1, 1024, 1280, 3])

        nIn = 3 #Number of input channels
        col1 = self.conv_pool(images, nIn, 16, 5, 5, 1, 1, 64, 64, 64, 64, 'SAME', True, 2)
        col2 = self.pc_par_conv_pool('res_2', images, 2, 2, 2, 2, nIn, 16, 5, 5, 1, 1, 32, 32, 32, 32, 'SAME', True, 2)
        col3 = self.pc_par_conv_pool('res_4',images, 4, 4, 4, 4, nIn, 16, 5, 5, 1, 1, 16, 16, 16, 16, 'SAME', True, 2)
        col4 = self.pc_par_conv_pool('res_8',images, 8, 8, 8, 8, nIn, 32, 5, 5, 1, 1, 8, 8, 8, 8, 'SAME', True, 2)
        col5 = self.pc_par_conv_pool('res_16',images, 16, 16, 16, 16, nIn, 32, 5, 5, 1, 1, 4, 4, 4, 4, 'SAME', True, 2)
        col6 = self.pc_par_conv_pool('res_32',images, 32, 32, 32, 32, nIn, 32, 5, 5, 1, 1, 2, 2, 2, 2, 'SAME', True, 2)
        col7 = self.pc_par_conv_pool('res_64',images, 64, 64, 64, 64, nIn, 64, 5, 5, 1, 1, 1, 1, 1, 1, 'SAME', True, 2)
        #col0 = tf.zeros([4,16,16,20])
        #col1 = tf.zeros([4,16,16,20])
        #col2 = tf.zeros([batchsize,16,16,20])
        #col3 = tf.zeros([batchsize,16,16,20])
        #col4 = tf.zeros([batchsize,32,16,20])
        #col5 = tf.zeros([batchsize,32,16,20])
        #col6 = tf.zeros([batchsize,32,16,20])
        #col7 = tf.zeros([batchsize,64,16,20])

        #mergedPool
        if self.data_format == 'NCHW':
            col_merge = tf.concat([col1, col2, col3, col4, col5, col6, col7], 1)
        else:
            col_merge = tf.concat([col1, col2, col3, col4, col5, col6, col7], 3)
        #self.print_activations(col_merge)

        #mergedSummaryConv + relu-mergedSummaryConv
        mergedSummaryConv = self.conv_('mergedSummaryConv', col_merge, 208, 1024, 1, 1, 1, 1, 'SAME', False, 0)
        #self.print_activations(mergedSummaryConv)

        #poolMergedSummaryConv
        poolMergedSummaryConv = self.max_pool_('mergedSummaryConv', mergedSummaryConv, 2, 2, 2, 2)
        #self.print_activations(poolMergedSummaryConv)

        #ip0 + relulp0
        resh1 = tf.reshape(poolMergedSummaryConv, [-1, 1024 * 10 * 8])
        ip0 = self._inner_product('ip0', resh1, 1024 * 10 * 8, 512)
        relulp0 = tf.nn.relu(ip0)
        #self.print_activations(ip0)

        #kernel = tf.get_variable(tf.truncated_normal([512, 13], 
        kernel_shape = [13, 512]
        kernel = tf.get_variable('ip3_weights', shape=kernel_shape, 
                                                                initializer=tf.contrib.layers.xavier_initializer()) 
        biases = tf.Variable(tf.zeros(shape=[13], dtype=tf.float32),
                                                     name='biases',
                                                     trainable=True)
        ip3 = tf.matmul(ip0, kernel, transpose_a=False, transpose_b=True) + biases
        relulp1 = tf.nn.relu(ip3)

        self.batchsize = batchsize

        return ip3

    def loss_function(self, logits, labels, aux_logits):
        sparse_labels = tf.reshape(labels, [self.batchsize, 1]) 
        indices = tf.reshape(tf.range(self.batchsize), [self.batchsize, 1]) 
        concated = tf.concat(axis=1, values=[indices, sparse_labels])
        num_classes = logits[0].get_shape()[-1].value
        dense_labels = tf.sparse_to_dense(concated, [self.batchsize, num_classes], 1.0, 0.0)
        one_hot_labels = tf.cast(dense_labels, logits.dtype)
        label_smoothing = 0.1 
        smooth_positives = 1.0 - label_smoothing
        smooth_negatives = label_smoothing / num_classes
        one_hot_labels = one_hot_labels * smooth_positives + smooth_negatives
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
        weight=1.0
        weight = tf.convert_to_tensor(weight, dtype=logits.dtype.base_dtype, name='loss_weight')
        loss = tf.multiply(weight, tf.reduce_mean(cross_entropy), name='value')
        tf.summary.scalar('loss', loss)
        return loss

