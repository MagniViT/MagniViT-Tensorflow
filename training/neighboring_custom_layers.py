import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import  Dense,multiply



class NeighborAggregator(Layer):
    """
    Aggregation of neighborhood information
    This layer is responsible for aggregatting the neighborhood information of the attentin matrix through the
    element-wise multiplication with an adjacency matrix. Every row of the produced
    matrix is averaged to produce a single attention score.
    # Arguments
        output_dim:            positive integer, dimensionality of the output space
    # Input shape
        2D tensor with shape: (n, n)
        2d tensor with shape: (None, None) correspoding to the adjacency matrix
    # Output shape
        2D tensor with shape: (1, units) corresponding to the attention coefficients of every instance in the bag
    """

    def __init__(self, output_dim, k,**kwargs):
        self.output_dim = output_dim
        self.k=k
        super(NeighborAggregator, self).__init__(**kwargs)

    @tf.function
    def get_affinity(self, Idx):
            """
            Create the adjacency matrix of each bag based on the euclidean distances between the patches
            Parameters
            ----------
            Idx:   a list of indices of the closest neighbors of every image
            Returns
            -------
            affinity:  an nxn np.ndarray that contains the neighborhood information for every patch.
            """
            columns = tf.experimental.numpy.ravel(Idx)

            rows = tf.experimental.numpy.ravel(tf.map_fn(fn=lambda t: tf.repeat(t, repeats=self.k+1) , elems=tf.range(tf.shape(Idx)[0])))

            rows=tf.cast(rows, tf.int64)
            columns=tf.cast(columns, tf.int64)


            # if self.mode == "siamese":
            #     neighbor_matrix = self.values[:, 1:]
            #     normalized_matrix = preprocessing.normalize(neighbor_matrix, norm="l2")
            #     if self.distance == "exp":
            #         similarities = np.exp(-normalized_matrix / self.temperature)
            #     elif self.distance == "d":
            #         similarities = 1 / (1 + normalized_matrix)
            #     elif self.distance == "log":
            #         similarities = np.log((normalized_matrix + 1) / (normalized_matrix + np.finfo(np.float32).eps))
            #     elif self.distance == "1-d":
            #         similarities = 1 - normalized_matrix
            #
            #     # values = np.concatenate((np.ones(Idx.shape[0]).reshape(-1, 1), similarities), axis=1)
            #
            #     values = np.concatenate((np.max(similarities, axis=1).reshape(-1, 1), similarities), axis=1)
            #
            #     values = values[:, :self.k + 1]
            #     values = values.ravel().tolist()
            #
            #     sparse_matrix = tf.sparse.SparseTensor(indices=list(zip(rows, columns)),
            #                                            values=values,
            #                                            dense_shape=[Idx.shape[0], Idx.shape[0]])
            # else:
            sparse_matrix = tf.sparse.SparseTensor(indices=tf.transpose([rows, columns]),
                                                       values=tf.ones(tf.shape(columns), tf.float32),
                                                       dense_shape=[tf.cast(tf.shape(Idx)[0], tf.int32),
                                                                    tf.cast(tf.shape(Idx)[0],tf.int32)])
            return sparse_matrix

    def call(self, inputs):
        data_input = inputs[0]
        neighbor_indices = inputs[1]

        adj_matrix = self.get_affinity(neighbor_indices[:,  :self.k + 1])

        sparse_data_input = adj_matrix.__mul__(data_input)

        reduced_sum = tf.sparse.reduce_sum(sparse_data_input, 1)
        # reduced_mean = tf.math.divide(reduced_sum, self.k+1)
        # sparse_mean = tf.sparse.reduce_sum(sparse_data_input, 1)
        A_raw = tf.reshape(tensor=reduced_sum, shape=(tf.shape(data_input)[1],))

        alpha = K.softmax(A_raw)
        return alpha, A_raw

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)


class Last_Sigmoid(Layer):
    """
    Attention Activation
    This layer contains the last sigmoid layer of the network
    # Arguments
        output_dim:         positive integer, dimensionality of the output space
        kernel_initializer: initializer of the `kernel` weights matrix
        bias_initializer:   initializer of the `bias` weights
        kernel_regularizer: regularizer function applied to the `kernel` weights matrix
        bias_regularizer:   regularizer function applied to the `bias` weights
        use_bias:           boolean, whether use bias or not
    # Input shape
        2D tensor with shape: (n, input_dim)
    # Output shape
        2D tensor with shape: (1, units)
    """

    def __init__(self, output_dim, subtyping,kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 pooling_mode="sum",
                 kernel_regularizer=None, bias_regularizer=None,
                 use_bias=True, **kwargs):
        self.output_dim = output_dim

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.pooling_mode = pooling_mode
        self.use_bias = use_bias
        self.subtyping=subtyping
        self.norm=tf.keras.layers.LayerNormalization()

        super(Last_Sigmoid, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None

        self.input_built = True

    def call(self, x):

        if self.subtyping:
            x = K.sum(x, axis=0, keepdims=True)
            x = K.dot(x, self.kernel)
            if self.use_bias:
                x = K.bias_add(x, self.bias)
            out = K.softmax(x)

        else:
            x = K.sum(x, axis=0, keepdims=True)
            x = K.dot(x, self.kernel)
            if self.use_bias:
                x = K.bias_add(x, self.bias)
            out = K.sigmoid(x)
        return out

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)


class CustomAttention(Layer):

    def __init__(
            self,
            weight_params_dim,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.weight_params_dim = weight_params_dim

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

        self.wq_init = self.kernel_initializer
        self.wk_init = self.kernel_initializer

        self.wq_regularizer = self.kernel_regularizer
        self.wk_regularizer = self.kernel_regularizer

    def build(self, input_shape):
        # Input shape.
        # List of 2D tensors with shape: (batch_size, input_dim).
        input_dim = input_shape[1]

        self.wq_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer=self.wq_init,
            name="wq",
            regularizer=self.wq_regularizer,
            trainable=True,
        )

        self.wk_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer=self.wk_init,
            name="wk",
            regularizer=self.wk_regularizer,
            trainable=True,
        )

        self.input_built = True

    def call(self, inputs):
        wsi_bag = inputs

        attention_weights = self.compute_attention_scores(wsi_bag)

        return attention_weights

    def compute_attention_scores(self, instance):
        q = tf.tensordot(instance, self.wq_weight_params, axes=1)

        k = tf.tensordot(instance, self.wk_weight_params, axes=1)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)

        matmul_qk = tf.tensordot(q, tf.transpose(k), axes=1)  # (..., seq_len_q, seq_len_k)

        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        return scaled_attention_logits

class CHARM(Layer):

    def __init__(self, n_classes,k):
        super(CHARM, self).__init__()


        self._fc1  = Dense(512, activation="relu",name='fc1')
        self.k=[2,4,8]
        self.n_classes = n_classes
        self.layer1 = CustomAttention(weight_params_dim=256)
        self.wv = tf.keras.layers.Dense(256)
        self.last_sigmoid = Last_Sigmoid(output_dim=2,
                                        name='FC1_sigmoid_1', kernel_regularizer=l2(1e-5,),
                                        pooling_mode='sum', subtyping=True)

    def call(self, inputs,mask=None):
        input_tensor = inputs[0]
        sp_matrix = inputs[1]
        dense = self._fc1(input_tensor)

        attention_matrix = self.layer1 (dense)
        value = self.wv(dense)
        pool_outputs=[]
        for k in self.k:
            alpha, a_raw =  NeighborAggregator(output_dim=1, k=k,name="alpha")([attention_matrix, sp_matrix])
            local_attn_output = multiply([alpha, value], name="mul_{}".format(k))
            pool_outputs.append(local_attn_output)

        spp_pool = tf.concat(pool_outputs,axis=0)
        out = self.last_sigmoid (spp_pool)

        return out



