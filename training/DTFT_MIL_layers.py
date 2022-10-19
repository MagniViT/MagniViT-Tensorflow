import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.regularizers import l2
from utils.utils import get_cam_1d
from tensorflow.keras import backend as K

class MILAttentionLayer(tf.keras.layers.Layer):
    """Implementation of the attention-based Deep MIL layer.
    Args:
      weight_params_dim: Positive Integer. Dimension of the weight matrix.
      kernel_initializer: Initializer for the `kernel` matrix.
      kernel_regularizer: Regularizer function applied to the `kernel` matrix.
      use_gated: Boolean, whether or not to use the gated mechanism.
    Returns:
      List of 2D tensors with BAG_SIZE length.
      The tensors are the attention scores after softmax with shape `(batch_size, 1)`.
    """

    def __init__(
            self,
            weight_params_dim,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=None,
            use_gated=False,
            **kwargs,
    ):

        super().__init__(**kwargs)

        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer

        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer

    def build(self, input_shape):

        # Input shape.
        # List of 2D tensors with shape: (batch_size, input_dim).
        input_dim = input_shape[1]

        self.v_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer=self.v_init,
            name="v",
            regularizer=self.v_regularizer,
            trainable=True,
        )

        self.w_weight_params = self.add_weight(
            shape=(self.weight_params_dim, 1),
            initializer=self.w_init,
            name="w",
            regularizer=self.w_regularizer,
            trainable=True,
        )

        if self.use_gated:
            self.u_weight_params = self.add_weight(
                shape=(input_dim, self.weight_params_dim),
                initializer=self.u_init,
                name="u",
                regularizer=self.u_regularizer,
                trainable=True,
            )
        else:
            self.u_weight_params = None

        self.input_built = True

    def call(self, inputs):

        # Assigning variables from the number of inputs.
        instances = self.compute_attention_scores(inputs)

        # Apply softmax over instances such that the output summation is equal to 1.
        alpha = tf.math.softmax(instances, axis=0)

        return alpha

    def compute_attention_scores(self, instance):

        # Reserve in-case "gated mechanism" used.
        original_instance = instance

        # tanh(v*h_k^T)
        instance = tf.math.tanh(tf.tensordot(instance, self.v_weight_params, axes=1))

        # for learning non-linear relations efficiently.
        if self.use_gated:
            instance = instance * tf.math.sigmoid(
                tf.tensordot(original_instance, self.u_weight_params, axes=1)
            )
        return tf.tensordot(instance, self.w_weight_params, axes=1)


class residual_block(tf.keras.layers.Layer):
    def __init__(self,
                 nChn=512,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.block = tf.keras.Sequential([
                Dense( nChn, use_bias=False),
                ReLU(),
                Dense( nChn, use_bias=False),
                ReLU()
            ], name='residual_block')
    def call(self, x):
        tt = self.block(x)
        x = x + tt
        return x

class DimReduction(tf.keras.layers.Layer):
    def __init__(self, m_dim=512,
                 numLayer_Res=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.fc1 = Dense(m_dim, use_bias=False)
        self.relu1 = ReLU()
        self.numRes = numLayer_Res

        # self.resBlocks = []
        # for ii in range(numLayer_Res):
        #     self.resBlocks.append(residual_block(m_dim))
        # self.resBlocks = tf.keras.Sequential(*self.resBlocks)

    def call(self, x):

        x = self.fc1(x)
        x = self.relu1(x)
        # if self.numRes > 0:
        #     x = self.resBlocks(x)
        return x

class Attention_with_Classifier(tf.keras.layers.Layer):

    def __init__(self, L, num_cls=2, droprate=0, **kwargs):
        super(Attention_with_Classifier, self).__init__(**kwargs)
        self.attention = MILAttentionLayer(weight_params_dim=L)
        self.classifier = tf.keras.Sequential([
            tf.keras.Input(shape=(512,)),
            tf.keras.layers.Dropout(rate=droprate),
            tf.keras.layers.Dense(num_cls),
        ],
            name="Classifier_1fc",
        )

    def call(self, x):  ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat =  tf.matmul(tf.transpose(AA), x)  ## K x L
        pred = self.classifier(afeat)  ## K x num_cls
        return pred
class GROUP_MIL(tf.keras.layers.Layer):

    def __init__(self,distill, training_flag, **kwargs):
        super().__init__(**kwargs)
        self.distill=distill
        self.instance_per_group=1
        self.training_flag=training_flag

        self.classifier = tf.keras.Sequential([
                tf.keras.Input(shape=(512,)),
                tf.keras.layers.Dropout(rate=0),
                tf.keras.layers.Dense(2),
            ],
            name="Classifier_1fc",
        )
        self.attention = MILAttentionLayer(512,use_gated=False, name="MILAttentionLayer")
        self.dimReduction = DimReduction(512, numLayer_Res=0, name="DimReduction")

    def  call(self, input_tensor):

        tfeat_tensor = input_tensor[0]
        index_chunk_list = input_tensor[1:]

        if not self.training_flag:
            self.slide_pseudo_feat = []
            self.slide_sub_preds = []

            tmidFeat = self.dimReduction(tfeat_tensor)

            AA = self.attention(tmidFeat)


            for enum, tindex in enumerate(range(len(index_chunk_list))):

                tindex = index_chunk_list[enum]
                subFeat_tensor = tf.squeeze(tf.gather(tmidFeat, tf.cast(tindex,tf.int32)))
                subFeat_tensor = tf.ensure_shape(subFeat_tensor, [None, 512])

                tAA = tf.squeeze(tf.gather(AA, tf.cast(tindex,tf.int32)),axis=-1)
                tAA = tf.nn.softmax(tAA, axis=0)
                tattFeats = tf.multiply(subFeat_tensor, tAA)
                tattFeat_tensor = tf.expand_dims(tf.reduce_sum(tattFeats, axis=0), axis=0)

                tPredict = self.classifier(tattFeat_tensor)
                self.slide_sub_preds.append(tPredict)

                patch_pred_logits = get_cam_1d(self.classifier, tattFeats)
                patch_pred_logits = tf.transpose(patch_pred_logits)  ## n x cls
                patch_pred_softmax = tf.nn.softmax(patch_pred_logits, axis=1)  ## n x cls

                sort_idx = tf.argsort(patch_pred_softmax[:, -1], direction='DESCENDING')

                topk_idx_max = sort_idx[:self.instance_per_group]
                topk_idx_min = sort_idx[-self.instance_per_group:]
                topk_idx = tf.concat([topk_idx_max, topk_idx_min], axis=0)

                af_inst_feat = tattFeat_tensor

                if self.distill == 'MaxMinS':
                        MaxMin_inst_feat = tf.gather(tmidFeat, topk_idx)
                        self.slide_pseudo_feat.append(MaxMin_inst_feat)
                elif self.distill == 'MaxS':
                        max_inst_feat = tf.gather(tmidFeat, topk_idx_max)
                        self.slide_pseudo_feat.append(max_inst_feat)
                elif self.distill == 'AFS':
                        self.slide_pseudo_feat.append(af_inst_feat)

            return self.slide_sub_preds, self.slide_pseudo_feat
        else:
            self.slide_pseudo_feat = []
            self.slide_sub_preds = []

            for enum, tindex in enumerate(range(len(index_chunk_list))):


                    tindex=index_chunk_list[enum]
                    subFeat_tensor = tf.squeeze(tf.gather(tfeat_tensor, tf.cast(tindex,tf.int32)))
                    subFeat_tensor = tf.ensure_shape(subFeat_tensor, [None, 1024])

                    tmidFeat = self.dimReduction(subFeat_tensor)

                    tAA = self.attention(tmidFeat)
                    tattFeats = tf.multiply(tmidFeat, tAA)
                    tattFeat_tensor = tf.expand_dims(tf.reduce_sum(tattFeats, axis=0),axis=0)

                    tPredict = self.classifier(tattFeat_tensor)
                    self.slide_sub_preds.append(tPredict)

                    patch_pred_logits = get_cam_1d(self.classifier, tattFeats)
                    patch_pred_logits = tf.transpose(patch_pred_logits)  ## n x cls
                    patch_pred_softmax = tf.nn.softmax(patch_pred_logits, axis=1)  ## n x cls

                    sort_idx = tf.argsort(patch_pred_softmax[:, -1], direction='DESCENDING')

                    topk_idx_max = sort_idx[:self.instance_per_group]
                    topk_idx_min = sort_idx[-self.instance_per_group:]
                    topk_idx = tf.concat([topk_idx_max, topk_idx_min], axis=0)

                    MaxMin_inst_feat = tf.gather(tmidFeat, topk_idx)

                    max_inst_feat = tf.gather(tmidFeat, topk_idx_max)
                    af_inst_feat = tattFeat_tensor

                    if self.distill == 'MaxMinS':
                            self.slide_pseudo_feat.append(MaxMin_inst_feat)
                    elif self.distill == 'MaxS':
                            self.slide_pseudo_feat.append(max_inst_feat)
                    elif self.distill == 'AFS':
                            self.slide_pseudo_feat.append(af_inst_feat)

            return self.slide_sub_preds, self.slide_pseudo_feat

