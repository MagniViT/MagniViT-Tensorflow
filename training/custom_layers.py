from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Layer,
    LayerNormalization,
    Conv2D,
    ZeroPadding2D,
    Dense,
)
from einops import rearrange, reduce
import tensorflow as tf
import tensorflow_addons as tfa


def exists(val):
    return val is not None


def moore_penrose_iter_pinv(x, iters=6):
    abs_x = tf.math.abs(x)
    col = tf.math.reduce_sum(abs_x, axis=-1)
    row = tf.math.reduce_sum(abs_x, axis=-2)
    transpose_pattern = "... i j -> ... j i"

    # z = rearrange(x, '... i j -> ... j i') / (tf.math.reduce_max(col) * tf.math.reduce_max(row))
    z = tf.einsum(transpose_pattern, x) / (
        tf.math.reduce_max(col) * tf.math.reduce_max(row)
    )

    I = tf.eye(tf.shape(x)[-1])
    # I = rearrange(I, 'i j -> () i j')

    I = tf.expand_dims(I, axis=0)

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))
    return z


class NystromAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        residual=True,
        residual_conv_kernel=33,
        eps=1e-8,
        return_attn=False,
        dropout=0.1,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.return_attn = return_attn
        self.avg_layer = tfa.layers.AdaptiveAveragePooling2D((8, 256))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = tf.keras.layers.Dense(self.inner_dim * 3, name="attn")

        self.to_out = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(dim, name="dense_1"),
                tf.keras.layers.Dropout(dropout, name="dropout"),
            ]
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2

            self.res_conv = tf.keras.models.Sequential(
                [
                    tf.keras.layers.ZeroPadding2D(
                        padding=(padding, 0), data_format="channels_first"
                    ),
                    tf.keras.layers.Conv2D(
                        filters=self.heads,
                        kernel_size=(kernel_size, 1),
                        padding="valid",
                        groups=heads,
                        data_format="channels_first",
                        name="conv_attn",
                    ),
                ]
            )

    def call(self, input_tensor, mask=None):

        b = tf.shape(input_tensor)[0]
        n = tf.shape(input_tensor)[1]
        _ = tf.shape(input_tensor)[2]

        h, m, iters, eps = (
            self.heads,
            self.num_landmarks,
            self.pinv_iterations,
            self.eps,
        )

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        padding = m - (n % m)
        paddings = tf.convert_to_tensor([[0, 0], [padding, 0], [0, 0]])

        def padded_matrix():
            return tf.pad(input_tensor, paddings, constant_values=0)

        input_tensor = tf.cond(
            tf.greater(remainder, 0), padded_matrix, lambda: tf.identity(input_tensor)
        )

        q, k, v = tf.split(self.to_qkv(input_tensor), num_or_size_splits=3, axis=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale

        q_landmarks = self.avg_layer(q)
        k_landmarks = self.avg_layer(k)

        einops_eq = "... i d, ... j d -> ... i j"
        sim1 = tf.einsum(einops_eq, q, k_landmarks)
        sim2 = tf.einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = tf.einsum(einops_eq, q_landmarks, k)

        # # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(
            lambda t: tf.nn.softmax(t, axis=-1), (sim1, sim2, sim3)
        )
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # # merge and combine heads

        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        out = self.to_out(out)
        out = out[:, -n:]

        if self.return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out


class TransLayer(Layer):
    def __init__(self, dim=512):
        super().__init__()
        self.norm = LayerNormalization()
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def call(self, input_tensor, mask=None):

        x = input_tensor + self.attn(self.norm(input_tensor))

        return x


class PPEG(Layer):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()

        self.proj = tf.keras.models.Sequential(
            [
                ZeroPadding2D(padding=7 // 2, data_format="channels_first"),
                Conv2D(
                    filters=dim,
                    kernel_size=7,
                    padding="valid",
                    strides=1,
                    groups=dim,
                    data_format="channels_first",
                    name="conv_1",
                ),
            ]
        )

        self.proj1 = tf.keras.models.Sequential(
            [
                ZeroPadding2D(padding=5 // 2, data_format="channels_first"),
                Conv2D(
                    filters=dim,
                    kernel_size=5,
                    padding="valid",
                    strides=1,
                    groups=dim,
                    data_format="channels_first",
                    name="conv_2",
                ),
            ]
        )

        self.proj2 = tf.keras.models.Sequential(
            [
                ZeroPadding2D(padding=3 // 2, data_format="channels_first"),
                Conv2D(
                    filters=dim,
                    kernel_size=3,
                    padding="valid",
                    strides=1,
                    groups=dim,
                    data_format="channels_first",
                    name="conv_3",
                ),
            ]
        )

    def call(self, input_tensor, mask=None):
        input = input_tensor[0]

        cls_token, feat_token = input[:, 0], input[:, 1:]

        transpose = tf.transpose(feat_token, perm=[0, 2, 1])

        cnn_feat = tf.reshape(transpose, (1, -1, input_tensor[1], input_tensor[2]))
        cnn_feat = tf.ensure_shape(cnn_feat, [1, 512, None, None])
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = tf.transpose(
            tf.reshape(x, [-1, tf.shape(x)[1], tf.shape(x)[2] * tf.shape(x)[3]]),
            perm=[0, 2, 1],
        )
        x = tf.concat((tf.expand_dims(cls_token, axis=0), x), axis=1)
        return x


class TransMIL(Layer):
    def __init__(self, n_classes, seed):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)

        self._fc1 = Dense(512, activation="relu", name="fc1")

        w_init = tf.random_normal_initializer(seed=seed)
        self.cls_token = tf.Variable(
            initial_value=w_init(shape=(1, 1, 512), dtype="float32"), trainable=True
        )

        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = LayerNormalization(name="norm")
        self._fc2 = Dense(self.n_classes, name="fc2")

    def call(self, inputs, mask=None):

        h = self._fc1(inputs)

        h = tf.expand_dims(h, axis=0)
        H = tf.shape(h)[1]

        _H, _W = tf.cast(
            tf.math.ceil(tf.math.sqrt(tf.cast(H, tf.float32))), tf.int32
        ), tf.cast(tf.math.ceil(tf.math.sqrt(tf.cast(H, tf.float32))), tf.int32)
        add_length = (_H * _W) - H

        h = tf.concat([h, h[:, :add_length, :]], axis=1)

        # ---->cls_token
        B = tf.shape(h)[0]
        cls_tokens = tf.broadcast_to(self.cls_token, [1, B, 512])
        h = tf.concat((cls_tokens, h), axis=1)

        # ---->Translayer x1
        h = self.layer1(h)

        # # # ---->PPEG
        h = self.pos_layer((h, _H, _W))  # [B, N, 512]
        # #
        # # ---->Translayer x2
        h = self.layer2(h)
        # # ---->cls_token
        h = self.norm(h)[:, 0]

        # # # ---->predict
        logits = self._fc2(h)  # [B, n_classes]
        Y_hat = tf.math.argmax(logits, axis=1)
        Y_prob = K.softmax(logits, axis=1)

        return Y_prob
