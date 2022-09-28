import os
import numpy as np
from flushed_print import print
import time
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_score, recall_score,accuracy_score
from dataset_utils.alternating import Alternating_task_generator
from dataset_utils.accumulating import Accumulating_task_generator
from training.custom_layers import TransMIL
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CallbackList
from collections import deque
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
from training.custom_callbacks import CustomReduceLRoP
from training.neighboring_custom_layers import CHARM
import tensorflow_probability as tfp
from utils.utils import get_cam_1d
from training.DTFT_MIL_layers import MILAttentionLayer, DimReduction, Classifier_1fc, Attention_with_Classifier

class MultiNet:
    def __init__(self, args):
        """
        Build the architercure of the Graph Att net
        Parameters
        ----------
        arch:          a dict describing the architecture of the neural network to be trained
        mode            :str, specifying the version of the model (siamese, euclidean)
        containing input_types
        and input_placeholders for each key and value pair, respecively.
        input_shape:   tuple(int, int, int) of the input shape of the patches

        """

        self.strategy=args.strategy
        self.input_shape = args.input_shape
        self.n_classes=args.n_classes

        self.seed=args.seed_value

        self.csv_file=args.csv_file

        self.save_dir=args.save_dir


        self.inputs = {
            'bag': Input(self.input_shape),
            'indices': Input(shape=(11,), name='indices')
        }

        #attn_output = TransMIL(n_classes=2, seed=self.seed)(self.inputs['bag'])

        out = CHARM(n_classes=2, k=[2,4,8])([self.inputs['bag'], self.inputs["indices"]])

        #self.net = Model(inputs=self.inputs['bag'], outputs=[attn_output])
        self.net = Model(inputs=[self.inputs['bag'], self.inputs["indices"]], outputs=[out])

    @property
    def model(self):
        return self.net

    def train(self,train_bags, val_bags,args,fold_id):
        """
        Train the Graph Att net
        Parameters
        ----------
        train_set       : a list of lists, each of which contains the np.ndarray of the patches of each image,
        the label of the image and a list of filenames of the patches
        check_dir       :str, specifying directory where the weights of the siamese net are stored
        irun            :int, id of the experiment
        ifold           :int, fold of the k-corss fold validation
        weight_file     :boolen, specifying whether there is a weightflie or not

        Returns
        -------
        A History object containing  a record of training loss values and metrics values at successive epochs,
        as well as validation loss values and validation metrics values.
        """

        train_gen = Alternating_task_generator(batch_size=1,csv_file=self.csv_file,feature_path=args.feature_path, filenames=train_bags, train=True)

        val_gen = Alternating_task_generator(batch_size=1, csv_file=self.csv_file,feature_path=args.feature_path, filenames=val_bags, train=True)

        classifier = Classifier_1fc(2, 0)
        attention = MILAttentionLayer(512,use_gated=False)
        dimReduction = DimReduction(512, numLayer_Res=0)
        attCls = Attention_with_Classifier(L=512,num_cls=2, droprate=0)

        if not os.path.exists(os.path.join(args.save_dir, fold_id)):
                os.makedirs(os.path.join(args.save_dir, fold_id))

        checkpoint_path = os.path.join(args.save_dir, fold_id, "{}.ckpt".format(args.experiment_name))

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         monitor='val_loss',
                                                         save_weights_only=True,
                                                         save_best_only=True,
                                                         mode='auto',
                                                         save_freq='epoch',
                                                         verbose=1)

        _callbacks = [cp_callback]
        callbacks = CallbackList(_callbacks, add_history=True, model=self.net)

        logs = {}

        self.optimizer = Adam(lr=args.init_lr, beta_1=0.9, beta_2=0.999)

        reduce_rl_plateau = CustomReduceLRoP(patience=10,
                                             factor=0.2,
                                             verbose=1,
                                             optim_lr=self.optimizer.learning_rate,
                                             mode="min",
                                             reduce_lin=False)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        train_acc_metric = tf.keras.metrics.Accuracy()
        val_acc_metric = tf.keras.metrics.Accuracy()
        train_loss_tracker = tf.keras.metrics.Mean()
        val_loss_tracker = tf.keras.metrics.Mean()

        def apply_accu_gradients():
            self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.net.trainable_variables))
            self.n_acum_step.assign(0)
            for i in range(len(self.gradient_accumulation)):
                self.gradient_accumulation[i].assign(tf.zeros_like(self.net.trainable_variables[i], dtype=tf.float32))

        @tf.function(experimental_relax_shapes=True)
        def accumulated_train_step(x,y):
            self.n_acum_step.assign_add(1)

            with tf.GradientTape() as tape:
                y_pred = self.net(x, training=True)
                loss = loss_fn(y, y_pred)

            gradients = tape.gradient(loss, self.net.trainable_variables)
            for i in range(len(self.gradient_accumulation)):
                self.gradient_accumulation[i].assign_add(gradients[i])
            tf.cond(tf.equal(self.n_acum_step, self.n_gradients), apply_accu_gradients, lambda: None)
            return loss, y_pred


        def accumulated_val_step(x,y):

            y_pred = self.net(x, training=True)
            loss = loss_fn(y, y_pred)
            return loss, y_pred


        @tf.function(experimental_relax_shapes=True)
        def train_step(x, y):
            with tf.GradientTape() as tape:
                logits = self.net(x, training=True)
                loss_value = loss_fn(y, logits)

            grads = tape.gradient(loss_value, self.net.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
            train_loss_tracker.update_state(loss_value)
            train_acc_metric.update_state(y, tf.argmax(logits,1))
            return {"train_loss": train_loss_tracker.result(), "train_accuracy": train_acc_metric.result()}

        @tf.function(experimental_relax_shapes=True)
        def val_step(x, y):
            val_logits = self.net(x, training=False)
            val_loss = loss_fn(y, val_logits)
            val_loss_tracker.update_state(val_loss)
            val_acc_metric.update_state(y, tf.argmax(val_logits, 1))
            return {"val_loss": val_loss_tracker.result(), "val_accuracy": val_acc_metric.result()}

        early_stopping = 20
        loss_history = deque(maxlen=early_stopping + 1)
        if self.strategy=='Accumulating gradients':
            self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False)
                                          for v in
                                          self.net.trainable_variables]
            self.n_gradients = tf.constant(2, dtype=tf.int32)

            self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)

        callbacks.on_train_begin(logs=logs)
        reduce_rl_plateau.on_train_begin()
        for epoch in range(args.max_epoch):
                    train_acc_metric.reset_states()
                    val_acc_metric.reset_states()
                    train_loss_tracker.reset_states()
                    val_loss_tracker.reset_states()

                    print("\nStart of epoch %d" % (epoch,))
                    start_time = time.time()
                    for step, (x_batch_train,y_batch_train) in enumerate(train_gen):
                        step_loss=[]
                        step_logits=[]
                        tslideLabel =y_batch_train

                        for i, features in enumerate(x_batch_train):

                            slide_pseudo_feat = []
                            slide_sub_preds = []
                            slide_sub_labels = []

                            feat_index = list(range(features.shape[0]))
                            index_chunk_list = np.array_split(np.array(feat_index), 4)
                            index_chunk_list = [sst.tolist() for sst in index_chunk_list]


                            for tindex in index_chunk_list:
                                slide_sub_labels.append(tslideLabel)

                                subFeat_tensor = tf.gather(features,tindex)

                                tmidFeat = dimReduction(subFeat_tensor)

                                tAA = attention(tmidFeat)
                                tattFeats = tf.multiply( tmidFeat, tAA)
                                tattFeat_tensor = tf.reduce_sum(tattFeats, axis=0)
                                tPredict = classifier(tattFeat_tensor)

                                slide_sub_preds.append(tPredict)

                                patch_pred_logits = get_cam_1d(classifier, tattFeats)
                                print (patch_pred_logits)
                                patch_pred_logits = tf.transpose(patch_pred_logits, 0, 1)  ## n x cls
                                patch_pred_softmax = tf.nn.softmax(patch_pred_logits, dim=1)  ## n x cls




        #                     if self.strategy == 'Accumulating gradients':
        #                         loss,logits=accumulated_train_step(features,  y_batch_train)
        #                         step_logits.append(logits.numpy())
        #                         step_loss.append(loss.numpy())
        #                     elif self.strategy == 'Alternating':
        #                         train_step(features,  y_batch_train)
        #
        #                 if self.strategy == 'Accumulating gradients':
        #                     logs["train_loss"]=np.mean(step_loss)
        #                     logs["logits"]=np.mean(step_logits,0)
        #                     train_loss_tracker.update_state(logs["train_loss"])
        #                     train_acc_metric.update_state(y_batch_train, tf.argmax(logs["logits"], 1))
        #
        #                 if (step+1) % 100 == 0:
        #                         print("Training loss (for one batch) at step {}: {}".format(step, train_loss_tracker.result()))
        #
        #             train_acc = train_acc_metric.result()
        #             print("Training acc over epoch: %.4f" % (float(train_acc),))
        #             for step,(x_batch_val, y_batch_val) in enumerate(val_gen):
        #
        #                 val_step_loss = []
        #                 val_step_logits = []
        #
        #                 for i, features in enumerate(x_batch_val):
        #                         if self.strategy == 'Accumulating gradients':
        #                             loss, logits = accumulated_val_step(features, np.expand_dims(y_batch_val,axis=0))
        #                             val_step_logits.append(logits.numpy())
        #                             val_step_loss.append(loss.numpy())
        #                         elif self.strategy == 'Alternating':
        #                             val_step(features, y_batch_val)
        #
        #                 if self.strategy == 'Accumulating gradients':
        #                     val_loss_tracker.update_state(np.mean(val_step_loss))
        #                     val_acc_metric.update_state(y_batch_val, tf.argmax(np.mean(val_step_logits, 0), 1))
        #
        #             logs["val_loss"] = val_loss_tracker.result()
        #             loss_history.append(val_loss_tracker.result())
        #             val_acc = val_acc_metric.result()
        #             print("Validation acc: %.4f" % (float(val_acc),))
        #             print("Time taken: %.2fs" % (time.time() - start_time))
        #             reduce_rl_plateau.on_epoch_end(epoch, val_loss_tracker.result())
        #             callbacks.on_epoch_end(epoch, logs=logs)
        #
        #             if len(loss_history) > early_stopping:
        #                 if loss_history.popleft() < min(loss_history):
        #                     print(f'\nEarly stopping. No validation loss '
        #                           f'improvement in {early_stopping} epochs.')
        #                     break
        #
        # callbacks.on_train_end(logs=logs)

    def predict(self,test_bags,args,fold_id ,test_model):

        """
        Evaluate the test set
        Parameters
        ----------
        test_set: a list of lists, each of which contains the np.ndarray of the patches of each image,
        the label of the image and a list of filenames of the patches

        Returns
        -------

        test_loss : float reffering to the test loss
        acc       : float reffering to the test accuracy
        precision : float reffering to the test precision
        recall    : float referring to the test recall
        auc       : float reffering to the test auc


        """

        checkpoint_path = os.path.join(self.save_dir,fold_id, "{}.ckpt".format(args.experiment_name))
        test_model.load_weights(checkpoint_path)

        test_gen = Alternating_task_generator(batch_size=1, csv_file=self.csv_file,feature_path=args.feature_path, filenames=test_bags, train=False)

        @tf.function(experimental_relax_shapes=True)
        def test_step(images):

            predictions = test_model(images, training=False)
            return predictions

        y_pred = []
        y_true = []
        y_score=[]
        for x_batch_val, y_batch_val in test_gen:
            test_step_logits = []
            for i, features in enumerate(x_batch_val):
                    logits = test_step(features)
                    test_step_logits.append(logits.numpy().tolist())

            y_pred.append(tf.argmax(np.mean(test_step_logits,0), axis=1))

            y_score.append(np.mean(test_step_logits,0))
            y_true.append(int(y_batch_val.tolist()))

        test_acc =accuracy_score(y_true, y_pred)
        print("Test acc: %.4f" % (float(test_acc),))


        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        enc = OneHotEncoder()
        y_test = enc.fit_transform(np.array(y_true).reshape(-1, 1)).toarray()

        y_score=np.reshape(y_score, (len(y_score), self.n_classes))

        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        auc_score=np.mean(list(roc_auc.values()))
        print("AUC {}".format(np.mean(list(roc_auc.values()))))

        precision = precision_score(y_true,  y_pred,average='macro')
        print("precision {}".format(precision))

        recall = recall_score(y_true,  y_pred,average='macro')
        print("recall {}".format(recall))

        return test_acc, auc_score, precision, recall
