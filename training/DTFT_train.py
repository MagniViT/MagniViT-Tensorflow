import os
import numpy as np
from flushed_print import print
import time
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_score, recall_score,f1_score
from dataset_utils.alternating import Alternating_task_generator
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CallbackList
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
from training.custom_callbacks import CustomReduceLRoP
from training.DTFT_MIL_layers import  GROUP_MIL, Attention_with_Classifier


class MultiNet:
    def __init__(self,args,training_flag):
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

        self.input_shape = args.input_shape
        self.n_classes = 2
        self.csv_file=args.csv_file
        self.feature_path=args.feature_path
        self.experiment_name=args.experiment_name
        self.epochs= args.epochs
        self.save_dir=args.save_dir
        self.numGroup = 4
        total_instance = 4
        self.instance_per_group = total_instance // self.numGroup
        self.training_flag=training_flag
        self.init_lr = args.init_lr

        self.distill = 'AFS'

        self.inputs = {
            'bag': Input(self.input_shape),
            'group_1': Input(1),
            'group_2': Input(1),
            'group_3': Input(1),
            'group_4': Input(1),
        }


        slide_pseudo_feat,slide_sub_preds = GROUP_MIL(distill="AFS", training_flag=self.training_flag)(
                                            [self.inputs['bag'],
                                           self.inputs['group_1'],
                                           self.inputs['group_2'],
                                           self.inputs['group_3'],
                                           self.inputs['group_4']])

        slide_pseudo_feat = tf.concat(slide_pseudo_feat, axis=0)
        slide_sub_preds = tf.concat(slide_sub_preds, axis=0)


        self.tier_1 = Model(inputs=[self.inputs['bag'],
                                       self.inputs['group_1'],
                                       self.inputs['group_2'],
                                       self.inputs['group_3'],
                                       self.inputs['group_4']
                                 ],
                         outputs=[slide_pseudo_feat,slide_sub_preds])

        self.inputs = {
            'att_tensor': Input(512,)
        }

        gSlidePred = Attention_with_Classifier(L=512)(self.inputs['att_tensor'])


        self.tier_2 = Model(inputs=self.inputs['att_tensor'], outputs=gSlidePred)


    @property
    def model_1(self):
        return self.tier_1

    @property
    def model_2(self):
        return self.tier_2


    def train(self, train_bags, val_bags, fold):
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

        train_gen = Alternating_task_generator(batch_size=1, csv_file=self.csv_file, feature_path=self.feature_path,
                                               filenames=train_bags, train=True)

        val_gen = Alternating_task_generator(batch_size=1, csv_file=self.csv_file, feature_path=self.feature_path,
                                             filenames=val_bags, train=True)

        os.makedirs(os.path.join(self.save_dir, fold, self.experiment_name, "tier_1"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, fold, self.experiment_name, "tier_2"), exist_ok=True)

        checkpoint_path_1 = os.path.join(os.path.join(self.save_dir, fold, self.experiment_name), 'tier_1',
                                         "{}.ckpt".format(self.experiment_name))
        checkpoint_path_2 = os.path.join(os.path.join(self.save_dir, fold, self.experiment_name), 'tier_2',
                                         "{}.ckpt".format(self.experiment_name))

        net_cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_1,
                                                             monitor='val_loss_2',
                                                             save_weights_only=True,
                                                             save_best_only=True,
                                                             mode='auto',
                                                             save_freq='epoch',
                                                             verbose=1)

        k_net_cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_2,
                                                               monitor='val_loss_2',
                                                               save_weights_only=True,
                                                               save_best_only=True,
                                                               mode='auto',
                                                               save_freq='epoch',
                                                               verbose=1)

        net_callbacks = CallbackList([net_cp_callback], add_history=True, model=self.tier_1)
        k_net_callbacks = CallbackList([k_net_cp_callback], add_history=True, model=self.tier_2)

        logs = {}

        net_callbacks.on_train_begin(logs=logs)
        k_net_callbacks.on_train_begin(logs=logs)

        optimizer_1 = Adam(lr=self.init_lr, beta_1=0.9, beta_2=0.999)
        optimizer_2 = Adam(lr=self.init_lr, beta_1=0.9, beta_2=0.999)

        reduce_rl_plateau = CustomReduceLRoP(patience=10,
                                             factor=0.2,
                                             verbose=1,
                                             optim_lr=optimizer_1.learning_rate,
                                             mode="min",
                                             reduce_lin=False)

        loss_fn_1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss_fn_2 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        train_acc_1_metric = tf.keras.metrics.Accuracy()
        train_acc_2_metric = tf.keras.metrics.Accuracy()

        val_acc_1_metric = tf.keras.metrics.Accuracy()
        val_acc_2_metric = tf.keras.metrics.Accuracy()

        train_loss_1_tracker = tf.keras.metrics.Mean()
        train_loss_2_tracker = tf.keras.metrics.Mean()

        val_loss_1_tracker = tf.keras.metrics.Mean()
        val_loss_2_tracker = tf.keras.metrics.Mean()

        @tf.function(experimental_relax_shapes=True)
        def train_step_1(x, y):

            with tf.GradientTape() as tape_1:
                  Y_prob,input_2_tier_2 = self.tier_1(x, training=True)
                  slide_loss_1 = loss_fn_1(y, Y_prob)

            grads_1 = tape_1.gradient(slide_loss_1, self.tier_1.trainable_weights)
            gradients, _ = tf.clip_by_global_norm(grads_1, 5.0)
            optimizer_1.apply_gradients(zip(grads_1, self.tier_1.trainable_weights))

            train_loss_1_tracker.update_state(slide_loss_1)

            return {"train_loss_1": train_loss_1_tracker.result(), 'tier_2_input':input_2_tier_2}

        def train_step_2(x, y):
            with tf.GradientTape() as tape_2:
                Y_prob = self.tier_2(x, training=True)
                slide_loss_2 = loss_fn_2(y, Y_prob)

            grads_2 = tape_2.gradient(slide_loss_2, self.tier_2.trainable_weights)
            gradients, _ = tf.clip_by_global_norm(grads_2, 5.0)
            optimizer_2.apply_gradients(zip(grads_2, self.tier_2.trainable_weights))

            train_loss_2_tracker.update_state(slide_loss_2)

            return {"train_loss_2": train_loss_2_tracker.result()}

        @tf.function(experimental_relax_shapes=True)
        def val_step_1(x, y):

            slide_sub_labels = tf.repeat(y, self.numGroup)
            slide_sub_preds, pool = self.tier_1(x, training=False)

            slide_loss_1 = loss_fn_1(slide_sub_labels,slide_sub_preds)
            val_loss_1_tracker.update_state(slide_loss_1)

            Y_prob_1 = tf.math.reduce_mean(tf.nn.softmax(slide_sub_preds, axis=1), axis=0,keepdims=True)

            val_acc_1_metric.update_state(y, tf.argmax(Y_prob_1, 1))
            return {"val_loss_1": val_loss_1_tracker.result(), "tier_2_input": pool}

        @tf.function(experimental_relax_shapes=True)
        def val_step_2(x, y):
            Y_prob_2 = self.tier_2(x, training=False)

            slide_loss_2 = loss_fn_2(y, Y_prob_2)
            val_loss_2_tracker.update_state(slide_loss_2)

            Y_prob_2 = tf.nn.softmax(Y_prob_2, axis=1)
            val_acc_2_metric.update_state(y, tf.argmax(Y_prob_2, 1))
            return {"val_loss_2": val_loss_2_tracker.result()}

        reduce_rl_plateau.on_train_begin()
        for epoch in range(self.epochs):

            train_loss_1_tracker.reset_states()
            train_loss_2_tracker.reset_states()

            train_acc_1_metric.reset_states()
            train_acc_2_metric.reset_states()

            val_loss_1_tracker.reset_states()
            val_loss_2_tracker.reset_states()

            val_acc_1_metric.reset_states()
            val_acc_2_metric.reset_states()

            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()
            for step, (x_batch_train, y_batch_train) in enumerate(train_gen):

                slide_sub_labels = np.repeat(y_batch_train, self.numGroup)
                tslideLabel=y_batch_train

                train_dict_1 = train_step_1(x_batch_train, slide_sub_labels)
                train_dict_2 = train_step_2(train_dict_1['tier_2_input'], tslideLabel)

                logs["train_loss_1"] = train_dict_1["train_loss_1"]
                logs["train_loss_2"] = train_dict_2["train_loss_2"]

                if (step + 1) % 100 == 0:
                    print("First training loss (for one batch) at step %d: %.4f" % (step, float(logs["train_loss_1"])))
                    print("Second training loss (for one batch) at step %d: %.4f" % (step, float(logs["train_loss_2"])))

            net_callbacks.on_epoch_begin(epoch, logs=logs)
            k_net_callbacks.on_epoch_begin(epoch, logs=logs)

            for step, (x_batch_val, y_batch_val) in enumerate(val_gen):
                net_callbacks.on_batch_begin(step, logs=logs)
                net_callbacks.on_test_batch_begin(step, logs=logs)

                k_net_callbacks.on_batch_begin(step, logs=logs)
                k_net_callbacks.on_test_batch_begin(step, logs=logs)

                tslideLabel = y_batch_val

                val_dict_1 = val_step_1(x_batch_val, y_batch_val)
                val_dict_2 = val_step_2(val_dict_1['tier_2_input'], tslideLabel)

                logs["val_loss_1"] = val_dict_1["val_loss_1"]
                logs["val_loss_2"] = val_dict_2["val_loss_2"]

                net_callbacks.on_test_batch_end(step, logs=logs)
                net_callbacks.on_batch_end(step, logs=logs)

                k_net_callbacks.on_test_batch_end(step, logs=logs)
                k_net_callbacks.on_batch_end(step, logs=logs)

            logs["val_loss_1"] = val_loss_1_tracker.result()
            logs["val_loss_2"] = val_loss_2_tracker.result()

            val_acc_1 = val_acc_1_metric.result()
            val_acc_2 = val_acc_2_metric.result()

            print("Validation acc 1: %.4f" % (float(val_acc_1),))
            print("Validation acc 2: %.4f" % (float(val_acc_2),))
            print("Time taken: %.2fs" % (time.time() - start_time))

            net_callbacks.on_epoch_end(epoch, logs=logs)
            k_net_callbacks.on_epoch_end(epoch, logs=logs)

        net_callbacks.on_train_end(logs=logs)
        k_net_callbacks.on_train_end(logs=logs)

    def predict(self, test_bags, fold, tier_1, tier_2):

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

        eval_accuracy_metric_1 = tf.keras.metrics.Accuracy()
        eval_accuracy_metric_2 = tf.keras.metrics.Accuracy()

        checkpoint_path_1 = os.path.join(os.path.join(self.save_dir, fold, self.experiment_name), 'tier_1',
                                         "{}.ckpt".format(self.experiment_name))
        checkpoint_path_2 = os.path.join(os.path.join(self.save_dir, fold, self.experiment_name), 'tier_2',
                                         "{}.ckpt".format(self.experiment_name))

        tier_1.load_weights(checkpoint_path_1)
        tier_2.load_weights(checkpoint_path_2)

        test_gen = Alternating_task_generator(batch_size=1, csv_file=self.csv_file, feature_path=self.feature_path,
                                               filenames = test_bags, train=False)

        @tf.function(experimental_relax_shapes=True)
        def test_step_1(images, labels):

            Y_prob_1, tier_2_in = tier_1(images, training=False)
            Y_prob_1 = tf.nn.softmax(Y_prob_1, axis=1)
            Y_prob_1 = tf.reduce_mean(Y_prob_1, axis=0, keepdims=True)
            pred_1 = tf.argmax(Y_prob_1, 1)
            eval_accuracy_metric_1.update_state(labels, pred_1)

            Y_prob_2 = tier_2(tier_2_in, training=False)
            Y_prob_2 = tf.nn.softmax(Y_prob_2, axis=1)
            pred_2 = tf.argmax(Y_prob_2, 1)
            eval_accuracy_metric_2.update_state(labels, pred_2)
            return pred_1, pred_2,Y_prob_2

        y_pred = []
        y_true = []
        y_score = []
        with tf.device('/cpu:0'):
            for enum, (x_batch_val, y_batch_val) in enumerate(test_gen):
                pred_1, pred_2, y_prob = test_step_1(x_batch_val, y_batch_val)

                y_true.append(y_batch_val)
                y_score.append(y_prob.numpy().tolist()[0])
                y_pred.append(pred_2.numpy().tolist()[0])

        test_acc_1 = eval_accuracy_metric_1.result()
        print("Test acc 1: %.4f" % (float(test_acc_1),))

        test_acc_2 = eval_accuracy_metric_2.result()
        print("Test acc 2: %.4f" % (float(test_acc_2),))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        enc = OneHotEncoder()
        y_test = enc.fit_transform(np.array(y_true).reshape(-1, 1)).toarray()
        y_score = np.reshape(y_score, (len(y_score), 2))
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        auc_score = np.mean(list(roc_auc.values()))
        print("AUC {}".format(np.mean(list(roc_auc.values()))))

        precision = precision_score(y_true, np.round(np.clip(y_pred, 0, 1)), average="macro")
        print("precision {}".format(precision))

        recall = recall_score(y_true, np.round(np.clip(y_pred, 0, 1)), average="macro")
        print("recall {}".format(recall))

        fscore = f1_score(y_true, np.round(np.clip(y_pred, 0, 1)), average="macro")
        print("f1_score {}".format(fscore))

        return test_acc_2, auc_score, precision, recall, fscore