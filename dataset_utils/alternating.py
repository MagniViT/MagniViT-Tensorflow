import random
import numpy as np
import tensorflow as tf
import os
import h5py
import pandas as pd
from sklearn import preprocessing

class Alternating_task_generator(tf.keras.utils.Sequence):
    def __init__(self, filenames,feature_path, csv_file, mode=None,train=True, batch_size=1):
        self.filenames = filenames
        self.train = train
        self.batch_size = batch_size
        self.csv_file=csv_file
        self.feature_path=feature_path
        self.mode=mode
        self.numGroup=4
        self.feature_path_256=os.path.join(self.feature_path,"h5_files")
        # self.feature_path_10 = os.path.join(self.feature_path, "10x_mag/features/torch/h5_fgitiles")
        # self.feature_path_5 = os.path.join(self.feature_path, "5x_mag/features/torch/h5_files")
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames)))

    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(len(self.filenames))

        if self.train == True:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        "returns one element from the data_set"
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        list_IDs_temp = [self.filenames[k] for k in indices]

        X20,index_chunk_list,y = self.__data_generation(self.feature_path_256,list_IDs_temp)
        # try:
        #     X10, y = self.__data_generation(self.feature_path_10, list_IDs_temp)
        #     try:
        #         X5, y = self.__data_generation(self.feature_path_5, list_IDs_temp)
        #         return (X20,X10,X5), np.asarray(y, np.float32)
        #     except:
        #         return (X20, X10), np.asarray(y, np.float32)
        # except:
        return [X20,index_chunk_list],tf.repeat(np.asarray(y, np.float32),1)

    def __Get_exploration_order(self, data, shuffle):
        "shuffles the elements of the trainset"
        indices = np.arange(len(data))
        if shuffle:
            random.shuffle(indices)
        return indices

    def __data_generation(self, path, filenames):
        """

        Parameters
        ----------
        batch_train:  a list of lists, each of which contains an np.ndarray of the patches of each image,
        the label of each image and a list of filenames of the patches

        Returns
        -------
        bag_batch: a list of np.ndarrays of size (numnber of patches,h,w,d) , each of which contains the patches of an image
        neighbors: a list  of the adjacency matrices of size (numnber of patches,number of patches) of every image
        bag_label: an np.ndarray of size (number of patches,1) reffering to the label of the image

        """

        try:
            for i in range(len(filenames)):
                with h5py.File(os.path.join(path,filenames[i]), "r") as hdf5_file:

                    base_name=os.path.splitext(os.path.basename(filenames[i]))[0]

                    features = hdf5_file['features'][:]

                    feat_index = list(range(features.shape[0]))
                    index_chunk_list = np.array_split(np.array(feat_index), 4)

                    references = pd.read_csv(self.csv_file)

                    if self.train:
                        try:
                            bag_label = references["train_label"].loc[references["train"] == base_name].values.tolist()[0]
                        except:
                            bag_label = references["val_label"].loc[references["val"] == base_name].values.tolist()[0]
                    else:
                        bag_label = references["test_label"].loc[references["test"] == base_name].values.tolist()[0]

            return features,index_chunk_list, bag_label
        except:
            return None







