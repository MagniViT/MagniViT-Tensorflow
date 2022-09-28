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
        self.k=[2,4,8]
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

        X20, y = self.__data_generation(self.feature_path_256,list_IDs_temp)
        # try:
        #     X10, y = self.__data_generation(self.feature_path_10, list_IDs_temp)
        #     try:
        #         X5, y = self.__data_generation(self.feature_path_5, list_IDs_temp)
        #         return (X20,X10,X5), np.asarray(y, np.float32)
        #     except:
        #         return (X20, X10), np.asarray(y, np.float32)
        # except:
        return [(X20)],tf.repeat(np.asarray(y, np.float32),1)

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
        matrices = []
        try:
            for i in range(len(filenames)):
                with h5py.File(os.path.join(path,filenames[i]), "r") as hdf5_file:

                    base_name=os.path.splitext(os.path.basename(filenames[i]))[0]
                    wsi_name=os.path.join(path,filenames[i])

                    features = hdf5_file['features'][:]
                    #neighbor_indices = hdf5_file['indices'][:]

                    references = pd.read_csv(self.csv_file)

                    if self.train:
                        try:
                            bag_label = references["train_label"].loc[references["train"] == base_name].values.tolist()[0]
                        except:
                            bag_label = references["val_label"].loc[references["val"] == base_name].values.tolist()[0]
                    else:
                        bag_label = references["test_label"].loc[references["test"] == base_name].values.tolist()[0]


            # for k in self.k:
            #     adjacency_matrix = self.get_affinity(neighbor_indices[:, :k + 1])
            #     matrices.append(adjacency_matrix)

            return features, bag_label
        except:
            return None

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

            rows = np.asarray([[enum] * len(item) for enum, item in enumerate(Idx)]).ravel()
            columns = Idx.ravel()
            if self.mode == "siamese":
                neighbor_matrix = self.values[:, 1:]
                normalized_matrix = preprocessing.normalize(neighbor_matrix, norm="l2")
                if self.distance == "exp":
                    similarities = np.exp(-normalized_matrix / self.temperature)
                elif self.distance == "d":
                    similarities = 1 / (1 + normalized_matrix)
                elif self.distance == "log":
                    similarities = np.log((normalized_matrix + 1) / (normalized_matrix + np.finfo(np.float32).eps))
                elif self.distance == "1-d":
                    similarities = 1 - normalized_matrix

                # values = np.concatenate((np.ones(Idx.shape[0]).reshape(-1, 1), similarities), axis=1)

                values = np.concatenate((np.max(similarities, axis=1).reshape(-1, 1), similarities), axis=1)

                values = values[:, :self.k + 1]
                values = values.ravel().tolist()

                sparse_matrix = tf.sparse.SparseTensor(indices=list(zip(rows, columns)),
                                                       values=values,
                                                       dense_shape=[Idx.shape[0], Idx.shape[0]])
            else:
                sparse_matrix = tf.sparse.SparseTensor(indices=list(zip(rows, columns)),
                                                       values=tf.ones(columns.shape, tf.float32),
                                                       dense_shape=[Idx.shape[0], Idx.shape[0]])

            return sparse_matrix







