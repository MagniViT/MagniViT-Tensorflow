import os
from sklearn.model_selection import train_test_split


def load_files(dataset_path, ext=".h5"):
    """
    Parameters
    --------------------
    :param dataset_path:
    :param n_folds:
    :return: list
        List contains split datasets for K-Fold cross-validation
    """
    dataset = {}
    dataset["train"] = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(os.path.join(dataset_path, "training"))
        for file in files
        if file.endswith(ext)
    ]
    dataset["test"] = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(os.path.join(dataset_path, "testing"))
        for file in files
        if file.endswith(ext)
    ]

    return dataset


def Get_train_valid_Path(Train_set, train_percentage=0.9):
    # random.seed(12321)
    # indexes = np.arange(len(Train_set))
    # random.shuffle(indexes)
    #
    # num_train = int(train_percentage * len(Train_set))
    # train_index, test_index = np.asarray(indexes[:num_train]), np.asarray(indexes[num_train:])
    #
    # Model_Train = [Train_set[i] for i in train_index]
    # Model_Val = [Train_set[j] for j in test_index]

    # Model_Train,Model_Val = train_test_split(Train_set, test_size = 0.1, random_state = 12321,stratify=True)
    train_labels = [
        int(1) if "tumor" in os.path.splitext(os.path.basename(path))[0] else int(0)
        for path in Train_set
    ]
    Model_Train, Model_Val, y_train, y_test = train_test_split(
        Train_set,
        train_labels,
        test_size=1 - train_percentage,
        random_state=12321,
        stratify=train_labels,
    )

    return Model_Train, Model_Val
