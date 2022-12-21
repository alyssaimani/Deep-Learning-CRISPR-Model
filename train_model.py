import numpy as np
from sklearn.model_selection import train_test_split
# function for split positive (label=1) and negative (label=0) dataset
def pos_neg_data(file):
    dataset = []
    positive = []
    negative = []

    file = open(file)
    for line in file:
        data = [i for i in line.strip().split(',')]
        label = data[3]
        if label == "1":
            positive.append(data)
        else:
            negative.append(data)
        dataset.append(data)
    
    return positive,negative

# function for load dataset x (23 integer code) and dataset y (label) for regression scheme
def load_data_reg(file):
    data_x = []
    data_y = []

    file = open(file)
    for line in file:
        data = [i for i in line.strip().split(',')]
        label = np.float(data[3])
        data_y.append(label)
        integer_code = [int(i) for i in data[4:]]
        data_x.append(integer_code)
    
    return data_x,data_y

# function for load dataset x (23 integer code) and dataset y (label) for classification scheme
def load_data_class(file):
    data_x = []
    data_y = []

    file = open(file)
    for line in file:
        data = [i for i in line.strip().split(',')]
        label = data[3]
        data_y.append(label)
        integer_code = [int(i) for i in data[4:]]
        data_x.append(integer_code)
    
    return data_x,data_y

# load positive and negative dataset
positive, negative = pos_neg_data("encoded_all_data.csv")
positive_hek293t, negative_hek293t = pos_neg_data("encoded_hek293t.csv")
positive_K562, negative_K562 = pos_neg_data("encoded_K562.csv")

# load data regression scheme
reg_dataset_x, reg_dataset_y = load_data_reg("encoded_all_data.csv")
reg_hek293t_x, reg_hek293t_y = load_data_reg("encoded_hek293t.csv")
reg_K562_x, reg_K562_y = load_data_reg("encoded_K562.csv")

# load data classification scheme
class_dataset_x, class_dataset_y = load_data_class("encoded_all_data.csv")
class_hek293t_x, class_hek293t_y = load_data_class("encoded_hek293t.csv")
class_K562_x, class_K562_y = load_data_class("encoded_K562.csv")

# Split dataset with test ratio = 0.2

# Split dataset for regression scheme
reg_x_train, reg_x_test, reg_y_train, reg_y_test = train_test_split(reg_dataset_x,reg_dataset_y,test_size=0.2,random_state=True) 

reg_hek293t_x_train, reg_hek293t_x_test, reg_hek293t_y_train, reg_hek293t_y_test = train_test_split(reg_hek293t_x,reg_hek293t_y,test_size=0.2,random_state=True) 

reg_K562_x_train, reg_K562_x_test, reg_K562_y_train, reg_K562_y_test = train_test_split(reg_K562_x,reg_K562_y,test_size=0.2,random_state=True)

# Split dataset for classification scheme
class_x_train, class_x_test, class_y_train, class_y_test = train_test_split(class_dataset_x,class_dataset_y,test_size=0.2,random_state=True) 

class_hek293t_x_train, class_hek293t_x_test, class_hek293t_y_train, class_hek293t_y_test = train_test_split(class_hek293t_x,class_hek293t_y,test_size=0.2,random_state=True) 

class_K562_x_train, class_K562_x_test, class_K562_y_train, class_K562_y_test = train_test_split(class_K562_x,class_K562_y,test_size=0.2,random_state=True)



