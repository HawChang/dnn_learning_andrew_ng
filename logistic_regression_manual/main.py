import numpy as np
# import matplotlib.pyplot as plt
# import h5py
# import scipy
# from PIL import Image
# from scipy import ndimage
from lr_utils import load_dataset


class LogisticModel(object):
    def __init__(self, learning_rate, normalize=False, num_iter=50, print_cost=False):
        """
        逻辑回归模型参数初始化
        :param learning_rate:学习速率
        :param num_iter:最大迭代次数
        :param print_cost:是否打印损失函数情况
        """
        self.learning_rate = learning_rate
        self.normalize = normalize
        self.num_iter = num_iter
        self.print_cost = print_cost

    def init(self, X):
        """
        逻辑回归模型中参数初始化
        :param X: 训练数据
        """
        dim, m = X.shape
        self.w = np.zeros((dim, 1))
        self.b = 0
        self.x_norm = np.ones((dim, 1))

    def propagate(self, X, Y):
        """
        一次前向和后向传播，得到各参数的梯度
        :param X: 输入值
        :param Y: 输出的期望值
        :return:
        """
        # 当前输入样本数
        # X矩阵的shape=(样本维度，样本数)
        dim, m = X.shape
        # 前向传播
        A = self.sigmoid(np.dot(self.w.T, X)+self.b)
        # A为该批样本得到的预测的值 行向量
        assert A.shape, (1, m)
        # 得到损失函数
        cost = -(1.0/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))

        # 反向传播
        dw = (1.0 / m) * np.dot(X, (A - Y).T)
        assert dw.shape, self.w.shape
        db = (1.0 / m) * np.sum(A - Y)
        assert db.dtype == float

        cost = np.squeeze(cost)
        assert cost.shape == ()

        grads = {"dw": dw,
                 "db": db}

        return grads, cost

    def train(self, X, Y):
        """
        给定训练数据，学习模型中的参数w和b。
        :param X: 训练数据
        :param Y: 训练标签
        """
        self.init(X)
        if self.normalize:
            X = self.normalize_rows(X, False)
        costs = []
        for i in range(self.num_iter):
            grads, cost = self.propagate(X, Y)

            # 得到参数的梯度
            dw = grads["dw"]
            db = grads["db"]

            # 更新参数
            self.w -= self.learning_rate*dw
            self.b -= self.learning_rate*db

            if i % 100 == 0:
                costs.append(cost)
                if self.print_cost:
                    print("Cost after iteration %i: %f" % (i, cost.squeeze()))

        return costs

    def predict(self, X):
        """
        根据该模型预测数据的标签
        :param X: 预测数据
        :return: 预测数据的标签
        """
        dim, m = X.shape
        X = self.normalize_rows(X, True)
        assert self.w.shape[0] == dim

        A = self.sigmoid(np.dot(self.w.T, X)+self.b)

        return np.where(A > 0.5, 1, 0)

    @staticmethod
    def sigmoid(x):
        """
        神经元中的激活函数
        :param x: 数值或者numpy数组
        :return: 激活函数的输出
        """
        return 1.0/(1+np.exp(-x))

    def normalize_rows(self, X, is_predict):
        """
        对于输入矩阵，按行进行数值归一化
        :param X: numpy矩阵
        :param is_predict:如果是预测，则使用训练时的归一化数据对预测数据做同样的归一化。
        :return:归一化后的numpy矩阵
        """
        dim, m = X.shape
        if not is_predict:
            self.x_norm = np.linalg.norm(X, axis=1, keepdims=True).reshape(dim, 1)
        return X/self.x_norm


# def sigmoid_gradient(x):
#     """
#     计算激活函数的导数
#     :param x: 输入值
#     :return: 此时激活函数对应的导数
#     """
#     s = sigmoid(x)
#     ds = s*(1-s)

def L1(y_pred, y):
    """
    L1损失定义
    :param y_pred: 预测的y值
    :param y: 实际的y值
    :return: L1损失值
    """
    return np.sum(np.abs(y_pred-y))


def L2(y_pred, y):
    """
    L2损失定义
    :param y_pred: 预测的y值
    :param y: 实际的y值
    :return: L2损失值
    """
    return np.sum(np.power(y_pred-y, 2))


def main():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]

    print("Number of training examples: m_train = " + str(m_train))
    print("Number of testing examples: m_test = " + str(m_test))
    print("Height/Width of each image: num_px = " + str(num_px))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_set_x shape: " + str(train_set_x_orig.shape))
    print("train_set_y shape: " + str(train_set_y.shape))
    print("test_set_x shape: " + str(test_set_x_orig.shape))
    print("test_set_y shape: " + str(test_set_y.shape))

    train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
    test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

    print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print("train_set_y shape: " + str(train_set_y.shape))
    print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print("test_set_y shape: " + str(test_set_y.shape))
    print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))

    # TODO: 如果不normalize，sigmoid中的np.exp会过大，求导的np.log(1-A)中1-A会出现0。怎么解决???
    model = LogisticModel(learning_rate=1, normalize=True, num_iter=10000, print_cost=True)
    costs = model.train(train_set_x_flatten, train_set_y)

    y_pred_train = model.predict(train_set_x_flatten)
    y_pred_test = model.predict(test_set_x_flatten)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_pred_train - train_set_y)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_pred_test - test_set_y)) * 100))

    d = {
            "costs": costs,
            "Y_prediction_test": y_pred_test,
            "Y_prediction_train": y_pred_train,
            "w": model.w,
            "b": model.b,
            "learning_rate": model.learning_rate,
            "num_iterations": model.num_iter}

    return d


if __name__ == "__main__":
    main()
