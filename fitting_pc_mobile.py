#coding=utf-8
from __future__ import unicode_literals

from read_xls import get_device_table

import pprint, pickle
from sklearn import svm
import math
import numpy as np
import random
import sys
import pickle
import os
from matplotlib import cm
import pandas as pd
import csv


from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score


pc_file = '../Downloads/2017-12-23-21-52-56-PC-TWF.xlsx'
# mobile_file = '../Downloads/export_result_oppo_20171224.xls'
#
# mobile_file_oppo = '../Downloads/export_result_oppo_20171224_pm.xls'
# mobile_file_meizu = '../Downloads/export_result_meizu_20171224_pm.xls'

mobile_file_oppo = '../Downloads/export_result_oppo_20171226.xls'
mobile_file_meizu = '../Downloads/export_result_meizu_20171227.xls'
mobile_file_oppo = '../Downloads/export_result_oppo_20171226.xls'
mobile_file_meizu = '../Downloads/export_result_meizu_20171227.xls'

### const
USE_GAMMA_CORRECTION = False
def gamma_correction(val):
    return math.pow(val, 1.0/0.4545)


pc_table_g = get_device_table(pc_file)
mobile_table_g_oppo = get_device_table(mobile_file_oppo)
mobile_table_g_meizu = get_device_table(mobile_file_meizu)

mobile_table_g = mobile_table_g_oppo

single_t_line_table_g = [17, 32, 27, 26, 34, 24, 33, 15, 11,
                         18, 29, 21, 22, 20, 23, 25, 12, 19]


def average(arr):
    return float(sum(arr)) / len(arr)


def map_single(row, single_t_line_table):
    [code, c, cb, t1, t1b, t2, t2b] = row
    if USE_GAMMA_CORRECTION:
        [c, cb, t1, t1b, t2, t2b] = map(gamma_correction, [c, cb, t1, t1b, t2, t2b])

    if code in single_t_line_table:
        if (t1b - t1) < (t2b - t2):
            return [code, c, cb, t2, t2b, t2, t2b]
        else:
            return [code, c, cb, t1, t1b, t1, t1b]
    return row


def gen_data_set(pc_table, mobile_table, single_t_line_table):
    gt = {}  # { code: [[cb-c, t1b-t1, t2b-t2]]}
    for row in pc_table:
        [code, c, cb, t1, t1b, t2, t2b] = map_single(row, single_t_line_table)
        if code not in gt:
            gt[code] = [[cb - c, t1b - t1, t2b - t2]]
        else:
            gt[code].append([cb - c, t1b - t1, t2b - t2])

    ## average as ground truth
    gt_code_value = {}  # { code: [cb-c, t1b-t1, t2b-t2]}
    for code in gt:
        gt_code_value[code] = [
            average(map(lambda item: item[0], gt[code])),
            average(map(lambda item: item[1], gt[code])),
            average(map(lambda item: item[2], gt[code])),
        ]

    ## mobile data
    data_set = []
    process_data(mobile_table, data_set, single_t_line_table, gt_code_value)
    # process_data(pc_table, data_set, single_t_line_table, gt_code_value)

    return data_set

def process_data(mobile_table, data_set, single_t_line_table, gt_code_value):
    for row in mobile_table:
        [code, c, cb, t1, t1b, t2, t2b] = map_single(row, single_t_line_table)
        [gt_c, gt_t1, gt_t2] = gt_code_value[code]
        data_set.append([c, cb, gt_c])
        data_set.append([t1, t1b, gt_t1])
        if code not in single_t_line_table:
            data_set.append([t2, t2b, gt_t2])


def train_model(data):
    half_len = len(data)

    # train
    X = []
    y = []
    for [c, cb, delta] in data[:half_len]:
        X.append([c, cb])
        y.append(delta)

    svr_rbf_general = svm.SVR(kernel='rbf')
    svr_linear_general = svm.SVR(kernel='linear')
    svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = svm.SVR(kernel='linear', C=1e3)
    svr_poly = svm.SVR(kernel='poly', C=1e3, degree=2)

    model_br = BayesianRidge()
    model_lr = LinearRegression()
    model_etc = ElasticNet()
    model_svr = SVR()
    model_gbr = GradientBoostingRegressor()

    # clf = svr_linear_general
    clf = svr_linear_general
    clf.fit(X, y)

    return clf


def validate(clf, data):
    half_len = len(data)
    # validate
    validate = []
    gt = []
    for [c, cb, delta] in data[:half_len]:
        validate.append([c, cb])
        gt.append(delta)

    test = clf.predict(validate)

    for i in range(len(test)):
        validate_row = [validate[i], test[i], gt[i], abs(test[i] - gt[i]), abs(test[i] - gt[i]) / gt[i] * 100]
        print validate_row


def f_nd(c0, c1, f):
    result = []
    for i in range(len(c0)):
        row = []
        for j in range(len(c0[i,::])):
            c0f = c0[i,j]
            c1f = c1[i,j]
            row.append(f(c0f, c1f))
        result.append(row)
    return np.array(result)


def gen_f(clf, x,y):

    feed = [[x,y]]
    output = clf.predict(feed)
    ret = output[0]
    if y < 160:
        ret = y-x;

    if x >= y:
        return 0
    elif ret <= 0:
        return 0
    elif ret > 255:
        return 255

    return ret

def gen_f_plain(clf, x,y):

    feed = [[x,y]]
    output = clf.predict(feed)
    ret = output[0]

    return ret


def plot_scatters_and_mesh(scatters, mesh_x,mesh_y,mesh_z):
    # plot scatters
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    dd = np.array(scatters)

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for c, m in [('r', 'o')]:
        xs = dd[::, 0]
        ys = dd[::, 1]
        zs = dd[::, 2]
        ax.scatter(xs, ys, zs, c=c, marker=m)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ### plot mesh
    ax.plot_wireframe(mesh_x, mesh_y, mesh_z, rstride=1, cstride=1)
    ax.set_zlim(0, 255)

    plt.show()


def save_model_data(f):
    outfile = './model.csv'
    out = open(outfile, 'w')
    csv_writer = csv.writer(out)
    list = []
    for x in range(1, 256):
        for y in range(160, 256):
            if x < y:
                csv_writer.writerow([x, y, f(x, y)])
    print "write done"


def train_and_plot():
    data_set = gen_data_set(pc_table_g, mobile_table_g_oppo, single_t_line_table_g)
    data_set_meizu = gen_data_set(pc_table_g, mobile_table_g_meizu, single_t_line_table_g)

    clf = train_model(data_set + data_set_meizu)
    validate(clf, data_set_meizu)

    mesh_x, mesh_y = np.mgrid[100:255:5, 100:255:5]
    f_plain = lambda a, b: gen_f_plain(clf, a, b)
    f = lambda a, b: gen_f(clf, a, b)

    mesh_z = f_nd(mesh_x, mesh_y, f_plain)
    plot_scatters_and_mesh(data_set_meizu + data_set, mesh_x, mesh_y, mesh_z)
    save_model_data(f)
    print 'done'


def cross_validation():
    #prepare data

    data_set = gen_data_set(pc_table_g, mobile_table_g_oppo, single_t_line_table_g)
    data_set_meizu = gen_data_set(pc_table_g, mobile_table_g_meizu, single_t_line_table_g)

    raw_data = np.array(data_set + data_set_meizu)
    X = raw_data[:, :-1]
    y = raw_data[:, -1]

    # models
    n_folds = 6
    model_br = BayesianRidge()
    model_lr = LinearRegression()
    model_etc = ElasticNet()
    model_svr = SVR()
    model_svr_linear = svm.SVR(kernel='linear')
    model_gbr = GradientBoostingRegressor()

    model_names = ['BayesianRidge', 'LinearRegression', 'ElasticNet', 'SVR', 'SVR_LINEAR', 'GBR']
    model_dic = [model_br, model_lr, model_etc, model_svr, model_svr_linear, model_gbr]

    # cross validation result
    cv_score_list = []
    pre_y_list = []
    for model in model_dic:
        scores = cross_val_score(model, X, y, cv=n_folds)
        cv_score_list.append(scores)
        pre_y_list.append(model.fit(X, y).predict(X))


    # model evluation
    n_samples, n_features = X.shape
    model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]
    model_metrics_list = []
    for i in range(len(model_dic)):  # 循环每个模型索引
        tmp_list = []  # 每个内循环的临时结果列表
        for m in model_metrics_name:  # 循环每个指标对象
            tmp_score = m(y, pre_y_list[i])  # 计算每个回归指标结果
            tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
        model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表
    df1 = pd.DataFrame(cv_score_list, index=model_names)  # 建立交叉检验的数据框
    df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框
    print ('samples: %d \t features: %d' % (n_samples, n_features))  # 打印输出样本量和特征数量
    print (70 * '-')  # 打印分隔线
    print ('cross validation result:')  # 打印输出标题
    print (df1)  # 打印输出交叉检验的数据框
    print (70 * '-')  # 打印分隔线
    print ('regression metrics:')  # 打印输出标题
    print (df2)  # 打印输出回归指标的数据框
    print (70 * '-')  # 打印分隔线
    print ('short name \t full name')  # 打印输出缩写和全名标题
    print ('ev \t explained_variance')
    print ('mae \t mean_absolute_error')
    print ('mse \t mean_squared_error')
    print ('r2 \t r2')
    print (70 * '-')  # 打印分隔线


if __name__ == '__main__':
    # train_and_plot()
    cross_validation()


