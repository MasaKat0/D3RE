import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from train import train, loss_func, test 
from model import NN, CNN

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor

from densratio import densratio
from pykliep import DensityRatioEstimator

import xgboost as xgb


file_names = ['books_processed_balanced',
 'dvd_processed_balanced',
 'electronics_processed_balanced',
 'kitchen_processed_balanced']

def calc_result(reg, x0, y0, x1, y1, dr=None):
    reg.fit(x0, y0, sample_weight=dr)
    train_loss = np.mean((y0 - reg.predict(x0))**2)
    test_loss = np.mean((y1 - reg.predict(x1))**2)
    rating_temp = y1.copy()
    rating_temp[rating_temp >= 3] = 100
    auc = calc_auc(rating_temp, reg.predict(x1))
    return train_loss, test_loss, auc

def calc_auc(y, f):
    fpr, tpr, _ = metrics.roc_curve(y, f, pos_label=100)
    auc = metrics.auc(fpr, tpr)
    return 1-auc

def main():
    ite = 10
    num_train_data = 2000
    num_test_data = 2000
    Net = NN

    model_num = 3

    learning_rate = 1e-4

    epoch = 200
    batchsize = 256

    seed = 2020

    for f_name_idx0 in range(len(file_names)):
        for f_name_idx1 in range(f_name_idx0+1, len(file_names)):
            train_loss_normal = np.zeros((ite, model_num))
            test_loss_normal = np.zeros((ite, model_num))
            auc_normal = np.zeros((ite, model_num))

            train_loss_kerulsif = np.zeros((ite, model_num))
            test_loss_kerulsif = np.zeros((ite, model_num))
            auc_kerulsif = np.zeros((ite, model_num))

            train_loss_kerkleip = np.zeros((ite, model_num))
            test_loss_kerkleip = np.zeros((ite, model_num))
            auc_kerkleip = np.zeros((ite, model_num))

            train_loss_pu = np.zeros((ite, model_num))
            test_loss_pu = np.zeros((ite, model_num))
            auc_pu = np.zeros((ite, model_num))

            train_loss_ulsif = np.zeros((ite, model_num))
            test_loss_ulsif = np.zeros((ite, model_num))
            auc_ulsif = np.zeros((ite, model_num))

            train_loss_nnpu = np.zeros((ite, model_num))
            test_loss_nnpu = np.zeros((ite, model_num))
            auc_nnpu = np.zeros((ite, model_num))

            train_loss_nnulsif = np.zeros((ite, model_num))
            test_loss_nnulsif = np.zeros((ite, model_num))
            auc_nnulsif = np.zeros((ite, model_num))
            
            f_name0 = file_names[f_name_idx0]
            f_name1 = file_names[f_name_idx1]

            for i in range(ite):
                np.random.seed(seed)

                if f_name0 != f_name1:

                    data0 = pd.read_csv('dataset/%s.csv'%f_name0)
                    data1 = pd.read_csv('dataset/%s.csv'%f_name1)

                    data0 = data0.dropna()
                    data1 = data1.dropna()

                    perm0 = np.random.permutation(len(data0))
                    perm1 = np.random.permutation(len(data1))

                    choice0 = np.zeros(len(data0))
                    choice0[perm0[:num_train_data]] = 1
                    data0['choice'] = choice0

                    choice1 = np.zeros(len(data1))
                    choice1[perm1[:num_test_data]] = 1
                    data1['choice'] = choice1

                    data0 = data0.get(['rating', 'text', 'item', 'choice'])
                    data1 = data1.get(['rating', 'text', 'item', 'choice'])

                    data = pd.concat([data0, data1])

                else:
                    data = pd.read_csv('dataset/%s.csv'%f_name0)

                    data = data.dropna()

                    perm = np.random.permutation(len(data))

                    choice = np.zeros(len(data))
                    choice[perm[:num_train_data+num_test_data]] = 1

                    data['choice'] = choice

                    print('N: ', len(data))

                text_data = data.text.values

                vectorizer = TfidfVectorizer(max_features=10000, min_df=0.0, max_df=0.8)
                #vectorizer = TfidfVectorizer(min_df=0.0, max_df=0.8)
                text_list_vec = vectorizer.fit_transform(text_data)

                #X = text_list_vec[data['choice'].values == 1].toarray()
                X = text_list_vec[data['choice'].values == 1].toarray()
                print(X.shape)

                pca = PCA(n_components=100)
                pca.fit(X)

                X_pca = pca.transform(X)

                rating0 = data[data['choice'].values == 1].rating.values[:num_train_data]
                rating1 = data[data['choice'].values == 1].rating.values[num_train_data:]

                X0 = X[:num_train_data]
                X1 = X[num_train_data:]

                X_pca0 = X_pca[:num_train_data]
                X_pca1 = X_pca[num_train_data:]

                result = densratio(X_pca0, X_pca1, sigma_range=[0.01, 0.05, 0.1, 0.5, 1], lambda_range=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1])

                dr0 = result.compute_density_ratio(X_pca0)

                kliep = DensityRatioEstimator()
                kliep.fit(X_pca0, X_pca1)
                #dr1 = np.ones(len(X_pca0))
                dr1 = kliep.predict(X_pca0)

                dim = X0.shape[1]

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                #device = 'cpu'

                model = Net(dim).to(device)
                optimizer = optim.Adam(params=model.parameters(), lr = learning_rate, weight_decay=1e-5)
                
                model = train(X0, X1, epoch, model, optimizer, device, batchsize=batchsize, method='PU')
                dr2 = test(X0, model, device, batchsize=100, method='PU')

                model = Net(dim).to(device)
                optimizer = optim.Adam(params=model.parameters(), lr = learning_rate, weight_decay=1e-5)
                
                model = train(X0, X1, epoch, model, optimizer, device, batchsize=batchsize, method='uLSIF')
                dr3 = test(X0, model, device, batchsize=100, method='uLSIF')
                
                model = Net(dim).to(device)
                optimizer = optim.Adam(params=model.parameters(), lr = learning_rate, weight_decay=1e-5)

                model = train(X0, X1, epoch, model, optimizer, device, batchsize=batchsize, method='nnPU')
                dr4 = test(X0, model, device, batchsize=100, method='PU')

                model = Net(dim).to(device)
                optimizer = optim.Adam(params=model.parameters(), lr = learning_rate, weight_decay=1e-5)

                model = train(X0, X1, epoch, model, optimizer, device, batchsize=batchsize, method='nnuLSIF')
                dr5 = test(X0, model, device, batchsize=100, method='uLSIF')

                dr3[dr3 < 0] = 0.
                dr5[dr5 < 0] = 0.

                dr0[~((dr0 > 0)&(dr0 < 100))] = 100
                dr1[~((dr1 > 0)&(dr1 < 100))] = 100
                dr2[~((dr2 > 0)&(dr2 < 100))] = 100
                dr3[~((dr3 > 0)&(dr3 < 100))] = 100
                dr4[~((dr4 > 0)&(dr4 < 100))] = 100
                dr5[~((dr5 > 0)&(dr5 < 100))] = 100

                print(dr3)
                print(dr4)
                print(dr5)

                print('meandr4', np.mean(dr4))
                print('meandr5', np.mean(dr5))

                reg = Ridge()
                reg = GridSearchCV(reg, {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}, cv=5)

                idx_model = 0

                x_train = X_pca0
                x_test = X_pca1
                
                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=None)
                train_loss_normal[i, idx_model] = train_loss
                test_loss_normal[i, idx_model] = test_loss
                auc_normal[i, idx_model] = auc

                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=dr0)
                train_loss_kerulsif[i, idx_model] = train_loss
                test_loss_kerulsif[i, idx_model] = test_loss
                auc_kerulsif[i, idx_model] = auc

                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=dr1)
                train_loss_kerkleip[i, idx_model] = train_loss
                test_loss_kerkleip[i, idx_model] = test_loss
                auc_kerkleip[i, idx_model] = auc

                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=dr2)
                train_loss_pu[i, idx_model] = train_loss
                test_loss_pu[i, idx_model] = test_loss
                auc_pu[i, idx_model] = auc

                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=dr3)
                train_loss_ulsif[i, idx_model] = train_loss
                test_loss_ulsif[i, idx_model] = test_loss
                auc_ulsif[i, idx_model] = auc

                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=dr4)
                train_loss_nnpu[i, idx_model] = train_loss
                test_loss_nnpu[i, idx_model] = test_loss
                auc_nnpu[i, idx_model] = auc

                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=dr5)
                train_loss_nnulsif[i, idx_model] = train_loss
                test_loss_nnulsif[i, idx_model] = test_loss
                auc_nnulsif[i, idx_model] = auc

                print('0:normal', test_loss_normal)
                print('0:nnulsif', test_loss_nnulsif)
                print('0:nnpu', test_loss_nnpu)

                print('0:normal', auc_normal)
                print('0:nnulsif', auc_nnulsif)
                print('0:nnpu', auc_nnpu)
                #reg = KernelRidge(alpha=1, kernel='rbf', gamma=0.1)
                
                #reg = KernelRidge(alpha=0.1, kernel='rbf', gamma=1)



                reg = KernelRidge()
                reg = GridSearchCV(reg, {'kernel': ['rbf'], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'gamma': [0.001, 0.01, 0.1, 1]}, cv=5)

                idx_model = 1

                x_train = X_pca0
                x_test = X_pca1
                
                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=None)
                train_loss_normal[i, idx_model] = train_loss
                test_loss_normal[i, idx_model] = test_loss
                auc_normal[i, idx_model] = auc

                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=dr0)
                train_loss_kerulsif[i, idx_model] = train_loss
                test_loss_kerulsif[i, idx_model] = test_loss
                auc_kerulsif[i, idx_model] = auc

                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=dr1)
                train_loss_kerkleip[i, idx_model] = train_loss
                test_loss_kerkleip[i, idx_model] = test_loss
                auc_kerkleip[i, idx_model] = auc

                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=dr2)
                train_loss_pu[i, idx_model] = train_loss
                test_loss_pu[i, idx_model] = test_loss
                auc_pu[i, idx_model] = auc

                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=dr3)
                train_loss_ulsif[i, idx_model] = train_loss
                test_loss_ulsif[i, idx_model] = test_loss
                auc_ulsif[i, idx_model] = auc

                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=dr4)
                train_loss_nnpu[i, idx_model] = train_loss
                test_loss_nnpu[i, idx_model] = test_loss
                auc_nnpu[i, idx_model] = auc

                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=dr5)
                train_loss_nnulsif[i, idx_model] = train_loss
                test_loss_nnulsif[i, idx_model] = test_loss
                auc_nnulsif[i, idx_model] = auc

                print('1:normal', test_loss_normal)
                print('1:nnulsif', test_loss_nnulsif)

                '''

                reg = KernelRidge()
                reg = GridSearchCV(reg, {'kernel': ['polynomial'], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'gamma': [2, 3, 4, 5]}, cv=5)

                idx_model = 2

                x_train = X0
                x_test = X1
                
                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=None)
                train_loss_normal[i, idx_model] = train_loss
                test_loss_normal[i, idx_model] = test_loss
                auc_normal[i, idx_model] = auc

                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=dr0)
                train_loss_kerulsif[i, idx_model] = train_loss
                test_loss_kerulsif[i, idx_model] = test_loss
                auc_kerulsif[i, idx_model] = auc

                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=dr1)
                train_loss_kerkleip[i, idx_model] = train_loss
                test_loss_kerkleip[i, idx_model] = test_loss
                auc_kerkleip[i, idx_model] = auc

                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=dr2)
                train_loss_pu[i, idx_model] = train_loss
                test_loss_pu[i, idx_model] = test_loss
                auc_pu[i, idx_model] = auc

                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=dr3)
                train_loss_ulsif[i, idx_model] = train_loss
                test_loss_ulsif[i, idx_model] = test_loss
                auc_ulsif[i, idx_model] = auc

                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=dr4)
                train_loss_nnpu[i, idx_model] = train_loss
                test_loss_nnpu[i, idx_model] = test_loss
                auc_nnpu[i, idx_model] = auc

                train_loss, test_loss, auc = calc_result(reg, x_train, rating0, x_test, rating1, dr=dr5)
                train_loss_nnulsif[i, idx_model] = train_loss
                test_loss_nnulsif[i, idx_model] = test_loss
                auc_nnulsif[i, idx_model] = auc

                '''

                seed += 1

                np.savetxt('results/train_loss_normal_%s_%s.csv'%(f_name0, f_name1), train_loss_normal, delimiter=',')
                np.savetxt('results/test_loss_normal_%s_%s.csv'%(f_name0, f_name1), test_loss_normal, delimiter=',')
                np.savetxt('results/auc_normal_%s_%s.csv'%(f_name0, f_name1), auc_normal, delimiter=',') 

                np.savetxt('results/train_loss_kerulsif_%s_%s.csv'%(f_name0, f_name1), train_loss_kerulsif, delimiter=',')
                np.savetxt('results/test_loss_kerulsif_%s_%s.csv'%(f_name0, f_name1), test_loss_kerulsif, delimiter=',')
                np.savetxt('results/auc_kerulsif_%s_%s.csv'%(f_name0, f_name1), auc_kerulsif, delimiter=',')

                np.savetxt('results/train_loss_kerkleip_%s_%s.csv'%(f_name0, f_name1), train_loss_kerkleip, delimiter=',')
                np.savetxt('results/test_loss_kerkleip_%s_%s.csv'%(f_name0, f_name1), test_loss_kerkleip, delimiter=',')
                np.savetxt('results/auc_kerkleip_%s_%s.csv'%(f_name0, f_name1), auc_kerkleip, delimiter=',')

                np.savetxt('results/train_loss_pu_%s_%s.csv'%(f_name0, f_name1), train_loss_pu, delimiter=',')
                np.savetxt('results/test_loss_pu_%s_%s.csv'%(f_name0, f_name1), test_loss_pu, delimiter=',')
                np.savetxt('results/auc_pu_%s_%s.csv'%(f_name0, f_name1), auc_pu, delimiter=',')

                np.savetxt('results/train_loss_ulsif_%s_%s.csv'%(f_name0, f_name1), train_loss_ulsif, delimiter=',')
                np.savetxt('results/test_loss_ulsif_%s_%s.csv'%(f_name0, f_name1), test_loss_ulsif, delimiter=',')
                np.savetxt('results/auc_ulsif_%s_%s.csv'%(f_name0, f_name1), auc_ulsif, delimiter=',')

                np.savetxt('results/train_loss_nnpu_%s_%s.csv'%(f_name0, f_name1), train_loss_nnpu, delimiter=',')
                np.savetxt('results/test_loss_nnpu_%s_%s.csv'%(f_name0, f_name1), test_loss_nnpu, delimiter=',')
                np.savetxt('results/auc_nnpu_%s_%s.csv'%(f_name0, f_name1), auc_nnpu, delimiter=',')

                np.savetxt('results/train_loss_nnulsif_%s_%s.csv'%(f_name0, f_name1), train_loss_nnulsif, delimiter=',')
                np.savetxt('results/test_loss_nnulsif_%s_%s.csv'%(f_name0, f_name1), test_loss_nnulsif, delimiter=',')
                np.savetxt('results/auc_nnulsif_%s_%s.csv'%(f_name0, f_name1), auc_nnulsif, delimiter=',')

if __name__ == "__main__":
    main()

