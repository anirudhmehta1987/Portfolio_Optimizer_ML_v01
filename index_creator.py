import numpy as np
import pandas as pd
import yfinance as yf
import scipy.optimize as sco
import plotly.graph_objects as go
import seaborn as sns
from direct_kernel import DirectKernel
from numpy.linalg import inv
import matplotlib.pyplot as plt







class Portfolio_Index(object): #Class that contains all methods for Portfolio construction

     def __init__(self):
         self.data = None


     def data_import_yf(self,ticker_list,start,end):
         #Method to import data from yfinance
         #:ticker_list- List of symbols to download from Yfinance
         #:start- start date of data download
         #:end- End date of data download
        close_dataframe  = pd.DataFrame()

        
        data = yf.download(ticker_list, start=start, end=end,group_by = "ticker")
        
        for index, ticker in enumerate(ticker_list):
            #data = yf.download(ticker, start=start, end=end)
            close_dataframe[ticker]=data[ticker]['Close']


        return close_dataframe

     def create_ret_series(self,ticker_list,price_series):
         #Method to create return series from price data from yfinance
         #:ticker_list- List of symbols to download from Yfinance
         #:price_series- Dataframe with price series
         return_series = pd.DataFrame()

         data_dataframe = price_series

         for index, ticker in enumerate(ticker_list):
            ticker_ret = 'Ret_' + ticker
            #Ret_ticker_ret = 'Ret_' + ticker
            data_dataframe[ticker_ret] = np.log(data_dataframe[ticker] / data_dataframe[ticker].shift(1))
            return_series[ticker_ret] = np.log(data_dataframe[ticker] / data_dataframe[ticker].shift(1))


         return_series.fillna(0, inplace=True)
         #self.excel_export(data_dataframe)
         
         return return_series

     def create_index_list(self,ret_,list_ref,weights):
         #Method to create index/model portfolio from return series
         #:ret_- Dataframe with "weighted" return series
         #:list_ref- None for In-Sample and contains last value of In-sample to create index for Hold-Out sample
         
         weighted_ret = np.dot(ret_.to_numpy(),np.asarray(weights).T)
         
         if not list_ref:
            new_index = [100] * len(weighted_ret)
         else:
             new_index = [list_ref[-1] ] * len(weighted_ret) #To set the last value of in_sample index as first value for Hold-Out sample


         #summed up all weight adjusted returns to calculate model portfolio return i.e. Pi = w1*R1 + w2*R2+ ...
         for idx in range(len(weighted_ret)):
             new_index[idx] = new_index[idx-1]*(1+weighted_ret[idx])


         return new_index   #returns a list of model portfolio index values



     def to_excel(self,data_dataframe,filename): #Method to create an excel file from dataframe and save in the proj working directory- Not used

         data_dataframe.to_excel('./'+ filename +'.xlsx', engine='xlsxwriter')




     def heatmap_2D(self,matrix_2D): #Method to create 2D heatmap

         #:matrix_2D- 2D array with correlation or covariance matrix
         sns.heatmap(matrix_2D, annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm',
                     linewidths=3, linecolor='black')
         plt.show()

     def est_covar(self,ret_):
         #Method to calculate robust covariance matrix using Ledolt Wolf shrinkage technique
         #:ret_- Dataframe with return series of various tickers
         X = ret_
         X = X - X.mean(axis=0)
         np.set_printoptions(formatter={'float_kind': '{:f}'.format})
         dk_obj = DirectKernel(X)
         est_cov = dk_obj.estimate_cov_matrix()
         return est_cov

     def sample_covariance(self,ret_):
         #Method to calculate sample covariance matrix
         #:ret_- Dataframe with return series of various tickers
         X = ret_
         X = X - X.mean(axis=0)
         sample_matrix = X.T @ X / float(len(ret_.columns) - 1)
         #print('sample_matrix :')
         #print(sample_matrix.to_numpy())
         return sample_matrix.to_numpy()

     def portfolio_annualised_performance(self,weights, mean_returns, cov_matrix):
         #Method to calculate annualized performance (risk and return) of a series
         #:weights- input weights for each ticker
         #:mean_returns- series of mean returns
         #:cov_matrix- 2D matrix of covariance


         returns = np.sum(mean_returns * weights) * 252
         std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
         #print('returns:', returns, 'S.D.:',std, 'Sharpe:',(returns-0.03)/std)
         return std, returns


     def max_sharpe_ratio(self,mean_returns, cov_matrix, risk_free_rate,weights):
         #Method to calculate annualized performance (risk and return) of a series
         #:weights- input weights for each ticker
         #:mean_returns- series of mean returns
         #:cov_matrix- 2D matrix of covariance
         #:risk_free_rate- Constant risk free rate

         def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
             p_std, p_ret = self.portfolio_annualised_performance(weights, mean_returns, cov_matrix)
             return -(p_ret - risk_free_rate) / p_std #Objective function that minimize negative sharpe ratio

         num_assets = len(mean_returns)
         args = (mean_returns, cov_matrix, risk_free_rate)

         constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Sum of weights = 1
         seed_weights = weights
         bound = (0.035,0.175) #Weight bounds
         bounds = tuple(bound for asset in range(num_assets))
         result = sco.minimize(neg_sharpe_ratio,seed_weights , args=args,
                               method='SLSQP', bounds=bounds, constraints=constraints)
         return result



     def mean_variance(self,mean_returns, cov_matrix,weights,lamda):
         #Method to calculate annualized performance (risk and return) of a series
         #:weights- input weights for each ticker
         #:mean_returns- series of mean returns
         #:cov_matrix- 2D matrix of covariance
         #:lamda- risk tolerance parameter
         def portfolio_volatility(weights, mean_returns, cov_matrix):
             p_std, p_ret = self.portfolio_annualised_performance(weights, mean_returns, cov_matrix)
             return (p_std*0.5*lamda - p_ret) #Obj function for mean-variance opt

         #def portfolio_volatility2(weights, mean_returns, cov_matrix):
         #    return self.portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]


         num_assets = len(mean_returns)
         args = (mean_returns, cov_matrix)
         seed_weights = weights
         constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) #sum of weights = 1
         bound = (0.035,0.175) #weight bounds
         bounds = tuple(bound for asset in range(num_assets))

         result = sco.minimize(portfolio_volatility, seed_weights, args=args,
                               method='SLSQP', bounds=bounds, constraints=constraints)

         return result

     def create_views_and_link_matrix(self,names, views):
         #Method to create viwe and link matrix for Black-Litterman
         #:names- input ticker list where investor has views
         #:views- list with views

         r, c = len(views), len(names)
         Q = [views[i][3] for i in range(r)]  # view matrix
         P = np.zeros([r, c])
         nameToIndex = dict()
         for i, n in enumerate(names):
             nameToIndex[n] = i
         for i, v in enumerate(views):
             name1, name2 = views[i][0], views[i][2]
             P[i, nameToIndex[name1]] = +1 if views[i][1] == '>' else -1
             P[i, nameToIndex[name2]] = -1 if views[i][1] == '>' else +1
         return np.array(Q), P


     def BL_prepare(self,w_market,mean_ret,est_cov,views,ticker_list,rf):
         mean, std = self.portfolio_annualised_performance(w_market, mean_ret, est_cov)
         lmb = (mean - rf) / (std)  # Calculate risk aversion
         Pi = np.dot(np.dot(lmb, est_cov), w_market)  # Calculate equilibrium excess returns
         Q, P = self.create_views_and_link_matrix(ticker_list, views)
         tau = .025  # scaling factor
         # Calculate omega - uncertainty matrix about views
         omega = np.dot(np.dot(np.dot(tau, P), est_cov), np.transpose(P))  # 0.025 * P * C * transpose(P)
         # Calculate equilibrium excess returns with views incorporated
         sub_a = inv(np.dot(tau, est_cov))
         sub_b = np.dot(np.dot(np.transpose(P), inv(omega)), P)
         sub_c = np.dot(inv(np.dot(tau, est_cov)), Pi)
         sub_d = np.dot(np.dot(np.transpose(P), inv(omega)), Q)
         Pi_adj = np.dot(inv(sub_a + sub_b), (sub_c + sub_d)) #Calculate the Pi adjusted returns based on views

         return Pi_adj

     def plot_performance(self,x_axis,x_OOS_axis,y_axis,y_OOS_axis):
         #Method to plot plotly charts
         #:x_axis, y_axis- input datafrmae for insample x axis and y axis
         #:x_OOS, y_OOS- input datafrmae for hold-out sample x axis and y axis
         fig = go.Figure()
         cus_width = 2
         fig.add_trace(go.Scatter(x=x_axis, y=y_axis['sharpe_index_'],
                                  mode='lines',
                                  name='sharpe_index_',line=dict(color='firebrick', width=cus_width)))

         fig.add_trace(go.Scatter(x=x_axis, y=y_axis['mean_var_index_'],
                                  mode='lines',
                                  name='mean_var_index_',line=dict(color='purple', width=cus_width)))

         fig.add_trace(go.Scatter(x=x_axis, y=y_axis['BL_mean_var_index_'],
                                  mode='lines', name='BL_mean_var_index_',line=dict(color='orange', width=cus_width)))

         fig.add_trace(go.Scatter(x=x_axis, y=y_axis['BL_sharpe_index_'],
                                  mode='lines',
                                  name='BL_sharpe_index_',line=dict(color='green', width=cus_width)))

         fig.add_trace(go.Scatter(x=x_axis, y=y_axis['eq_weight_index_'],
                                  mode='lines', name='eq_weight_index_',line=dict(color="#0000ff", width=cus_width)))

         fig.add_trace(go.Scatter(x=x_axis, y=y_axis['priv_index_'],
                                  mode='lines', name='priv_index_',line=dict(color="#ffe476",width=cus_width)))

         #Chart for OOS
         fig.add_trace(go.Scatter(x=x_OOS_axis, y=y_OOS_axis['sharpe_index_'],
                                  mode='lines',
                                  name='sharpe_index_OOS',line=dict(color='firebrick', width=cus_width)))

         fig.add_trace(go.Scatter(x=x_OOS_axis, y=y_OOS_axis['mean_var_index_'],
                                  mode='lines',
                                  name='mean_var_index_OOS',line=dict(color='purple', width=cus_width)))

         fig.add_trace(go.Scatter(x=x_OOS_axis, y=y_OOS_axis['BL_mean_var_index_'],
                                  mode='lines', name='BL_mean_var_index_OOS',line=dict(color='orange', width=cus_width)))

         fig.add_trace(go.Scatter(x=x_OOS_axis, y=y_OOS_axis['BL_sharpe_index_'],
                                  mode='lines',
                                  name='BL_sharpe_index_OOS',line=dict(color='green', width=cus_width)))

         fig.add_trace(go.Scatter(x=x_OOS_axis, y=y_OOS_axis['eq_weight_index_'],
                                  mode='lines', name='eq_weight_index_OOS',line=dict(color="#0000ff",width=cus_width)))

         fig.add_trace(go.Scatter(x=x_OOS_axis, y=y_OOS_axis['priv_index_'],
                                  mode='lines', name='priv_index_OOS',line=dict(color="#ffe476",width=cus_width)))

         fig.show()


     def eig_plot(self,cov_mat):
         #Method to plot eigen value and PCA of EiV
         #:cov_mat- 2D Input matrix robust or sample

         fig = go.Figure()
         eig_vals, eig_vecs = np.linalg.eig(cov_mat)
         #print('Eigenvectors \n%s' % eig_vecs)
         #print('\nEigenvalues \n%s' % eig_vals)
         # Make a list of (eigenvalue, eigenvector) tuples
         eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

         # Sort the (eigenvalue, eigenvector) tuples from high to low
         eig_pairs.sort()
         eig_pairs.reverse()

         # Visually confirm that the list is correctly sorted by decreasing eigenvalues
         #print('Eigenvalues in descending order:')
         #for i in eig_pairs:
         #    print(i[0])

         tot = sum(eig_vals)
         var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
         cum_var_exp = np.cumsum(var_exp)
         #print(var_exp)
         #(cum_var_exp)
         trace1 = dict(
             type='bar',
             x=['PC %s' % i for i in range(1, 6)],
             y=var_exp,
             name='Individual'
         )

         trace2 = dict(
             type='scatter',
             x=['PC %s' % i for i in range(1, 6)],
             y=cum_var_exp,
             name='Cumulative'
         )

         fig.add_trace(trace1)

         fig.add_trace(trace2)


         fig.update_layout(
             title_text='Sampled Results',  # title of plot
             xaxis_title_text='Value',  # xaxis label
             yaxis_title_text='Percentage',  # yaxis label
             bargap=0.1,  # gap between bars of adjacent location coordinates
             bargroupgap=0.1  # gap between bars of the same location coordinates
         )

         fig.show()


