from index_creator import *
import datetime



if __name__ == '__main__':
    ticker_list_US_Tech = ["MCD", "RSG", "SPGI", "TAL", "FISV", "GPN", "MA", "MSCI", "ADBE", "NEE", "WM",
                  "AMZN", "MSFT", "NVDA", "PGR", "VRSN", "GLD"]
    num_samples = len(ticker_list_US_Tech)

    m_cap = [403.02e9, 392.90e9,283.60e9, 243.17e9, 236.79e9, 292.72e9, 231.03e9, 214.99e9, 218.79e9,
           403.02e9, 392.90e9,283.60e11, 243.17e10, 236.79e6, 292.72e8, 331.03e9, 331.03e7]
    rf = 0.03
    risk_aversion_param = 2


    views = [('MSFT', '>', 'AMZN', 0.02),
             ('PGR', '<', 'NVDA', 0.02)]

    obj_portfolio_index = Portfolio_Index()

    #ticker_list_US_Tech = ["GOOG","ADSK","TSLA","V","CRM","GLD","FB",
    #                       "AMZN", "MSFT", "NVDA","LMT","HON","BABA","HDB","ARKW"]

    #ticker_list_quality_beta =["ASIANPAINT.NS","BRITANNIA.NS","COALINDIA.NS","COLPAL.NS",
                              # "GODREJPROP.NS","MARICO.NS","HINDUNILVR.NS","HINDZINC.NS","INFY.NS",
                              # "LTI.NS","TCS.NS","ITC.NS","LT.NS","PIDILITIND.NS","DABUR.NS"]

    equal_weights = [1 / len(ticker_list_US_Tech) for i in range(len(ticker_list_US_Tech))] #Naive 1/N weights
    w_market = np.array(m_cap) / sum(m_cap) #Market Weights for BL
    #prive_weight = [0.0365,0.035,0.0515,.1259,.0471,.0639,.055,.0718,.0672,.0373,.0356,
    #            .0831,.059,.1335,.047,.0538]
    #prive_weight = [0.0365, 0.035, 0.0515, .1259, .0471, .0639, .055, .0718, .0672, .0373, .0356,
    #                .0831, .059, .1335, .047, .0438, .02]

    prive_weight = [0.0365, 0.035, 0.0515, .1259, .0471, .0639, .055, .0718, .0672, .0373, .0356,
                    .0831, .059, .1335, .047, .0438, .02]
    start_date = datetime.datetime(2015,1,1) #In sample data period
    end_date = datetime.datetime(2020, 1, 1)
    start_date_OOS = datetime.datetime(2020,1,1) #Holdout sample data period
    end_date_OOS = datetime.datetime(2020, 8, 11)

    price_data = obj_portfolio_index.data_import_yf(ticker_list_US_Tech,start_date,end_date) #method to download data from YFinance
    ret_ = obj_portfolio_index.create_ret_series(ticker_list_US_Tech,price_series=price_data) #Return calcs from price series
    price_data_OOS = obj_portfolio_index.data_import_yf(ticker_list_US_Tech,start_date_OOS,end_date_OOS) #method to download data for Hold-Out from YFinance
    ret_OOS = obj_portfolio_index.create_ret_series(ticker_list_US_Tech,price_series=price_data_OOS) #Hold-out Return calcs from price series

    #Summary Statistics of return series
    mean_ret = ret_.mean(axis=0) #Mean of returns
    obj_portfolio_index.heatmap_2D(ret_.corr()) #Calling method to plot Correlation Heatmap
    est_cov = obj_portfolio_index.est_covar(ret_) #Robust covariance estimate
    #obj_portfolio_index.heatmap_2D(est_cov) #Calling method to plot Covariance Heatmap
    #obj_portfolio_index.eig_plot(est_cov) #Call Method to plot PCA from eigen values for robust covar

    sample_cov = obj_portfolio_index.sample_covariance(ret_) #calling method to calculate sample covariance estimate
    Pi_adj = obj_portfolio_index.BL_prepare(w_market, mean_ret, est_cov,views,ticker_list_US_Tech,rf) #calling method to prepare for Black Litterman


    #Calling methods for calculating optimal weights using "Robust Covariance" based on mean-variance and Sharpe Obj Func & BL equivalent of both
    print('----------------------------------Sharpe Starts----------------------------------')
    opt_weights_sharpe = obj_portfolio_index.max_sharpe_ratio(mean_ret,est_cov,rf,equal_weights)
    print('----------------------------------Mean-Variance Starts------------------------------')
    opt_weights_mean_var = obj_portfolio_index.mean_variance(mean_ret,est_cov,equal_weights,risk_aversion_param)
    print('----------------------------------Mean-Variance-BL Starts------------------------------')
    opt_weights_BL_mean_var = obj_portfolio_index.mean_variance(Pi_adj, est_cov,w_market,risk_aversion_param)
    print('----------------------------------Sharpe-BL Starts----------------------------------')
    opt_weights_BL_sharpe = obj_portfolio_index.max_sharpe_ratio(Pi_adj, est_cov, rf,w_market)

    #Calculating Weighted return series from optimal weights to create Model Portfolio/Index
    weighted_ret_sharpe = ret_*opt_weights_sharpe['x']
    weighted_ret_mean_var = ret_*opt_weights_mean_var['x']
    weighted_ret_BL_mean_var = ret_ * opt_weights_BL_mean_var['x']
    weighted_ret_BL_sharpe = ret_ * opt_weights_BL_sharpe['x']
    eq_weighted_Ret = ret_ * equal_weights
    weighted_ret_priv_ = ret_*prive_weight

    #Hold-Out Period- Calculating Weighted return series from optimal weights to create Model Portfolio/Index
    weighted_ret_sharpe_OOS = ret_OOS*opt_weights_sharpe['x']
    weighted_ret_mean_var_OOS = ret_OOS*opt_weights_mean_var['x']
    weighted_ret_BL_mean_var_OOS = ret_OOS * opt_weights_BL_mean_var['x']
    weighted_ret_BL_sharpe_OOS = ret_OOS * opt_weights_BL_sharpe['x']
    eq_weighted_Ret_OOS = ret_OOS * equal_weights
    weighted_ret_priv_OOS = ret_OOS*prive_weight

    # create Model Portfolio/Index from weighted return series above
    sharpe_index_ = obj_portfolio_index.create_index_list(ret_,[],opt_weights_sharpe['x'])
    mean_var_index_ = obj_portfolio_index.create_index_list(ret_,[],opt_weights_mean_var['x'])
    BL_mean_var_index_ = obj_portfolio_index.create_index_list(ret_,[],opt_weights_BL_mean_var['x'])
    BL_sharpe_index_ = obj_portfolio_index.create_index_list(ret_,[],opt_weights_BL_sharpe['x'])
    eq_weight_index_ = obj_portfolio_index.create_index_list(ret_,[],equal_weights)
    priv_index_ = obj_portfolio_index.create_index_list(ret_,[],prive_weight)

    #Hold-Out-create Model Portfolio/Index from weighted return series above
    sharpe_index_OOS = obj_portfolio_index.create_index_list(ret_OOS,sharpe_index_,opt_weights_sharpe['x'])
    mean_var_index_OOS = obj_portfolio_index.create_index_list(ret_OOS,mean_var_index_,opt_weights_mean_var['x'])
    BL_mean_var_index_OOS = obj_portfolio_index.create_index_list(ret_OOS,BL_mean_var_index_,opt_weights_BL_mean_var['x'])
    BL_sharpe_index_OOS = obj_portfolio_index.create_index_list(ret_OOS,BL_sharpe_index_,opt_weights_BL_sharpe['x'])
    eq_weight_index_OOS = obj_portfolio_index.create_index_list(ret_OOS,eq_weight_index_,equal_weights)
    priv_index_OOS = obj_portfolio_index.create_index_list(ret_OOS,priv_index_,prive_weight)

    #Combine Model Portfolio/Index in one dictionary for easy reference and use
    index_dict ={'sharpe_index_' : sharpe_index_,'mean_var_index_':mean_var_index_,'BL_mean_var_index_':BL_mean_var_index_,
                 'BL_sharpe_index_':BL_sharpe_index_,'eq_weight_index_':eq_weight_index_,'priv_index_':priv_index_}

    #Hold-Out- Combine Model Portfolio/Index in one dictionary for easy reference and use
    index_dict_OOS ={'sharpe_index_' : sharpe_index_OOS,'mean_var_index_':mean_var_index_OOS,
                     'BL_mean_var_index_':BL_mean_var_index_OOS, 'BL_sharpe_index_':BL_sharpe_index_OOS,
                     'eq_weight_index_':eq_weight_index_OOS,'priv_index_':priv_index_OOS}

    #Combine optimal weights for diff obj functions in one dictionary for easy reference and use
    weight_dict ={'sharpe_index_' : opt_weights_sharpe['x'],'mean_var_index_':opt_weights_mean_var['x'],
                  'BL_mean_var_index_':opt_weights_BL_mean_var['x'],
                 'BL_sharpe_index_':opt_weights_BL_sharpe['x'],'eq_weight_index_':equal_weights}


    #Calling plot method to plot all indexes/model portfolios in one chart for In-Sample and Hold-Out Sample
    obj_portfolio_index.plot_performance(x_axis=ret_.index, x_OOS_axis=ret_OOS.index, y_axis=index_dict,y_OOS_axis =index_dict_OOS )

    # # Calling all same methods as above for calculating optimal weights using "Sample Covariance" based on mean-variance and Sharpe Obj & BL equivalent of both
    # print('------------------------------------Stats with Sample Covariance--------------------------')
    #
    # print('----------------------------------Sharpe Starts----------------------------------')
    # opt_weights_sharpe_1 = obj_portfolio_index.max_sharpe_ratio(mean_ret, sample_cov, rf, equal_weights)
    # print('----------------------------------Mean-Variance Starts------------------------------')
    # opt_weights_mean_var_1 = obj_portfolio_index.mean_variance(mean_ret, sample_cov, equal_weights, risk_aversion_param)
    # print('----------------------------------Mean-Variance-BL Starts------------------------------')
    # opt_weights_BL_mean_var_1 = obj_portfolio_index.mean_variance(Pi_adj, sample_cov, w_market, risk_aversion_param)
    # print('----------------------------------Sharpe-BL Starts----------------------------------')
    # opt_weights_BL_sharpe_1 = obj_portfolio_index.max_sharpe_ratio(Pi_adj, sample_cov, rf, w_market)
    #
    # weighted_ret_sharpe_1 = ret_ * opt_weights_sharpe_1['x']
    # weighted_ret_mean_var_1 = ret_ * opt_weights_mean_var_1['x']
    # weighted_ret_BL_mean_var_1 = ret_ * opt_weights_BL_mean_var_1['x']
    # weighted_ret_BL_sharpe_1 = ret_ * opt_weights_BL_sharpe_1['x']
    # eq_weighted_Ret_ = ret_ * equal_weights
    # weighted_ret_priv_ = ret_ * prive_weight
    #
    # weighted_ret_sharpe_OOS_1 = ret_OOS * opt_weights_sharpe_1['x']
    # weighted_ret_mean_var_OOS_1 = ret_OOS * opt_weights_mean_var_1['x']
    # weighted_ret_BL_mean_var_OOS_1 = ret_OOS * opt_weights_BL_mean_var_1['x']
    # weighted_ret_BL_sharpe_OOS_1 = ret_OOS * opt_weights_BL_sharpe_1['x']
    # eq_weighted_Ret_OOS_1 = ret_OOS * equal_weights
    # weighted_ret_priv_OOS = ret_OOS * prive_weight
    #
    # sharpe_index_1 = obj_portfolio_index.create_index_list(weighted_ret_sharpe_1, [])
    # mean_var_index_1 = obj_portfolio_index.create_index_list(weighted_ret_mean_var_1, [])
    # BL_mean_var_index_1 = obj_portfolio_index.create_index_list(weighted_ret_BL_mean_var_1, [])
    # BL_sharpe_index_1= obj_portfolio_index.create_index_list(weighted_ret_BL_sharpe_1, [])
    # eq_weight_index_ = obj_portfolio_index.create_index_list(eq_weighted_Ret, [])
    # priv_index_ = obj_portfolio_index.create_index_list(weighted_ret_priv_, [])
    #
    # sharpe_index_OOS_1 = obj_portfolio_index.create_index_list(weighted_ret_sharpe_OOS_1, sharpe_index_1)
    # mean_var_index_OOS_1 = obj_portfolio_index.create_index_list(weighted_ret_mean_var_OOS_1, mean_var_index_1)
    # BL_mean_var_index_OOS_1 = obj_portfolio_index.create_index_list(weighted_ret_BL_mean_var_OOS_1, BL_mean_var_index_1)
    # BL_sharpe_index_OOS_1 = obj_portfolio_index.create_index_list(weighted_ret_BL_sharpe_OOS_1, BL_sharpe_index_1)
    # eq_weight_index_OOS = obj_portfolio_index.create_index_list(eq_weighted_Ret_OOS, eq_weight_index_)
    # priv_index_OOS = obj_portfolio_index.create_index_list(weighted_ret_priv_OOS, priv_index_)
    #
    # index_dict_1 = {'sharpe_index_': sharpe_index_1, 'mean_var_index_': mean_var_index_1,
    #               'BL_mean_var_index_': BL_mean_var_index_1,
    #               'BL_sharpe_index_': BL_sharpe_index_1, 'eq_weight_index_': eq_weight_index_,
    #               'priv_index_': priv_index_}
    #
    # index_dict_OOS_1 = {'sharpe_index_': sharpe_index_OOS_1, 'mean_var_index_': mean_var_index_OOS_1,
    #                   'BL_mean_var_index_': BL_mean_var_index_OOS_1, 'BL_sharpe_index_': BL_sharpe_index_OOS_1,
    #                   'eq_weight_index_': eq_weight_index_OOS, 'priv_index_': priv_index_OOS}
    #
    # weight_dict_1 = {'sharpe_index_': opt_weights_sharpe_1['x'], 'mean_var_index_': opt_weights_mean_var_1['x'],
    #                'BL_mean_var_index_': opt_weights_BL_mean_var_1['x'],
    #                'BL_sharpe_index_': opt_weights_BL_sharpe_1['x'], 'eq_weight_index_': equal_weights}
    #
    # #print(weight_dict)
    #
    # obj_portfolio_index.plot_performance(x_axis=ret_.index, x_OOS_axis=ret_OOS.index, y_axis=index_dict_1,
    #                                      y_OOS_axis=index_dict_OOS_1)
    #
    # obj_portfolio_index.eig_plot(cov_mat=sample_cov)  # Call Method to plot PCA from eigen values for sample cov
