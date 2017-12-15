# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 19:43:29 2017

@author: MANOJ
"""
import warnings
import numpy as np
import pandas as pd
import glob, os, re
from os.path import join
from pytz import UTC
from pytz import timezone
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from sknn.mlp import Regressor, Layer
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn import metrics

from timeit import default_timer as timer
import time






raw_csv_path = r'C:/Users/MANOJ/Dropbox/Manoj Thesis/NILMTK DataSets/Pecan Street (Wiki Energy)/Exported Data From interactive Data/Building 6636/knime/Energy consumption Austin/2014/raw_8236'
disaggregated_csv_path = r'C:/Users/MANOJ/Dropbox/Manoj Thesis/NILMTK DataSets/Pecan Street (Wiki Energy)/Exported Data From interactive Data/Building 6636/knime/Energy consumption Austin/2014'
irradiance_path = r'C:/Users/MANOJ/Dropbox/Manoj Thesis/NILMTK DataSets/Pecan Street (Wiki Energy)/Exported Data From interactive Data/Building 6636/knime/Energy consumption Austin/2014'
weather_path = r'C:/Users/MANOJ/Dropbox/Manoj Thesis/NILMTK DataSets/Pecan Street (Wiki Energy)/Exported Data From interactive Data/Building 6636/knime/weather_austin'


def get_files_by_file_size(dirname, reverse=True):
    """ Return list of file paths in directory sorted by file size """

    # Get list of files
    filepaths = []
    for basename in os.listdir(dirname):
        filename = os.path.join(dirname, basename)
        if os.path.isfile(filename):
            filepaths.append(filename)

    # Re-populate list with filename, size tuples
    for i in xrange(len(filepaths)):
        filepaths[i] = (filepaths[i], os.path.getsize(filepaths[i]))

    # Sort list by file size
    # If reverse=True sort from largest to smallest
    # If reverse=False sort from smallest to largest
    filepaths.sort(key=lambda filename: filename[1], reverse=reverse)

    # Re-populate list with just filenames
    for i in xrange(len(filepaths)):
        filepaths[i] = filepaths[i][0]

    return filepaths
    
def load_raw_data(path):
    #noting all csv files
    allFiles_raw = glob.glob(path+"/*.csv")
    #reading all the files with file names which contain the key word 'minutely_data'
    df_raw_read = (pd.read_csv(file_,index_col=None,names=[re.search('minutely_data_(.+?)_.csv',file_).group(1)],header=0) for file_ in allFiles_raw)
    df_raw = pd.concat(df_raw_read,axis=1)
    df_raw.index = pd.to_datetime(df_raw.index)
    return df_raw
def load_disaggregated_data(disaggregated_file_name):
    disagg_file = join(disaggregated_csv_path + disaggregated_file_name)
    #removing the time zone information from index and using localhour
    df_disaggregated = pd.read_csv(disagg_file,index_col=0,parse_dates=[0],date_parser=lambda x: pd.to_datetime(x.rpartition('-')[0]))
    df_disaggregated = df_disaggregated.rename(columns={col: col.split("type='")[1].replace("', instance=",'').replace(')])','') for col in df_disaggregated.columns})
    return df_disaggregated
def load_weather_data(path,weather_file_name,column,index):
    weather_file = join(path +weather_file_name )
    #removing the time zone information from index and using localhour
    df = pd.read_csv(weather_file,sep=';',index_col=index,parse_dates=[index], date_parser=lambda x: pd.to_datetime(x.rpartition('-')[0]))
    #df.index = pd.to_datetime(df.index)#.tz_localize(UTC)#.tz_convert('US/Eastern')
    df = df[column]
    return df
def load_irradiance_data(path,irradiance_file_name,column,index):
    irradiance_file = join(path+irradiance_file_name)
    df = pd.read_csv(irradiance_file,sep=';',index_col=index)
    df.index = pd.to_datetime(df.index)
    df = df[column]
    return df
def groupby_hourly(df):
    df_hourly = df.resample(rule='H')
    return df_hourly
def general_pandas_plot(df,col1,col2,col3):

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    # Twin the x-axis twice to make independent y-axes.
    axes = [ax1, ax2, ax3]
    # Make some space on the right side for the extra y-axis.
    fig.subplots_adjust(right=0.9)
    # Move the last y-axis spine over to the right by 20% of the width of the axes
    axes[-1].spines['right'].set_position(('axes', 1.05))
    # To make the border of the right-most axis visible, we need to turn the frame
    # on. This hides the other plots, however, so we need to turn its fill off.
    axes[-1].set_frame_on(True)
    axes[-1].patch.set_visible(False)
    
    x = df.index
    y1 = df[col1]
    y2 = df[col2]
    y3 = df[col3]
    ax1.plot(x,y1,'-g',label=col1);ax1.set_ylabel(col1,color='g')#;ax1.legend(loc=0)
    ax2.plot(x,y2,'-b',label=col2);ax2.set_ylabel(col2,color='b')#;ax2.legend(loc=1)
    ax3.plot(x,y3,'-r',label=col3);ax3.set_ylabel(col3,color='r')#;ax3.legend(loc=2)
    ax1.set_xlabel('Date Time')
#    plt.legend(loc=0)
    plt.show()
'''#######-------Features creation--------------############'''

def features_creation(df):
    df['sin_hour'] = np.sin((df.index.hour)*2*np.pi/24)
    df['cos_hour'] = np.cos((df.index.hour)*2*np.pi/24)#later try 24 vector binary format
    df['hour'] = df.index.hour # 0 to 23
    df['day_of_week'] = df.index.dayofweek #Monday = 0, sunday = 6
    df['weekend'] = [ 1 if day in (5, 6) else 0 for day in df.index.dayofweek ] # 1 for weekend and 0 for weekdays
    df['month'] = df.index.month
    df['week_of_year'] = df.index.week
    # day = 1 if(10Hrs -19Hrs) and Night = 0 (otherwise)
    df['day_night'] = [1 if day<20 and day>9 else 0 for day in df.index.hour ]
    return df 
def occupancy(df):
    df_nights = df[df['day_night']==0]
    df_nights_grouped = df_nights.groupby([df_nights.index.hour]).mean()
    df_nights_grouped.columns = [str(col) + '_hourly_night_mean' for col in df_nights_grouped.columns]
    df_nights_grouped_mean = df_nights_grouped.mean(axis=0)
    df_nights_grouped_mean_df = pd.DataFrame(df_nights_grouped_mean)
    df_nights_grouped_mean_df_transposed = df_nights_grouped_mean_df.transpose()
    joined =df.join(df_nights_grouped_mean_df_transposed,how='outer')
    
    return joined
    
def lag_column(df,column_names,lag_period=1):
#df              > pandas dataframe
#column_names    > names of column/columns as a list
#lag_period      > number of steps to lag ( +ve or -ve) usually postive 
#to include past values for current row 
    for column_name in column_names:
        column_name = [str(column_name)]
        for i in np.arange(1,lag_period+1,1):
            new_column_name = [col +'_'+str(i) for col in column_name]
            df[new_column_name]=(df[column_name]).shift(i)
    return df
    
def plot_corr(df,selected_columns=None,annot=False):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.
    Input:
        df: pandas DataFrame
        selected_columns: if you want to select columns containing 1 or more substring
        Example: selected_columns="air" - selects columns air conditioner and such
        selected columns = "air|temp" - selects columns containing air conditioner and temperature'''
        
    if selected_columns==None:
        df_correlation = df.corr()
    else:
        df_correlation = df[df.filter(regex=selected_columns).columns].corr()
    fig = plt.figure()
    plot = fig.add_axes()
    plot = sns.heatmap(df_correlation, annot=annot)
    plot.xaxis.tick_top() 
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.show()
    
def plot_model(target,y_train,y_test,training_predictions,test_prediction):
    axis = target.index[0:len(y_train)]
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.scatter(y_train,training_predictions)
    plt.title('Prediction Vs Truth Fit')
    plt.ylabel('Truth')
    plt.xlabel('Predictions')
    
    plt.subplot(2, 1, 2)
    plt.plot(axis,training_predictions,color='red',label='Prediction')
    plt.plot(axis,y_train,color = 'blue', label='Truth')
    plt.title('Prediction Vs Truth - Model Fit on training data')
    plt.legend(loc='best')
    plt.xlabel('time (s)')
    plt.ylabel('Consumption')
    plt.show()

    plt.figure()
    plt.plot(np.arange(0,len(y_test),1),test_prediction,label='Predictions' )
    plt.plot(np.arange(0,len(y_test),1),y_test,label='Truth' )
    plt.title('Prediction Vs Truth on Test Data')
    plt.legend(loc='best')
    plt.xlim([0,len(y_test)])
    plt.show()
    
def linear_regression(features,target,test_size_percent=0.2,cv_split=5):
    ''' Features -> Pandas Dataframe with attributes as columns
        target -> Pandas Dataframe with target column for prediction
        Test_size_percent -> Percentage of data point to be used for testing'''
    X_array = features.as_matrix()
    y_array = target.as_matrix()    
    ols = linear_model.LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X_array, y_array.T.squeeze(), test_size=test_size_percent, random_state=4)
#    model = ols.fit(X_train, y_train)
    ols.fit(X_train, y_train)
#    test_prediction_model = ols.predict(X_test)
    tscv = TimeSeriesSplit(cv_split)
    
    training_score = cross_val_score(ols,X_train,y_train,cv=tscv.n_splits) 
    testing_score = cross_val_score(ols,X_test,y_test,cv=tscv.n_splits)
    print"Cross-val Training score:", training_score.mean()
#    print"Cross-val Testing score:", testing_score.mean()
    training_predictions = cross_val_predict(ols,X_train,y_train,cv=tscv.n_splits)
    testing_predictions = cross_val_predict(ols,X_test,y_test,cv=tscv.n_splits)
    
    training_accuracy = metrics.r2_score(y_train,training_predictions) 
#    test_accuracy_model = metrics.r2_score(y_test,test_prediction_model)
    test_accuracy = metrics.r2_score(y_test,testing_predictions)
    
#    print"Cross-val predicted accuracy:", training_accuracy
    print"Test-predictions accuracy:",test_accuracy

    plot_model(target,y_train,y_test,training_predictions,testing_predictions)
    return ols
    
    
def neural_net(features,target,test_size_percent=0.2,cv_split=3,n_iter=100,learning_rate=0.01):
    '''Features -> Pandas Dataframe with attributes as columns
        target -> Pandas Dataframe with target column for prediction
        Test_size_percent -> Percentage of data point to be used for testing'''
    scale=preprocessing.MinMaxScaler()
    X_array = scale.fit_transform(features)
    y_array = scale.fit_transform(target)
    mlp = Regressor(layers=[Layer("Rectifier",units=5), # Hidden Layer1
                            Layer("Rectifier",units=3)  # Hidden Layer2
                            ,Layer("Linear")],     # Output Layer
                        n_iter = n_iter, learning_rate=0.01)
    X_train, X_test, y_train, y_test = train_test_split(X_array, y_array.T.squeeze(), test_size=test_size_percent, random_state=4)
    mlp.fit(X_train,y_train)
    test_prediction = mlp.predict(X_test)
    tscv = TimeSeriesSplit(cv_split)
    
    training_score = cross_val_score(mlp,X_train,y_train,cv=tscv.n_splits) 
    testing_score = cross_val_score(mlp,X_test,y_test,cv=tscv.n_splits)
    print"Cross-val Training score:", training_score.mean()
#    print"Cross-val Testing score:", testing_score.mean()
    training_predictions = cross_val_predict(mlp,X_train,y_train,cv=tscv.n_splits)
    testing_predictions = cross_val_predict(mlp,X_test,y_test,cv=tscv.n_splits)
    
    training_accuracy = metrics.r2_score(y_train,training_predictions) 
#    test_accuracy_model = metrics.r2_score(y_test,test_prediction_model)
    test_accuracy = metrics.r2_score(y_test,testing_predictions)
    
#    print"Cross-val predicted accuracy:", training_accuracy
    print"Test-predictions accuracy:",test_accuracy

    plot_model(target,y_train,y_test,training_predictions,testing_predictions)
    return mlp
    
    
    
def svm_regressor(features,target,test_size_percent=0.2,cv_split=5):
    
    scale=preprocessing.MinMaxScaler()
    X_array = scale.fit_transform(features)
    y_array = scale.fit_transform(target)  
    X_train, X_test, y_train, y_test = train_test_split(X_array, y_array.T.squeeze(), test_size=test_size_percent, random_state=4)
    svr = SVR(kernel='rbf',C=10,gamma=1)
    svr.fit(X_train,y_train.ravel())
    test_prediction = svr.predict(X_test)
    tscv = TimeSeriesSplit(cv_split)
    
    training_score = cross_val_score(svr,X_train,y_train,cv=tscv.n_splits) 
    testing_score = cross_val_score(svr,X_test,y_test,cv=tscv.n_splits)
    print"Cross-val Training score:", training_score.mean()
#    print"Cross-val Testing score:", testing_score.mean()
    training_predictions = cross_val_predict(svr,X_train,y_train,cv=tscv.n_splits)
    testing_predictions = cross_val_predict(svr,X_test,y_test,cv=tscv.n_splits)
    
    training_accuracy = metrics.r2_score(y_train,training_predictions) 
#    test_accuracy_model = metrics.r2_score(y_test,test_prediction_model)
    test_accuracy = metrics.r2_score(y_test,testing_predictions)
    
#    print"Cross-val predicted accuracy:", training_accuracy
    print"Test-predictions accuracy:",test_accuracy
    return svr

#    plot_model(target,y_train,y_test,training_predictions,testing_predictions)
    
def Random_forest(features,target,test_size_percent=0.2,cv_split=3):
    X_array = features.as_matrix()
    y_array = target.as_matrix()        
    model_rdf = RandomForestRegressor()
    X_train, X_test, y_train, y_test = train_test_split(X_array, y_array.T.squeeze(), test_size=test_size_percent, random_state=4)
    model_rdf.fit(X_train,y_train)
    test_prediction = model_rdf.predict(X_test)
    tscv = TimeSeriesSplit(cv_split)
    
    training_score = cross_val_score(model_rdf,X_train,y_train,cv=tscv.n_splits) 
    testing_score = cross_val_score(model_rdf,X_test,y_test,cv=tscv.n_splits)
    print"Cross-val Training score:", training_score.mean()
#    print"Cross-val Testing score:", testing_score.mean()
    training_predictions = cross_val_predict(model_rdf,X_train,y_train,cv=tscv.n_splits)
    testing_predictions = cross_val_predict(model_rdf,X_test,y_test,cv=tscv.n_splits)
    
    training_accuracy = metrics.r2_score(y_train,training_predictions) 
#    test_accuracy_model = metrics.r2_score(y_test,test_prediction_model)
    test_accuracy = metrics.r2_score(y_test,testing_predictions)
    
#    print"Cross-val predicted accuracy:", training_accuracy
    print"Test-predictions accuracy:",test_accuracy

    plot_model(target,y_train,y_test,training_predictions,testing_predictions)
    return model_rdf

def normalize_dataframe(df):
    ''' This function does a Min-Max (0-1) normalization on all the columns
    and return a normalized dataframe and a dataframe with column minium, maximum, mean,
    standard deviation ,variance'''
    result = df.copy()
    index = ['max','min','mean','std','var']
    normalization_para = pd.DataFrame(index=index)
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        mean_value = df[feature_name].mean()
        std_value = df[feature_name].std()
        var_value = df[feature_name].var()
        spec = [max_value,min_value,mean_value,std_value,var_value]
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        normalization_para[feature_name] = spec
    return result, normalization_para
def normalize(df):
    return (df-df.min())/(df.max()-df.min())
    
def mean_squared_error(df,df_norm_table,target_column):
    '''result dataframe is taken in, denormalized using the denormalisation parameters
    target column must be specified'''
    pred = df.filter(regex='Prediction')
    target = df[[target_column]]
    
    pred_deno = (pred*(df_norm_table[target_column]['max']-df_norm_table[target_column]['min']))+df_norm_table[target_column]['min']
    target_deno = (target*(df_norm_table[target_column]['max']-df_norm_table[target_column]['min']))+df_norm_table[target_column]['min']
    MSE = ((pred_deno.values-target_deno.values)**2).sum()/len(pred_deno)
    return MSE
def denormalize(df,df_norm_table,target_column):
    '''result dataframe is taken in, denormalized using the denormalisation parameters
    target column must be specified'''
    pred = df.filter(regex='Prediction')
    target = df[[target_column]]
    
    pred_deno = (pred*(df_norm_table[target_column]['max']-df_norm_table[target_column]['min']))+df_norm_table[target_column]['min']
    target_deno = (target*(df_norm_table[target_column]['max']-df_norm_table[target_column]['min']))+df_norm_table[target_column]['min']
    return pd.concat([pred_deno,target_deno],axis=1)
    
def join_df_weather(df):
    return df.join([weather,irradiance,humidity]).dropna()

def builds_dictionary(house_list_dir):
    buildings_list = []
    buildings = {}
    list_of_buildings = pd.read_csv(os.path.join(house_list_dir,'house_list_AC_area.csv'))
    for index,row in list_of_buildings.iterrows():
        buildings = {
                     "dataid":int(row['dataid']),
                    "start_period":row['date_enrolled'],
                    "end_period":row['date_withdrawn'],
                    "year_constructed":row['house_construction_year'],
                    "total_area":row['total_square_footage'],
                    "first_floor_area":row['first_floor_square_footage'],
                    "second_floor_area":row['second_floor_square_footage'],
                    "third_floor_area":row['third_floor_square_footage'],}
        buildings_list.append(buildings)
    return buildings_list

'''search for a string in between 2 string'''
def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
from math import isnan        
def sort_by_disagg_acc():
    Error_list = {}
    for columns in dataframe.columns:
        ac_number = find_between(columns,'air conditioner','_')
        if ac_number == '1':
            try:
                consum_col = columns
                consum_col_raw = columns.replace('1_','_')
                df_all=dataframe[[consum_col]].join(dataframe_raw[[consum_col_raw]])
                df_all = df_all.shift(-5)
                df_AC_on = df_all[df_all[consum_col_raw] >30.0][start:end]# optionally place ->[df_all[consum_col_raw] >10.0] in between df_all & [start:end]
                Error_list[columns.split('_',1)[1]]=metrics.r2_score(df_AC_on[consum_col_raw],df_AC_on[consum_col])
            except:
                continue
        else:
            continue
    clean_list = {k: Error_list[k] for k in Error_list if not isnan(Error_list[k])}
    return sorted(clean_list.items(), key=lambda x: x[1],reverse=True)
    
def plot_test_data(model,predicted,X,y,model_name=None):
    predictions=pd.Series(predicted.ravel(),index=y.index)
    predictions = predictions.rename(consum_col_pred)
    results=pd.concat([predictions,y],axis=1)
    if (model_name == 'mlp') or (model_name == 'MLP'):
        results.plot(title=model_name+'-Prediction_Vs_DisaggData - score {}'.format(model.score(X.values,y.values)))
    else:
        if model_name==None:
            model_name = 'Model'
        results.plot(title=model_name+'-Prediction_Vs_DisaggData - score {}'.format(model.score(X,y)))
    return results,predictions

def plot_scatter_predVsraw(predictions,df_raw,model_name=None):
    if predictions.max() <= 2: #indirectly finding if the frame is normalized or not
        predictions_df = pd.DataFrame(predictions).join(normalize(df_raw[[consum_col_raw]]).shift(-5)) #the raw data was not lagged for temperature effects, so i lag here
    else:
        predictions_df = pd.DataFrame(predictions).join(df_raw[[consum_col_raw]].shift(-5)) #the raw data was not lagged for temperature effects, so i lag here
    plt.scatter(predictions_df[consum_col_pred],predictions_df[consum_col_raw],s=150)
    if model_name == None:
        model_name = 'Model'
    plt.xlabel(consum_col_pred);plt.ylabel(consum_col_raw+' from raw data');plt.title(model_name+'-Prediction_Vs_RawData')
    print"\nR2 score: ",metrics.r2_score(predictions_df[consum_col_pred],predictions_df[consum_col_raw]),"\n"
    return predictions_df
    
def cross_val_pred_plot(model,X,y,consum_col,consum_col_pred,denorm_target,model_name=None,print_plot=False,cv=5):
    if 'multi' or 'mlp' or 'preceptron' in model_name.lower():
        warnings.filterwarnings("ignore", category=DeprecationWarning) #run this line separately
        whole_pred = cross_val_predict(model,X.values,y.values,cv=5)
    else:
        whole_pred = cross_val_predict(model,X,y,cv=5)
    whole_predictions=pd.Series(whole_pred.ravel(),index=y.index)
    whole_predictions = whole_predictions.rename(consum_col_pred)
    whole = pd.DataFrame(whole_predictions).join(y)
    whole[whole[consum_col_pred] <0.0] = 0
    r2 = metrics.r2_score(y,whole_pred)
    if print_plot:
        if ('multi' or 'mlp' or 'preceptron') in model_name.lower():
            whole.plot(title=model_name+'-Whole dataset predictions - score {}'.format(r2))
        else:
            if model_name==None:
                model_name = 'Model';print"\nInsert model name\n";
            whole.plot(title=model_name+'-Whole dataset predictions - score {}'.format(r2))
        plt.ylabel('Power consumption in Watts')
#        plt.xlabel('Date Time')
    #    print"\nR2 score: ",metrics.r2_score(y,whole_pred),"\n"
    
    if (model_name == 'svr') or (model_name == 'mlp'):
        denorm_whole = whole*(denorm_target.max().values[0]-denorm_target.min().values[0])+denorm_target.min().values[0]
        mae = metrics.mean_absolute_error(denorm_whole[consum_col],denorm_whole[consum_col_pred])
        mse = metrics.mean_squared_error(denorm_whole[consum_col],denorm_whole[consum_col_pred])
        whole = denorm_whole
#        if 'mlp' in model_name:
#            print'calculating metrics of MLP'
#            acc = model.score(X.values,y.values)
#        else:
#            print'calculating metrics of SVR'
#            acc = model.score(X,y)
    else:
        print'calculating metrics of LNR or RDF'
        mae = metrics.mean_absolute_error(y,whole_pred)
        mse = metrics.mean_squared_error(y,whole_pred)
#        acc = model.score(X,y)
    return whole,r2,mae,mse

def add_metrics_plot(x_loc,ax,acc,r2,mae,mse):
    y_loc = ax.get_ylim()[1]
    ax.text( x=x_loc,y=y_loc*0.96,s='Accuracy Score: {:.3f}'.format(acc),fontsize=20)
    ax.text( x=x_loc,y=y_loc*0.92,s='R2 Score: {:.3f}'.format(r2),fontsize=20)
    ax.text( x=x_loc,y=y_loc*0.88,s='MAE: {:.3f}'.format(mae),fontsize=20)
    ax.text( x=x_loc,y=y_loc*0.84,s='RMSE: {:.3f}'.format(np.sqrt(mse)),fontsize=20)
'''-----------------------------------------------------------------------------------------------'''
#'''#######-------loading temperature data--------------############'''
#weather = load_weather_data(weather_path,'/Austin_weather_2014.csv','temperature',0)
#
#'''#######-------loading irradiance data--------------############'''
## Irradiance data was loaded from building 8236 which has a PV generation unit, which indirectly gives
## information on the amount irradiance received on the building
#irradiance = load_irradiance_data(irradiance_path,'/8236_austin_all_appliance_2014.csv','gen',1)
#
#'''#######-------loading raw undisaggregated data--------------############'''
## H5 file downloaded from pecan dataport was extracted and written as csv files
#raw_8236 = load_raw_data(raw_csv_path)
#
#'''#######-------loading disaggregated data--------------############'''
## Disaggregation using FHMM by NILMTK was done and result was written as CSV file
#disagg_8236 = load_disaggregated_data('/FHMM_8236_2014march-november.csv')
#
#'''#######-------grouping minutely data into hourly--------------############'''
## i have made this 10hours shift so that it matches my database setting done in knime
## let me proceed with this setting and see if i can reproduce the same prediction accuracy
#disagg_8236_hourly = groupby_hourly(disagg_8236).shift(-5)
#raw_8236_hourly = groupby_hourly(raw_8236)
#
## Combining the disaggregated data , temperature and irradiance
#disagg_8236_hourly_temperature_irr = pd.concat([disagg_8236_hourly,weather,irradiance],axis=1,join='inner')
#
#'''ploting AC, temperature and IRR ro check correlation'''
##general_pandas_plot(disagg_8236_hourly_temperature_irr,'air conditioner1','temperature','gen')
#
#'''plotting correlation matrix'''
##plot_corr(disagg_8236_hourly_temperature_irr)
#
#disagg_8236_features = features_creation(disagg_8236_hourly_temperature_irr)
#
#
#"""-------code where i am trying to add occupancy based on (feature > feature_hourly_night_mean)""" 
##disagg_8236_features_with_occ = occupancy(disagg_8236_features)
##df=pd.DataFrame(occupancy(disagg_8236_features))
##df_trans = df.transpose()
##disagg_8236_features.join(df_trans,how='outer')
##
##disagg_8236_features_with_occ.fillna(value=None, method='backfill',inplace=True)
##
##columns_interested = disagg_8236.columns
##
##for column_name in columns_interested:
##    print(disagg_8236_features_with_occ[str(column_name)])
#
##disagg_8236_features_with_occ['AC_occ'] = [1 for disagg_8236_features_with_occ['air conditioner1']>disagg_8236_features_with_occ['air conditioner1_hourly_night_mean'] else 0]
#'''--------------------------------------------------------------------------------------------------'''
#
#''' _________ MACHINE LEARNING _____________ '''
## LAG NECESSARY COLUMNS
#df = disagg_8236_features.copy()
#df = lag_column(df,['air conditioner1'],24)
#df['air conditioner1_168'] = df['air conditioner1'].shift(168) # 168
#df = lag_column(df,['temperature'],5)
#df['gen_5'] = df['gen'].shift(5) # 168
#
##Specify the features and target columns
#df.dropna(inplace=True)
#X = df.drop(['air conditioner1','washing machine1','dish washer1','electric furnace1',
#             'microwave1'],axis=1)
##X = df[['temperature','temperature_1', 'temperature_2','temperature_3','temperature_4','temperature_5','gen_5']]
#y = df[['air conditioner1']]
#
#
##model = sm.OLS(y,X).fit()
##predictions = model.predict(X)
##model.summary()
#'''--------------------------------------------------------------------------------------------------'''
#
##Normalize the data
#X_norm, x_norm_table = normalize_dataframe(X)
#y_norm, y_norm_table = normalize_dataframe(y)
#
#
#    
#model_LR = linear_regression(X,y,test_size_percent=0.3,cv_split=5)
#model_SVR = svm_regressor(X,y,test_size_percent=0.3,cv_split=5) 
#model_RF = Random_forest(X,y,test_size_percent=0.3,cv_split=3)
#model_NN = neural_net(X,y,test_size_percent=0.3,cv_split=2,n_iter=50,learning_rate=0.1)

'''##############################################################################################################'''

params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'xx-large',
         'font.size': 17}
plt.rcParams.update(params)

house_list_dir = 'C:/Users/MANOJ/Dropbox/Manoj Thesis/NILMTK DataSets/Pecan Street (Wiki Energy)/notebook/base disaggregation codes'
hdf_dir = 'C:/Users/MANOJ/Dropbox/Manoj Thesis/NILMTK DataSets/Pecan Street (Wiki Energy)/Exported Data From interactive Data/All_houses_perfomance/'
weather_path = r'C:/Users/MANOJ/Dropbox/Manoj Thesis/NILMTK DataSets/Pecan Street (Wiki Energy)/Exported Data From interactive Data/Building 6636/knime/weather_austin'
irradiance_path = r'C:/Users/MANOJ/Dropbox/Manoj Thesis/NILMTK DataSets/Pecan Street (Wiki Energy)/Exported Data From interactive Data/Building 6636/knime/Energy consumption Austin/2014'

hdf_file_name = get_files_by_file_size(hdf_dir)

'''weather file'''
weather = pd.DataFrame(load_weather_data(weather_path,'/Austin_weather_2014.csv','temperature',0),columns=['temperature'])
weather = (weather - 32)*0.55555
humidity = pd.DataFrame(load_weather_data(weather_path,'/Austin_weather_2014.csv','humidity',0),columns=['humidity'])
irradiance = pd.DataFrame(load_irradiance_data(irradiance_path,'/8236_austin_all_appliance_2014.csv','gen',1))
irradiance[irradiance['gen'] <0.0] = 0
###################################################
combined = pd.HDFStore(hdf_file_name[0])
key = combined.keys()[0].replace('/','')

dataframe = combined.get(key)
'''always close the hdf file'''
combined.close()
###################################################
combined_raw = pd.HDFStore(hdf_file_name[1])
key = combined_raw.keys()[0].replace('/','')

dataframe_raw = combined_raw.get(key)
'''always close the hdf file'''
combined_raw.close()
###################################################
combined_mains = pd.HDFStore(hdf_file_name[2])
key = combined_mains.keys()[0].replace('/','')

dataframe_mains = combined_mains.get(key)
'''always close the hdf file'''
combined_mains.close()

# air conditioner 1 alone selected
df = dataframe[dataframe.filter(regex='conditioner1').columns]
df_raw = dataframe_raw.copy()
df_mains = dataframe_mains.copy().add_suffix('_mains')


start = '2014-05' #try for 3 months
end = '2014-10'




buildings_dict = builds_dictionary(house_list_dir)
building_numbers_acc = sort_by_disagg_acc()



#building = building_numbers_acc[0][0]
def lag_and_clean(df,consum_col,consum_col_raw):
    # LAG NECESSARY COLUMNS
    dframe = df.copy()
    dframe = lag_column(dframe,[consum_col],24)
    dframe[consum_col+'_168'] = dframe[consum_col].shift(168) # 168
    
    dframe = lag_column(dframe,['temperature'],5)
    dframe['gen_5'] = dframe['gen'].shift(5) # 168
    dframe['gen_6'] = dframe['gen'].shift(6) # 168
    
    #removing poorly disaggregated rows where disagg ac is greater than actual AC.
    dframe = dframe.drop(dframe[abs(dframe[consum_col] - dframe[consum_col_raw]) >30].index,axis=0)[start:end]
    #selecting disaggregated data alone (removing raw data)
    dframe = dframe.drop([consum_col_raw],axis=1)
    #Specify the features and target columns
    dframe.dropna(inplace=True)
    return dframe
    
def get_train_test_data(consum_col,consum_col_raw,model_name):
    
    df_mod = df[[consum_col]].join(df_raw[[consum_col_raw]]).shift(-5)
    df_mod_amb = df_mod.join([weather,irradiance]).dropna()
    df_mod_features = features_creation(df_mod_amb.copy())
    
    dframe = lag_and_clean(df_mod_features,consum_col,consum_col_raw)
    features = dframe.drop([consum_col,'sin_hour','cos_hour','day_of_week','month','week_of_year'],axis=1)
    target = dframe[[consum_col]]
    
    if not (('svr' in model_name)or('mlp' in model_name)):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=41234)
        return X_train, X_test, y_train, y_test, features, target, None
    else:
        X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(normalize(features), normalize(target), test_size=0.2, random_state=41234)
        return X_train_norm, X_test_norm, y_train_norm, y_test_norm, normalize(features),normalize(target),target

saved_model = {'lnr':"C:/Users/MANOJ/Desktop/ML Results + Model/Linear_model.sav",
               'svr':"C:/Users/MANOJ/Desktop/ML Results + Model/SVR_model.sav",
               'rdf':"C:/Users/MANOJ/Desktop/ML Results + Model/RDF_model.sav",
               'mlp':"C:/Users/MANOJ/Desktop/ML Results + Model/MLP_model.sav"}
               
def best_model(model_name):
    rdf_params = {'max_features':[5,10,15,20],'n_estimators':[10,15,20]}
#    layer_opt = np.random.randint(low=3,high=15,size=3)
#    mlp_params = {'n_iter':[80],'learning_rate':[0.02],
#                  'hidden0__type':['Rectifier'],
#                  'hidden0__units':layer_opt,
#                  'hidden1__type':['Rectifier'],
#                  'hidden1__units':layer_opt}
    if 'rdf' in model_name:
#        best_model = GridSearchCV(RandomForestRegressor(), rdf_params).fit(features, target)
#        regressor = best_model.best_estimator_
#        return RandomForestRegressor(max_features=regressor.max_features,n_estimators=regressor.n_estimators)
        return RandomForestRegressor()
    elif 'mlp' in model_name:
        mlp = Regressor(layers=[Layer("Rectifier",units=6),Layer("Rectifier",units=4),Layer("Linear")])
#        warnings.filterwarnings("ignore", category=DeprecationWarning) #run this line separately
#        best_model = GridSearchCV(mlp, mlp_params).fit(features, target)
#        regressor = best_model.best_estimator_
        return mlp
#        return Regressor(layers=[Layer(best_mlp_reg.layers[0].type,
#                            units=best_mlp_reg.layers[0].units),# Hidden Layer1
#                            Layer(best_mlp_reg.layers[1].type,
#                            units=best_mlp_reg.layers[1].units), # Hidden Layer2
#                        Layer(best_mlp_reg.layers[-1].type)],
#                        n_iter=best_mlp_reg.n_iter,
#                        learning_rate=best_mlp_reg.learning_rate,
#                        learning_rule=best_mlp_reg.learning_rule)     # Output Layer
        
    
def load_models(model_name,load_from_file):
    if model_name == 'lnr':
        return LinearRegression()
    elif model_name == 'svr':
        return SVR(kernel='rbf')
    elif model_name == 'rdf':
        return RandomForestRegressor()
    elif model_name == 'mlp':
        return best_model(model_name)


def machine_model(building_name=None,load_from_file=None):
    building_error_list = []
    building = building_name
    consum_col = 'air conditioner1_'+building
    consum_col_raw = 'air conditioner_'+building
    consum_col_mains = 'air conditioner_'+building+'_mains'
    consum_col_pred = 'Predicted_'+consum_col
    
    
    learning_models={'mlp':'Multi-Layer Perceptron',
                     'svr':'Support Vector Regression',
                     'lnr':'Linear Regression',
                     'rdf':'Random Forest Regression'}    

    for model_name in ['lnr','svr','rdf','mlp']:
        global features,target,denorm_target
        X_train, X_test, y_train, y_test,features,target,denorm_target = get_train_test_data(consum_col,consum_col_raw,model_name)
        building_error_dict = {}
        print building,":",model_name
        
        model = load_models(model_name,load_from_file)
#        if not load_from_file:
#            if 'mlp' in model_name:
#                model.fit(X_train.values,y_train.values)
#            else:
#                model.fit(X_train,y_train)
        whole,r2,mae,mse = cross_val_pred_plot(model,features,target,consum_col,consum_col_pred,denorm_target,model_name=model_name,print_plot=False)
        building_error_dict['Learning model'] = learning_models.get(model_name)
        building_error_dict['Building number'] = int(building)
        building_error_dict['R2 Score'] = round(r2,3)
        building_error_dict['Mean Absolute Error'] = round(mae,3)
        building_error_dict['Root Mean Squared Error'] = round(np.sqrt(mse),3)
        building_error_list.append(building_error_dict)
        del X_train, X_test, y_train, y_test,features,target,denorm_target
    return {int(building):building_error_list}
    
metrics_frame = pd.DataFrame()

building_name = '5949'
a=machine_model(building_name=building_name,load_from_file=False)
b = pd.DataFrame.from_dict(a.get(int(building_name))).set_index('Building number')
metrics_frame=pd.concat([metrics_frame,b])

metrics_frame.to_hdf('C:/Users/MANOJ/Desktop/ML Results + Model/16september2017/metricsframe_5949','frame')
#
#
#
#get_frame1 = pd.HDFStore('C:/Users/MANOJ/Desktop/ML Results + Model/metricsframe_part1.h5')
#get_frame2 = pd.HDFStore('C:/Users/MANOJ/Desktop/ML Results + Model/metricsframe_part2.h5')
#get_frame3 = pd.HDFStore('C:/Users/MANOJ/Desktop/ML Results + Model/metricsframe_part3.h5')
#key = get_frame1.keys()[0].replace('/','')
#metrics_frame1 = get_frame1.get(key)
#metrics_frame2 = get_frame2.get(key)
#metrics_frame3 = get_frame3.get(key)
#get_frame1.close()
#get_frame2.close()
#get_frame2.close()
##
#metrics_frame = pd.concat([metrics_frame1,metrics_frame2,metrics_frame3])
metrics_frame=metrics_frame.drop(metrics_frame[metrics_frame['R2 Score'] < 0.3].index,axis=0)
metrics_frame['Building number'] = metrics_frame.index
metrics_frame = metrics_frame.set_index(['Building number','Learning model'])
metrics_frame.index=metrics_frame.index.set_names('Regression Models',level=1)
##############################################################################
import matplotlib.font_manager as fm
prop = fm.FontProperties(fname='C:/Users/MANOJ/Dropbox/Manoj Thesis/Report/for journal publication/Version 1-BN- 31-08-2017/pala.ttf')
plt.rc('font', family=prop.get_name())
##############################################################################
fig = plt.figure()
#plt.rc('font',**{'family':'serif','serif':['Palatino']}) 
ax1 = plt.subplot(111)
ax1.set_ylabel('Accuracy Score',fontsize=60,labelpad=30)
ax1.set_xlabel('Regression Model',fontsize=60,labelpad=30)
ax1.patch.set_facecolor('white')
ax1.yaxis.grid(which="major", color='k', linestyle='-.', linewidth=0.7)

metrics_frame[['Accuracy score']].unstack(level=0).xs('Accuracy score', axis=1, drop_level=True).plot(kind='bar',ax=ax1)
labels = [item.get_text() for item in ax1.get_xticklabels()]
labels = [c.replace(' R','\nR').replace(' P','\nP') for c in labels]

ax1.set_xticklabels(labels)
ax1_frame=ax1.legend(frameon=1,loc='upper left',title='Building',prop={'size': 50})
ax1_frame.get_title().set_fontsize('50')
frame = ax1_frame.get_frame().set_alpha(0.5)
#frame.set_edgecolor('black')
ax1.xaxis.label.set_visible(False)
plt.setp(ax1.xaxis.get_ticklabels(), rotation=0 ,fontsize=60)
plt.setp(ax1.yaxis.get_ticklabels(), rotation=0 ,fontsize=60)
##############################################################################
fig = plt.figure()
#plt.rc('font',**{'family':'serif','serif':['Palatino']}) 
ax1 = plt.subplot(111)
ax1.set_ylabel('R2 Score',fontsize=60,labelpad=30)
ax1.set_xlabel('Regression Model',fontsize=60,labelpad=30)
ax1.patch.set_facecolor('white')
ax1.yaxis.grid(which="major", color='k', linestyle='-.', linewidth=0.7)

metrics_frame[['R2 Score']].unstack(level=0).xs('R2 Score', axis=1, drop_level=True).plot(kind='bar',ax=ax1)
labels = [item.get_text() for item in ax1.get_xticklabels()]
labels = [c.replace(' R','\nR').replace(' P','\nP') for c in labels]

ax1.set_xticklabels(labels)
ax1_frame=ax1.legend(frameon=1,loc='upper left',title='Building',prop={'size': 50})
ax1_frame.get_title().set_fontsize('50')
frame = ax1_frame.get_frame().set_alpha(0.5)
#frame.set_edgecolor('black')
ax1.xaxis.label.set_visible(False)
plt.setp(ax1.xaxis.get_ticklabels(), rotation=0 ,fontsize=60)
plt.setp(ax1.yaxis.get_ticklabels(), rotation=0 ,fontsize=60)
##############################################################################
fig = plt.figure()
plt.rc('font',**{'family':'serif','serif':['Palatino']}) 
ax1 = plt.subplot(111)
ax1.set_ylabel('Mean Absolute Error',fontsize=60,labelpad=30)
ax1.set_xlabel('Regression Model',fontsize=60,labelpad=30)
ax1.patch.set_facecolor('white')
ax1.yaxis.grid(which="major", color='k', linestyle='-.', linewidth=0.7)

metrics_frame[['Mean Absolute Error']].unstack(level=0).xs('Mean Absolute Error', axis=1, drop_level=True).plot(kind='bar',ax=ax1)
labels = [item.get_text() for item in ax1.get_xticklabels()]
labels = [c.replace(' R','\nR').replace(' P','\nP') for c in labels]
ax1.set_xticklabels(labels)
ax1_frame=ax1.legend(frameon=1,loc='upper left',title='Building',prop={'size': 50})
ax1_frame.get_title().set_fontsize('50')
frame = ax1_frame.get_frame().set_alpha(0.6)
#frame.set_edgecolor('black')
ax1.xaxis.label.set_visible(False)
plt.setp(ax1.xaxis.get_ticklabels(), rotation=0 ,fontsize=60)
plt.setp(ax1.yaxis.get_ticklabels(), rotation=0 ,fontsize=60)
##############################################################################
fig = plt.figure()
#plt.rc('font',**{'family':'serif','serif':['Palatino']}) 
ax1 = plt.subplot(111)
ax1.set_ylabel('Root Mean Squared Error',fontsize=60,labelpad=30)
ax1.set_xlabel('Regression Model',fontsize=60,labelpad=30)
ax1.patch.set_facecolor('white')
ax1.yaxis.grid(which="major", color='k', linestyle='-.', linewidth=0.7)

metrics_frame[['Root Mean Squared Error']].unstack(level=0).xs('Root Mean Squared Error', axis=1, drop_level=True).plot(kind='bar',ax=ax1)
labels = [item.get_text() for item in ax1.get_xticklabels()]
labels = [c.replace(' R','\nR').replace(' P','\nP') for c in labels]
ax1.set_xticklabels(labels)

ax1_frame=ax1.legend(frameon=1,loc='upper left',title='Building',prop={'size': 50})
ax1_frame.get_title().set_fontsize('50')
frame = ax1_frame.get_frame().set_alpha(0.6)
#frame.set_edgecolor('black')
ax1.xaxis.label.set_visible(False)
plt.setp(ax1.xaxis.get_ticklabels(), rotation=0 ,fontsize=60)
plt.setp(ax1.yaxis.get_ticklabels(), rotation=0 ,fontsize=60)
##############################################################################
fig = plt.figure()
plt.rc('font',**{'family':'serif','serif':['Palatino']}) 
ax1 = plt.subplot(111)
ax1.set_ylabel('R2 Score',fontsize=60,labelpad=30)
ax1.set_xlabel('Regression Model',fontsize=60,labelpad=30)
ax1.patch.set_facecolor('white')
ax1.yaxis.grid(which="major", color='k', linestyle='-.', linewidth=0.7)
ax1.set_ylim([0,1])
metrics_frame[['R2 Score']].unstack(level=0).xs('R2 Score', axis=1, drop_level=True).plot(kind='bar',ax=ax1)
labels = [item.get_text() for item in ax1.get_xticklabels()]
labels = [c.replace(' R','\nR').replace(' P','\nP') for c in labels]
ax1.set_xticklabels(labels)

ax1_frame=ax1.legend(frameon=1,loc='upper left',title='Building',prop={'size': 50})
ax1_frame.get_title().set_fontsize('50')
frame = ax1_frame.get_frame().set_alpha(0.6)
#frame.set_edgecolor('black')
ax1.xaxis.label.set_visible(False)
plt.setp(ax1.xaxis.get_ticklabels(), rotation=0 ,fontsize=60)
plt.setp(ax1.yaxis.get_ticklabels(), rotation=0 ,fontsize=60)

