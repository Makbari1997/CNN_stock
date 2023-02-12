import pytse_client as tse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def get_stock_history(stock_name):
    '''Gets the name of the stock and downloads its history from TSE
    Parameters:
    ------------
    stock_name: str
    ------------
    Return: Pandas DataFrame    
    '''
    tse.download(symbols=stock_name, write_to_csv='True')
    ticker = tse.Ticker(stock_name)
    return ticker.history

def plot_figure(prediction, real, address):
    '''gets the data and saves figure to the given address
    Parameters:
    -------------
    prediction, real : 1-d array
    address : str
    '''
    plt.clf()
    plt.plot(np.arange(len(prediction)), prediction, color = 'blue', label='predicted values')
    plt.plot(np.arange(len(prediction)), real, color='red', label='real values')
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.legend()
    plt.savefig(address)

def predict_stock(data):
    '''Predicts flow of the stock in the next day based on the given standard pytse_client history dataframe 
    Parameters: 
    ------------
    data: pandas DataFrame
    '''
    '''gets data and fits a cnn to them
    Parameters: 
    -------------
    data : pandas.DataFrame
    -------------
    Return : tensorflow.keras.Sequential
    '''
    #DATA PREPROCESSING
    #extracting week_day feature
    data['date'] = pd.to_datetime(data['date'])
    week_day = []
    for date in data.iloc[:, 0]:
        week_day.append((date.weekday() + 2) % 7 + 1)
    #creating data_set with new features
    data_set = pd.DataFrame({
        'open' : data.iloc[:, 1].values,
        'high' : data.iloc[:, 2].values,
        'low' : data.iloc[:, 3].values,
        'close' : data.iloc[:, 8].values,
        'adj_close' : data.iloc[:, 4].values,
        'daily_growth' : data.iloc[:, 8].values - data.iloc[:, 1].values,
        'month' : pd.to_datetime(data['date']).dt.month,
        'month_day' : pd.to_datetime(data['date']).dt.day,
        'week_day' : week_day,
        'value' : data.iloc[:, 5],
        'volune' : data.iloc[:, 6],
        'count' : data.iloc[:, 7]
    })

    #normalizing data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data_set)

    train_len = int(0.8*len(scaled_data))

    #creating training set
    X_train = []
    Y_train = []
    for i in range(10, train_len):
        X_train.append(scaled_data[i-10:i, :])
        Y_train.append(scaled_data[i, 4])
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    
    #CREATING CNN MODEL
    model = tf.keras.Sequential()
    #adding firs convolutional layer
    model.add(tf.keras.layers.Conv2D(
        filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'
    ))
    model.add(tf.keras.layers.AvgPool2D(pool_size=(2,2)))
    #adding second convolutional layer
    model.add(tf.keras.layers.Conv2D(
        filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'
    ))
    model.add(tf.keras.layers.AvgPool2D(pool_size=(2,2)))
    #adding third convolutional layer
    model.add(tf.keras.layers.Conv2D(
        filters=16, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'
    ))
    model.add(tf.keras.layers.AvgPool2D(pool_size=(2,2)))
    #flatting dinput to feed the Dense layer
    model.add(tf.keras.layers.Flatten())
    #addind Dense layers each followed by a dropout layer to prevent overfitting
    model.add(tf.keras.layers.Dense(units=100, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(units=1, activation='relu'))
    #compling model
    model.build(input_shape=X_train.shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')
    #fitting model respect to given data
    model.fit(x=X_train, y=Y_train, epochs=100, batch_size=16)

    #TESTING MODEL
    test_data = scaled_data[train_len - 10:]
    X_test, Y_test = [], []
    for i in range(10, len(test_data)):
        X_test.append(test_data[i-10:i, :])
        Y_test.append(test_data[i, 4])
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

    prediction = model.predict(X_test)
    #drawing performance on test data
    plot_figure(prediction, Y_test, 'test.png')
    #drawing performance on training data
    plot_figure(model.predict(X_train), Y_train, 'train.png')

    #predicting flow of the stock on the next day based on last 10 days(2 weeks)
    input_data = np.array([scaled_data[-10:, :]])
    input_data = np.reshape(input_data, (input_data.shape[0], input_data.shape[1], input_data.shape[2], 1))
    predicted_price = model.predict(input_data)

    today_price = scaled_data[-1, 4]
    growth = predicted_price - today_price
    print('****************************Result****************************')
    if growth < 0:
        print('The closing price of stock may decrease on the next day!')
    if growth > 0:
        print('The closing price of stock may increase on the next day!')
    if growth == 0:
        print('The closing price of stock may not change on the next day!')
    print('**************************************************************')

if __name__ == '__main__':
    stock_name = input('Enter name of the stock in persian: ')

    data = get_stock_history(stock_name)

    predict_stock(data)