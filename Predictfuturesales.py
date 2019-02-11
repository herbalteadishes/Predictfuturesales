# -*- coding: utf-8 -*-
# Basic packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import model_selection
import xgboost as xgb



pd.set_option('display.max_columns',None)

train = pd.read_csv('sales_train_v2.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')
items = pd.read_csv('items.csv')
item_cats = pd.read_csv('item_categories.csv')
shops = pd.read_csv('shops.csv')

#print(train.head(5))
#print(test.head(5))
#print(submission.head(5))

def Unreasonable_data(data):
        print("Min Value:",data.min())
        print("Max Value:",data.max())
        #print("Average Value:",data.mean())
        #print("Center Point of Data:",data.median())

def eda(data):    
    print("-----------Information-----------")
    print(data.info())
    print("-----------Data Types-----------")
    print(data.dtypes)
    print("----------Missing value-----------")
    print(data.isnull().sum())
    print("----------Shape of Data----------")
    print(data.shape)
    print("---------Unreasonable data-------")
    Unreasonable_data(data)
    
    
def graph_insight(data):
    print(set(data.dtypes.tolist()))
    df_num = data.select_dtypes(include = ['float64', 'int64'])
    df_num.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8);

#eda(train)
#graph_insight(train)

#print('before train shape:', train.shape)
train = train[(train.item_price > 0) & (train.item_cnt_day > 0)]
#print('after train shape:', train.shape)
#print(train.head(10))
train = pd.merge(train,test.drop('ID',axis = 1),on = ['shop_id','item_id'],how = 'right')

#print(train.isnull().sum())#发现test集中部分shopid+itemid组合，不在train中
#
train['date'].fillna('01.01.2013',inplace=True)
train['date_block_num'].fillna(0,inplace=True)
train['item_price'].fillna(0,inplace=True)
train['item_cnt_day'].fillna(0,inplace=True)
#print(train.isnull().sum())

train['month'] = train.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%m'))
train['year'] = train.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%Y'))
train = train.drop('date', axis=1)

#train中日销量变月销量；日价格变月价格（一个月每天的均值）
train_sum_item_cnt_day = train.groupby(['shop_id','item_id','date_block_num','month','year']).sum().sort_index().reset_index()
#print(train_sum_item_cnt_day.head(10))
train_avg_item_price = train.groupby(['shop_id','item_id','date_block_num','month','year']).mean().sort_index().reset_index()
#print(train_avg_item_price.head(10))
df_train = []
df_train = pd.DataFrame(train_avg_item_price,columns=['shop_id', 'item_id', 'date_block_num','item_price','month','year'])
df_train = pd.concat([df_train,train_sum_item_cnt_day['item_cnt_day']],axis=1)
df_train = df_train.rename(columns={'item_cnt_day':'item_cnt_month'})

#print(df_train.head(10))

sales_date_block_num = df_train.groupby(['date_block_num']).sum()
sales_date_block_num = sales_date_block_num.reset_index()
sales_date_block_num_date_block_num = sales_date_block_num['date_block_num']
sales_date_block_num_item_cnt_month = sales_date_block_num['item_cnt_month']


plt.title('The correlation between date_block_num and item_cnt_month')
plt.xlabel('date_block_num')
plt.ylabel('item_cnt_month')
x = range(0,34,1)
plt.xticks(x)
plt.bar(sales_date_block_num_date_block_num,sales_date_block_num_item_cnt_month)


df_train = pd.merge(df_train,items.drop('item_name',axis = 1),on=['item_id'],how='left')#加入item_category_id列
df_train = pd.DataFrame(df_train,columns=['shop_id', 'item_id', 'date_block_num','month','year','item_category_id','item_price','item_cnt_month'])
#print(train.head(5))
test['date_block_num'] = 34
test['month'] = 11
test['year'] = 2015
test = pd.merge(test,items.drop('item_name',axis = 1),on=['item_id'],how='left')#加入item_category_id列
df_test = test.drop(['ID'],axis = 1)
'''
#test价格：方案1：取以前价格均值
df_test = df_test.set_index(['shop_id', 'item_id'])
df_test_price = df_train.groupby(['shop_id','item_id']).mean().sort_index().reset_index()

df_test['item_price'] = 0
for index, i in enumerate(df_test_price.values):
    (shop_id, item_id) = (i[0], i[1])
    df_test.loc[(shop_id, item_id)]['item_price'] = i[4]
df_test = df_test.reset_index()     
'''
def read_data(train_data, test_data, split_proportion):  
    train= train_data
    test_x=test_data.values
    train_x = train[train.columns[0:7]].values #修改需要输入的数据列
    train_y = train[train.columns[-1]].values #修改标签/需要预测的数据列
    train_x,val_x, train_y, val_y = model_selection.train_test_split(train_x,train_y,test_size=split_proportion , random_state = 0, stratify = None, shuffle=True )
    return train_x, train_y, val_x, val_y, test_x

train_x, train_y, val_x, val_y, test_x = read_data(df_train,df_test,0.2)
#print(test_x[0:10])

#test价格：方案2：xgboost预测
def read_data_price(train_data, split_proportion):  
    train= train_data
    train_x_price = train[train.columns[0:6]].values #修改需要输入的数据列
    train_y_price = train[train.columns[-2]].values #修改标签/需要预测的数据列
    train_x_price, val_x_price, train_y_price, val_y_price = model_selection.train_test_split(train_x_price,train_y_price,test_size=split_proportion , random_state = 0, stratify = None, shuffle=True )
    return train_x_price, train_y_price, val_x_price, val_y_price
train_x_price, train_y_price, val_x_price, val_y_price = read_data_price(df_train,0)
price_model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='reg:linear')
price_model.fit(train_x_price, train_y_price)

df_test_price = price_model.predict(test_x)
test_x = np.column_stack((test_x,df_test_price))
print(test_x[0:10,:])


model = xgb.XGBRegressor(max_depth=8, learning_rate=0.1, n_estimators=1000,min_child_weight=300,colsample_bytree=0.8,subsample=0.8,eta=0.3,seed=42,silent=0, objective='reg:linear')
model.fit(train_x, train_y)
results = model.predict(val_x)

sum_mean=0
for i in range(len(results)):
    sum_mean+=(results[i]-val_y[i])**2
sum_erro=np.sqrt(sum_mean/len(results))
print ("RMSE:",sum_erro)

results = model.predict(test_x)



'''
#被放弃的做了一半的LSTM方案，有待进一步研究
#放弃原因：我选的特征太多，导致缺失数据太多。而LSTM要求每个时间都要输入所有特征对应的数据。若强行使用LSTM，须人工填充数据，会大大降低准确率。我减少特征，又会大大减小输入信息量，不能保证准确率。
#扩展每一项至33个月
num_month = 33

i=0
sample = df_train.loc[df_train['shop_id'] == df_test['shop_id'][i]]
sample = sample.loc[df_train['item_id'] == df_test['item_id'][i]]
month_list=[m for m in range(num_month+1)]
shop = []
for j in range(num_month+1):
    shop.append(df_test['shop_id'][i])
item = []
for k in range(num_month+1):
    item.append(df_test['item_id'][i])
months_full = pd.DataFrame({'shop_id':shop, 'item_id':item,'date_block_num':month_list})
sales_33month = pd.merge(sample, months_full, how='right', on=['shop_id','item_id','date_block_num'])
sales_33month = sales_33month.sort_values(by=['date_block_num'])
sales_33month.fillna(0.00,inplace=True)
for n in range(1,6):
    sales_33month["T_" + str(n)] = sales_33month.item_cnt_month.shift(n)
    sales_33month.fillna(0.0, inplace=True)
print(sales_33month)
'''
'''
for i in range(len(test)):
    sample = df_train.loc[df_train['shop_id'] == df_test['shop_id'][i]]
    sample = sample.loc[df_train['item_id'] == df_test['item_id'][i]]
    month_list=[m for m in range(num_month+1)]
    shop = []
    for j in range(num_month+1):
        shop.append(df_test['shop_id'][i])
    item = []
    for k in range(num_month+1):
        item.append(df_test['item_id'][i])
    months_full = pd.DataFrame({'shop_id':shop, 'item_id':item,'date_block_num':month_list})
    sales_33month = pd.merge(sample, months_full, how='right', on=['shop_id','item_id','date_block_num'])
    sales_33month = sales_33month.sort_values(by=['date_block_num'])
    sales_33month.fillna(0.00,inplace=True)
    for n in range(1,6):
        df_train["T_" + str(n)] = df_train.item_cnt_month.shift(n)
        df_train.fillna(0.0, inplace=True)
''' 
        
test_previous = test
test = test.set_index(['shop_id', 'item_id'])
test['item_cnt_month'] = 0
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
for index, j in enumerate(test_previous.values):
    (shop_id, item_id) = (j[1], j[2])
    test.loc[(shop_id, item_id)]['item_cnt_month'] = results[index]
    
test = test.reset_index().drop(['shop_id', 'item_id'], axis=1)
test.to_csv('submission_xgboost.csv', index=False)

