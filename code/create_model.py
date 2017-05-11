import pandas as pd
from sklearn.linear_model import LogisticRegression

def fit_model(X, y):
	logistic = LogisticRegression()
	# print(len(X)), len(y)
	logistic.fit(X,y)

def main():
	train = pd.read_table('../../Data Mining VU data/train_small.txt') 
	train_X = train['price_usd']
	train_y = train['click_bool']
	fit_model(train_X, train_y)
	val = pd.read_table('../../Data Mining VU data/val_small.txt') 
	# print train.head()

if __name__ == '__main__':
	main()