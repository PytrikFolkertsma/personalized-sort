import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
	train = pd.read_table('../../Data Mining VU data/train_small.txt') 
	X = train[['price_usd']]
	y = train['click_bool'].tolist()
	train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.5, random_state=23)
	logistic = LogisticRegression()
	logistic.fit(X,y)
	predictions = logistic.predict(test_X)
	probabilities = logistic.predict_proba(test_X)
	output = open('../../Data Mining VU data/predictions.txt', 'w')
	output.write('probability\tprediction\tgroundtruth\n')
	# [i for i in range(len(predictions))]
	[output.write(str(probabilities[i][1]) + '\t' + str(predictions[i]) + '\t' + str(test_y[i]) + '\n') for i in range(len(predictions))]
	output.close()

if __name__ == '__main__':
	main()