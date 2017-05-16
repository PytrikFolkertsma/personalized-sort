import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble


def sort_predictions_per_srch_id(df):
	#sorts the predictions for each srch_id
	# return df.groupby('srch_id').apply()
	return df.sort(['prediction'], axis=0, ascending=False)

def make_predictions(train_X, train_y, test_X):
	###########calculate parameters
#########
	clf = ensemble.GradientBoostingClassifier(learning_rate=0.15, max_depth=4, n_estimators=200)
	
	#train on click and make predictions
	clf.fit(train_X, train_y['click_bool'])
	predictions_click = clf.predict(test_X)
	probabilities_click = clf.predict_proba(test_X)
	
	#train on book and make predictions
	clf.fit(train_X, train_y['booking_bool'])
	predictions_book = clf.predict(test_X)
	probabilities_book = clf.predict_proba(test_X)

	#combine predictions
	predictions = pd.DataFrame({'pred_click': predictions_click, 'pred_book': predictions_book})
	predictions['predicted_class'] = 0
	predictions.ix[predictions.pred_click == 1, 'predicted_class'] = 1
	predictions.ix[predictions.pred_book == 1, 'predicted_class'] = 2
 	# print predictions.head()
 	return predictions['predicted_class'].tolist()

def main():
	data = pd.read_table('../../Data Mining VU data/prepared_data.txt') 
	X = data
	y = data[['click_bool', 'booking_bool']]

	train_X = X[0:50000]
	test_X = X[50000:100000]
	train_y = y[0:50000]
	test_y = y[50000:100000]

	# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.5)

	features = [
		# "srch_id",
		# "prop_id",
		# "visitor_hist_starrating",
		"visitor_hist_adr_usd",
		"prop_starrating",
		"prop_review_score",
		"price_usd",
		"srch_room_count",
		# "srch_query_affinity_score",
		'promotion_flag',
		'srch_length_of_stay',
		'srch_booking_window',
		'srch_adults_count', 
		'srch_children_count',
		'prop_location_score1',
		'prop_location_score2',
	    # 'loc_ratio2',
		# "click_bool",
		# "booking_bool"
	]

	
	predictions = make_predictions(train_X[features], train_y, test_X[features])
	final_predictions = pd.DataFrame({'srch_id': test_X['srch_id'], 'prop_id': test_X['prop_id'], 'score': test_X['score'], 'prediction': predictions})
	final_predictions = final_predictions[['srch_id', 'prop_id', 'score', 'prediction']]
	
	sorted_predictions = final_predictions.groupby('srch_id').apply(sort_predictions_per_srch_id)
	sorted_predictions.to_csv('../../Data Mining VU data/predictions.txt', sep='\t', index=False)


if __name__ == '__main__':
	main()