import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble


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

	#calculate parameters
	clf = ensemble.GradientBoostingClassifier(learning_rate=0.15, max_depth=4, n_estimators=200)
	
	clf.fit(train_X[features], train_y['click_bool'])
	predictions_click = clf.predict(test_X[features])
	probabilities_click = clf.predict_proba(test_X[features])
	
	clf.fit(train_X[features], train_y['booking_bool'])
	predictions_book = clf.predict(test_X[features])
	probabilities_book = clf.predict_proba(test_X[features])

	output = open('../../Data Mining VU data/predictions.txt', 'w')
	output.write('srch_id\tprop_id\tprob_click\tprob_book\tpred_click\tpred_book\tscore\n')
	
	srch_id = test_X['srch_id'].tolist()
	prop_id = test_X['prop_id'].tolist()
	click_bool = test_y[ 'click_bool'].tolist()
	booking_bool = test_y['booking_bool'].tolist()
	score = test_X['score'].tolist()

	for i in range(len(predictions_click)):
		output.write('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(str(srch_id[i]), str(prop_id[i]), str(probabilities_click[i][1]), str(probabilities_book[i][1]), str(predictions_click[i]), str(predictions_book[i]), str(score[i])) + '\n')

	# [output.write( str(probabilities_click[i][1]) + '\t' + str(probabilities_book[i][1]) + '\t' + str(test_y['click_bool'][i]) + str(test_y['booking_bool'][i]) + '\n') for i in range(len(predictions_click))]
	output.close()

if __name__ == '__main__':
	main()