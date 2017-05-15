import pandas as pd
from sklearn import preprocessing


def add_score(df):
	df['score'] = 0
	df.ix[df.click_bool == 1, 'score'] = 1
	df.ix[df.booking_bool == 1, 'score'] = 5
	return df 

def scale(df):
	features = [
	# "visitor_hist_starrating",
	"prop_review_score",
	"price_usd",
	"srch_room_count",
	# "srch_query_affinity_score",
	'promotion_flag',
	'srch_length_of_stay',
	'srch_booking_window',
	'srch_adults_count', 
	'srch_children_count',
    # 'loc_ratio2'
    ]
	df[features].astype(float).apply(preprocessing.scale, axis=0, raw=True)
	return df

def extract_features(df):
	#extract 
	features = [
	"srch_id",
	"prop_id",
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
	"click_bool",
	"booking_bool"]
	data = df[features]
	data.fillna(0, inplace=True)
	return data

def main():
	df = pd.read_csv('../../Data Mining VU data/xaa_100.000.csv') 
	df = extract_features(df)
	df = add_score(df)
	# df = combine_clicked_booked(df)
	# df = scale(df)
	df.to_csv('../../Data Mining VU data/prepared_data.txt', sep='\t', index=False)

if __name__ == '__main__':
	main()