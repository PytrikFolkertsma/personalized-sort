import pandas as pd

def combine_clicked_booked(df):
	#combines click_bool and booking_bool
	#changes click_bool to 2 if booking_bool=1, then deletes booking_bool
	df.ix[df.booking_bool == 1, 'click_bool'] = 2
	df = df.drop(['booking_bool'], axis=1)
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
	"click_bool",
	"booking_bool"]
	data = df[features]
	data.fillna(0, inplace=True)
	return data

def main():
	df = pd.read_csv('../../Data Mining VU data/xaa_100.000.csv') 
	df = extract_features(df)
	# print df.head()
	df = combine_clicked_booked(df)
	df.to_csv('../../Data Mining VU data/prepared_data.txt', sep='\t', index=False)

if __name__ == '__main__':
	main()