import pandas as pd

def extract_features(df):
	#extract 
	#visitor_hist_starrating
	#visitor_hist_adr_usd
	#prop_starrating
	#prop_review_score
	#price_usd
	#srch_room_count
	#srch_query_affinity_score
	features = ["visitor_hist_starrating",
	"visitor_hist_adr_usd",
	"prop_starrating",
	"prop_review_score",
	"price_usd",
	"srch_room_count",
	"srch_query_affinity_score",
	"click_bool",
	"booking_bool"]
	data = df[features]
	return data

def main():
	df = pd.read_csv('../../Data Mining VU data/xaa_100.000.csv') 
	df = extract_features(df)
	# print df.head(100)
	df.to_csv('../../Data Mining VU data/prepared_data.txt', sep='\t')

if __name__ == '__main__':
	main()