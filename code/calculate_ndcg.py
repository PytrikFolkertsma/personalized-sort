import pandas as pd 
import math


def calculate_ndcg(df):
	dcg = sum([(2**float(row['score'])-1)/ math.log(float(e+2),2) for e, (i, row) in enumerate(df.iterrows()) if row['score'] != 0])
	ordered = df.sort_values(by='score', axis=0, ascending=False)
	total = sum([(2**float(row['score'])-1)/ math.log(float(e+2),2) for e, (i, row) in enumerate(ordered.iterrows()) if row['score'] != 0])
	return dcg/total

def main():
	#order on likelihood booked.
	data = pd.read_table('../../Data Mining VU data/predictions.txt')
	ndcg = data.groupby('srch_id').apply(calculate_ndcg)
	average_ndcg = sum(ndcg)/len(ndcg)
	print average_ndcg

if __name__ == '__main__':
	main()