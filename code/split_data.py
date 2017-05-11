import sys

#split into 90% training, 10% validation.

validation = open('../../Data Mining VU data/train_small.txt', 'w')
training = open('../../Data Mining VU data/val_small.txt', 'w')

count = 0

for line in open(sys.argv[1]):
	if count < 50000:
		validation.write(line)
	else:
		training.write(line)
	count += 1
	if count % 10000 == 0:
		print count

training.close()
validation.close()