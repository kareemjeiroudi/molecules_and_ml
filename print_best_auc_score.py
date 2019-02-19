max = 0.5
fp = open("OUTPUT_02.txt")
for line in fp.readlines():
	if line.startswith("auc"):
		auc_score = float(line.split(' ')[2])
		if auc_score > max:
			max = auc_score
fp.close()
print("The best found score is {}".format(max))