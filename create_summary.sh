#!/bin/bash
for x in */; do 
	cd $x
	if [ -f log.txt ]; then
		model=$(cat args.txt |grep model)
		score=$(cat log.txt | grep Accuracy | tail -1)
		echo "$x, $model, $score" >> ../summary.txt
	fi
	cd ..
done

cat summary.txt | sort --field-separator=',' --key=3 > summary-sorted.txt
mv summary-sorted.txt summary.txt
