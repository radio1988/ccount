mkdir -p auto_neg

for f in res/classification1/pos/*gz
do 
name=$(basename $f)
python auto_neg.py $f > auto_neg/$name.log
done