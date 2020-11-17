

DATASET="pubmed"

for i in 0.01 0.1 1 10 40 80
  do
    for j in 0.01 0.1 0 1 10 50
      do
        for k in 0.1 0 1 10
          do
            for w1 in 0.1 0.4 0.6 0.9
              do
                for w2 in 0.1 0.4 0.6 0.9
                  do
                    python run_model.py --dataset $DATASET --temperature $i --alpha $j --beta $k --W1 $w1 --W2 $w2
                  done
              done
        done
      done
done

#for ((i = 0; i < 2; i++))
#do
#  python run_model.py
#done