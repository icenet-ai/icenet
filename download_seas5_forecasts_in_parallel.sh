#!/bin/bash

init_date="2011-11-01"
n_iters=$((12 * 9 + 2)) # Number of iterations to reach 2012-12-01
max_n_jobs=12 # No more than this many jobs can run

for ((i=0; i<$n_iters; i++)); do

        # Necessary to avoid max number of requests error
        while test $(jobs -p | wc -w) -ge $max_n_jobs; do 
                sleep 1
        done

        # echo -e "import time; time.sleep(20)" | python3 &  # Testing
        python3 -u icenet2/download_seas5_forecasts.py --init_date=$init_date > seas5_download_logs/"$init_date.txt" 2>&1 &

        echo -e "Running $(jobs -p | wc -w) jobs after submitting $init_date"

        init_date=`date "+%C%y-%m-%d" -d "$init_date+1 month"`
        sleep 1 # Necessary to avoid 'API rate limit exceeded' error
done
