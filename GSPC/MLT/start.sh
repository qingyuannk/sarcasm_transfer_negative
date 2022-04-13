echo "MLT model train"
dirs="2022-4-12/2/log"
[ ! -d $dirs ] && mkdir -p $dirs
[ -d $dirs ] && echo "Directory exists"

nohup python3 train.py param.yaml 0  > $dirs/final_result-1-3.txt 2>&1 &
nohup python3 train.py param.yaml 1  > $dirs/final_result-2-3.txt 2>&1 &
nohup python3 train.py param.yaml 2  > $dirs/final_result-3-3.txt 2>&1 &
nohup python3 train.py param.yaml 3  > $dirs/final_result-4-3.txt 2>&1 &
nohup python3 train.py param.yaml 4  > $dirs/final_result-5-3.txt 2>&1 &
nohup python3 train.py param.yaml 5  > $dirs/final_result-6-3.txt 2>&1 &
nohup python3 train.py param.yaml 6  > $dirs/final_result-7-3.txt 2>&1 &
nohup python3 train.py param.yaml 7  > $dirs/final_result-8-3.txt 2>&1 &