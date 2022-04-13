
import os
import sys
import time
from tkinter.tix import DirSelectBox
import numpy as np



def gpu_info(gpu_index):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('\n')[gpu_index].split('|')
    power = int(gpu_status[1].split()[-3][:-1])
    memory = int(gpu_status[2].split('/')[0].strip()[:-3])
    return power, memory    


def narrow_setup():
    id = [0,1,2,3,4,5,6,7]
    for gpu_id in id:
        gpu_power, gpu_memory = gpu_info(gpu_id)
        if(gpu_memory < 10000 or gpu_power < 20):  # set waiting condition
            return gpu_id
    return -1
           
params=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
random_num=random_num=[32,64,4096,8192,10240,20480,102400,102400]
model_output=2
def run():
    for alpha in params:
        for index,num in enumerate(random_num):
            gpu_id = narrow_setup()
            interval=60
            while(gpu_id == -1):
                gpu_id = narrow_setup()
                time.sleep(interval)
            dir="../result/{}/{}/".format(model_output,int(alpha*10))
            log=dir+'log/'
            model=dir+'model/'
            result=dir+'result/'
            if not os.path.exists(dir):
                os.makedirs(dir)
            if not os.path.exists(log):   
                os.makedirs(log)
            if not os.path.exists(model):   
                os.makedirs(model)
            if not os.path.exists(result):   
                os.makedirs(result)
            filename = log+"alpha_result-{}-2.txt".format(index)
            cmd = "nohup python3 train.py param.yaml {}  {} {} > {} 2>&1 &".format(gpu_id,alpha,num,filename)
            print(cmd)
            os.system(cmd)
            interval_1=1
            gpu_power, gpu_memory = gpu_info(gpu_id)
            gpu = 'gpu id:%d' % gpu_id
            gpu_power_str = 'gpu power:%d W |' % gpu_power
            gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
            while(gpu_memory < 10000 or gpu_power < 50):
                gpu_power, gpu_memory = gpu_info(gpu_id)
                symbol = 'monitoring: ' + '>' * interval
                sys.stdout.write('\r' +  gpu + ' ' + gpu_memory_str + ' ' + gpu_power_str + ' '+symbol)
                sys.stdout.flush()                
                time.sleep(interval_1)
                interval_1+=5
            print("\n")

            



if __name__ == '__main__':
    run()
    narrow_setup()
