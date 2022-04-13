- ##### The  config parameters setting of model training and testing

  ```yaml
  model_output: 2   					##two or three classification 
  epoch: 30         					##the epoches of training
  batch: 32         					## the batch sizes of training
  positive_se : 0.2 					## the positive sentence in Sp(only use in Sp-SPC)
  lr: 0.00001       					## the learning rate of Amda 
  clean_tag: 'neutral'        ## Two classification filter the neutral out
  class: ["negative","positive"]  ## Two classificatioin
  #clean_tag: ''        ## Three classification filter the neutral out
  #class: ["negative","positive","neutral"]  ## Three classificatioin
  UNCASED: "../../transformers/bertweet-base" ##bertweet local path
  VOCAB: 'vocab.txt'
  path: "2022-4-12"            ##save model checkpoint,log,result dictory
  start_dir: "../data/SemEval2017-Task1/train_data" ## The  dictory of train dataset
  dataset:
    senti_train: []
    senti_test: ["../data/SemEval2017-Task1/test_data/SemEval2017-task4-test.subtask-A.english.txt"]   
    sms_train: ["../data/SemEval2018-Task3/train_data/sar_train.txt"]
    sms_test: ["../data/SemEval2018-Task3/test_data/sar_test.txt"]
    
  ```

  

- ##### The procession of training and testing

  ```bash
  cd SPC && bash -x start.sh
  cd MLT && bash -x start.sh
  cd SPC_S && bash -x start.sh
  cd Sp_SPC && bash -x start.sh
  ```

- ##### The analysis of experiments in all model

  ```bash
  cd SPC/path(config file define) && python3 RA.py
  cd MLT/path(config file define) && python3 RA.py
  cd SPC_S/path(config file define) && python3 RA.py
  cd Sp_SPC/path(config file define) && python3 RA.py
  ```

- Weight $\alpha$ in the Sp Task  

  ```bash
  - python3 run.py
  - cd path/2(3)
  - python3 weighted.py
  ```

  

- Encoding Feature Visualization

  ```bash
  Generate_picture.ipynb
  ```

- Case Study

  ```
  python3 example.py
  ```

  