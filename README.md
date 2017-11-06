# Learning Structured Text Representations

Code for the paper:

[Learning Structured Text Representations](https://arxiv.org/abs/1705.09207)  
Yang Liu and Mirella Lapata,
Accepted by TACL

## Dependencies
This code is implemented with Tensorflow and the data preprocessing is with Gensim

## Document Classification

#### Data
The pre-processed YELP 2013 data can be downloaded at https://drive.google.com/open?id=0BxGUKratNjbaZjFIR1MtbkdzZVU

#### Preprocessing
To preprocess the data, run
```
python prepare_data.py path-to-train path-to-dev path-to-test
```
This will generate a pickle file, the format for the input data can be found in the sample folder


#### Training
```
python cli.py --data_file path_to_pkl --rnn_cell lstm --batch_size 16 --dim_str 50 --dim_sem 75 --dim_output 5 --keep_prob 0.7 --opt Adadgrad
--lr 0.05 --norm 1e-4 --gpu -1 --sent_attention max --doc_attention max --log_period 5000
```
This will train the Tree-Matrix structured attention model in the paper on the training-set and present results on the devset/testset




## License
MIT
