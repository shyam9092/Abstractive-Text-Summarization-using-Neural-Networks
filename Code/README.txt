Team 13: Abstractive Text Summarization
------------------------------

This is a readme file to run and test the EEE 511: Project .
Current directory has structures like below.
------------------------
├── Code
│   ├── training.py
│   ├── testing.py
│   ├── install.sh
│   ├── inference_encoder_model1.pbtxt
│   ├── inference_decoder_model1.pbtxt
│   └── README.txt

Steps to run training.py and testing.py
--------------------------
The code folder contains the following:
training.py:                        the source code which contains the text summarization model
inference_encoder_model1.pbtxt:     the source file which contains the trained weights for the encoder.
inference_decoder_model1.pbtxt:     the source file which contains the trained weights for the decoder.
testing.py:                         the code which will test the summaries
install.sh:                         the bash script which will install all the depencies for the model
README.txt:                         the README file which contains the instructions for the project. 

Follow steps to test the model:

Download the dataset from the following link:
https://drive.google.com/drive/folders/1hLJ55W_T-dz9FLHdj4x_IWCAwUPL93CY?usp=sharing

=> Unzip the dataset 

$ unzip Reviews.csv.zip

=> It will create Reviews.csv file which contains our dataset

=> SSH into your google cloud instance

Copy the following files using the scp command:

1. training.py
2. testing.py
3. install.sh
4. inference_encoder_model1.pbtxt
5. inference_decoder_model1.pbtxt
6. Reviews.csv

$ gcloud compute scp training.py testing.py install.sh Reviews.csv example-instance:~/

=> Run the following command to install the depencies:

$ ./install.sh

=> Now, all the depencies are installed. Run the following command to start training.  
=> Alternatively, you can skip the following step and use the pre-trained weights for testing using inference_encoder_model1.pbtxt and inference_decoder_model1.pbtxt.

$ python3 training.py

=> Following log will appear:

2021-04-12 08:14:01.932401: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2021-04-12 08:14:01.932430: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
[nltk_data] Downloading package stopwords to /home/shyam/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
% of rare words in vocabulary: 64.94107820529116
Total Coverage of rare words: 1.379596176857665
% of rare words in vocabulary: 75.83159755096888
Total Coverage of rare words: 3.4313692657017127
2021-04-12 08:15:07.381072: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-04-12 08:15:07.381269: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2021-04-12 08:15:07.381285: W tensorflow/stream_executor/cuda/cuda_driver.cc:326] failed call to cuInit: UNKNOWN ERROR (303)
2021-04-12 08:15:07.381303: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (instance-2): /proc/driver/nvidia/version does not exist
2021-04-12 08:15:07.381474: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-04-12 08:15:07.382340: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 150)]        0                                            
__________________________________________________________________________________________________
embedding (Embedding)           (None, 150, 100)     1930900     input_1[0][0]                    
__________________________________________________________________________________________________
lstm (LSTM)                     [(None, 150, 300), ( 481200      embedding[0][0]                  
__________________________________________________________________________________________________
input_2 (InputLayer)            [(None, None)]       0                                            
__________________________________________________________________________________________________
lstm_1 (LSTM)                   [(None, 150, 300), ( 721200      lstm[0][0]                       
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, None, 100)    1930900     input_2[0][0]                    
__________________________________________________________________________________________________
lstm_2 (LSTM)                   [(None, 150, 300), ( 721200      lstm_1[0][0]                     
__________________________________________________________________________________________________
lstm_3 (LSTM)                   [(None, None, 300),  481200      embedding_1[0][0]                
                                                                 lstm_2[0][1]                     
                                                                 lstm_2[0][2]                     
__________________________________________________________________________________________________
attention_layer (AttentionLayer ((None, None, 300),  180300      lstm_2[0][0]                     
                                                                 lstm_3[0][0]                     
__________________________________________________________________________________________________
concat_layer (Concatenate)      (None, None, 600)    0           lstm_3[0][0]                     
                                                                 attention_layer[0][0]            
__________________________________________________________________________________________________
time_distributed (TimeDistribut (None, None, 19309)  11604709    concat_layer[0][0]               
==================================================================================================
Total params: 18,051,609
Trainable params: 18,051,609
Non-trainable params: 0
______________________________________________________________inference_encoder_model1.pbtxt
783/783 [==============================] - 1914s 2s/step - loss: 0.8330 - val_loss: 0.5035


=> Run the following command to run the testing phase of the model:

$ python3 testing.py

=> Following log will appear: 

Review: noodles fine filling make simple cheap meal vegetables slices meat eggs thrown spice packet bit overwhelming starting stick using quarter packet first dehydrated vegetables nice touch though take bit time noodles rehydrated instead following instructions package letter try throwing veggies bit time say minutes ahead water boiling set veggies separate container water time cook noodles 
Original summary: pretty good 
Predicted summary:  pretty good
Precision is :0.75
Recall is :0.75
F Score is :0.7500009500000033


Review: person forever counting calories great little treat black licorice lover pieces calories hooked ever since tasted 
Original summary: delicious 
Predicted summary:  great licorice
Precision is :0.4
Recall is :0.6
F Score is :0.48000095200000475


Review: bones kept dogs busy half hour like natural bones leave flaky mess like bones one thing like cost thought received good value would really like buy often 
Original summary: dogs love them 
Predicted summary:  dogs love em
Precision is :0.6923076923076923
Recall is :0.6
F Score is :0.6428580931122487


Review: kept raising price product great remained kept raising price last order enough going back 
Original summary: will be returning 
Predicted summary:  not what expected
Precision is :0.2777777777777778
Recall is :0.2777777777777778
F Score is :0.2777787277777868


Overall precision is:  0.48499037898035086
Overall recall is:  0.37439055990030795
Overall fscore is:  0.3907306091631807


