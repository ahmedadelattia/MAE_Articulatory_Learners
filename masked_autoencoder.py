import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Bidirectional, GRU, Dense, Dropout, TimeDistributed, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import numpy as np
from time import time
from tqdm import tqdm
import os
from test_model import test_model, test_model_by_af
from metrics import *


class Mask(layers.Layer):
    #A layer that either randomly or deterministically masks PTs. Random masking is used for training and deterministic masking is used for testing.
    def __init__(self,  num_masked_PTs,  **kwargs):
        super().__init__(**kwargs)
        #The token that will be used to mask PTs. It is a trainable variable.
        self.mask_token = tf.Variable(
            tf.random.normal([1]), trainable=True
        )
        # Maximum number of PTs that will be masked.
        self.num_masked_PTs = num_masked_PTs


    def build(self, input_shape):
        
        self.num_PTs = input_shape[-2]
        self.time_steps = input_shape[-3]

    def random_masking(self, PTs):
        '''Randomly mask PTs
        Actual number of masked PTs is up to self.num_masked_PTs not equal to it. 
        For every batch, the number of masked PTs is randomly chosen between 0 and self.num_PTs - self.num_masked_PTs times. 
        It is possible that the same PT is chosen more than once. This sampling process is modeled by the birthday problem.
        
        Arguments:
        PTs: An array of the shape (batch_size, time_steps, num_PTs, 2) where the last dimension is the x and y coordinates of the PTs.
        Output:
        PTs_masked: An array of the same shape as PTs but with the masked PTs replaced with self.mask_token.
        '''
        masked_PTs =  tf.random.uniform((self.batch_size,self.num_masked_PTs), 0,self.num_PTs, tf.dtypes.int32) #indices of masked PTs
        #The following line repeats the masked_PTs for every time step. This means that the selection of maksed_PTs varies accross patches but not accross time steps.
        masked_PTs = tf.repeat(tf.expand_dims(masked_PTs, axis = 1), repeats = self.time_steps, axis = 1)
        masked_PTs = tf.reshape(masked_PTs, (self.batch_size, self.time_steps, self.num_masked_PTs))
        masked_PTs = tf.one_hot(masked_PTs, self.num_PTs, dtype = PTs.dtype) #one hot encoding of masked PTs
        #collapse the one hot matrix into a vector of length num_PTs, and cast it to bool and then back to PTs.dtype to make sure the elements of the one hot encoded matrix is only ones or zeros
        masked_PTs = tf.cast(tf.cast(tf.math.reduce_sum(masked_PTs, axis = -2),tf.bool),PTs.dtype)
        #The following line repeats the one hot encoded matrix for every coordinate (x and y) of the PTs. 
        masked_PTs = tf.concat((masked_PTs[...,tf.newaxis], masked_PTs[...,tf.newaxis]), axis=-1)
        #flip the one hot encoded matrix to only keep the unmasked PTs
        unmasked_pts_oh = 1 - masked_PTs
        #replace the masked PTs with self.mask_token
        PTs_masked = (PTs * unmasked_pts_oh) + (self.mask_token * masked_PTs)
        return PTs_masked

    def replace_manual_mask(self, PTs): 
        """
        For manual masking (during testing), before feeding the input to the model, we manually set the PTs to be masked with 100. This layer replaces the 100 with self.mask_token.
        100 was chosen because it is outside the range of the coordinates of the PTs. nan would've been a better choise but 100 is easier to work with.
        
        Arguments:
        PTs: An array of the shape (batch_size, time_steps, num_PTs, 2) where the last dimension is the x and y coordinates of the PTs.
        Outputs:
        PTs_masked: An array of the same shape as PTs but with the masked PTs replaced with self.mask_token.
        """
        PTs_masked = tf.where(tf.equal(PTs, 100), self.mask_token, PTs)
        return PTs_masked

    def mask_in_place(self, PTs, batch_size):
        """
        Detects whether the input is masked manually or randomly and calls the appropriate masking function.
        Arguments:
        PTs: An array of the shape (batch_size, time_steps, num_PTs, 2) where the last dimension is the x and y coordinates of the PTs.
        batch_size: The batch size of the input.
        Outputs:
        PTs_masked: An array of the same shape as PTs but with the masked PTs replaced with self.mask_token.
        """
        
        self.batch_size = batch_size
        cond = tf.equal(PTs,100)
        cond = tf.math.reduce_mean(tf.cast(cond, tf.float32))
        cond = tf.cast(cond, tf.bool)
        PTs_masked = tf.cond(cond, lambda: self.replace_manual_mask(PTs), lambda: self.random_masking(PTs))
        return PTs_masked


    def call(self, PTs, enable_masking = True):
        """
        Calls the masking function, with the option to disable masking (for debugging purposes).
        
        Arguments:
        PTs: An array of the shape (batch_size, time_steps, num_PTs, 2) where the last dimension is the x and y coordinates of the PTs.
        enable_masking: A boolean that determines whether to mask the input or not.
        Outputs:
        PTs_masked: An array of the same shape as PTs but with the masked PTs replaced with self.mask_token.
        """
        batch_size = tf.shape(PTs)[0]
        if enable_masking == False:
            return PTs
        else:
            PTs_masked = self.mask_in_place(PTs, batch_size)
            return PTs_masked

class MaskedAutoEncoder(tf.keras.Model):
    """
    A  tf.keras.Model subclass that implements a masked autoencoder. Model subclassing is finicky when it comes to initializing the model. It does 
    hoevever allow for more flexibility in the model architecture. The model is a masked autoencoder that takes as input a batch of frames of PTs, masks them randomly,
    and reconstructs the entire batch of PTs. The model is trained to minimize the reconstruction loss. 
    """
    def __init__(self, num_PTs=8, time_steps=200, num_masked_PTs = 1):
        """
        Arguments:
        num_PTs: The number of PTs in each patch.
        time_steps: The number of time steps in each frame.
        num_masked_PTs: The maximum number of PTs to be masked in each frame.
        """
        super(MaskedAutoEncoder, self).__init__()
        self.num_PTs = num_PTs
        self.time_steps = time_steps
        self.num_masked_PTs = num_masked_PTs

        #initialize all the layers used in the model
        self.masker = Mask(num_masked_PTs=self.num_masked_PTs)
    
        self.encoder_dense_1 = TimeDistributed(Dense(16, activation='relu'))
        self.encoder_dense_2 = TimeDistributed(Dense(32, activation='relu'))
        self.encoder_dense_3 = TimeDistributed(Dense(64, activation='relu'))
        self.encoder_dense_4 = TimeDistributed(Dense(128, activation='relu'))
        self.encoder_dense_5 = TimeDistributed(Dense(num_PTs, activation='relu'))
        
        self.encoder_bn_1 = BatchNormalization()
        self.encoder_bn_2 = BatchNormalization()
        self.encoder_bn_3 = BatchNormalization()
        self.encoder_bn_4 = BatchNormalization()
        self.encoder_bn_5 = BatchNormalization()

        self.encoder_dropout_1 = Dropout(0)
        self.encoder_dropout_2 = Dropout(0)
        self.encoder_dropout_3 = Dropout(0)
        self.encoder_dropout_4 = Dropout(0)  
        self.encoder_dropout_5 = Dropout(0)

        self.encoder_gru = Bidirectional(GRU(128, return_sequences=True, dropout=0))
        self.encoder_gru_1 = Bidirectional(GRU(64, return_sequences=True, dropout=0))

        self.decoder_gru = Bidirectional(GRU(128, return_sequences=True, dropout=0))
        self.decoder_gru_2 = Bidirectional(GRU(64, return_sequences=True, dropout=0))

        self.decoder_dense_1 = TimeDistributed(Dense(128, activation='relu'))
        self.decoder_dense_2 = TimeDistributed(Dense(64, activation='relu'))
        self.decoder_dense_3 = TimeDistributed(Dense(32, activation='relu'))
        self.decoder_dense_4 = TimeDistributed(Dense(16, activation='relu'))
        self.decoder_dense_5 = TimeDistributed(Dense(2))


        self.decoder_dropout_1 = Dropout(0)
        self.decoder_dropout_2 = Dropout(0)
        self.decoder_dropout_3 = Dropout(0)
        self.decoder_dropout_4 = Dropout(0)
        self.decoder_dropout_5 = Dropout(0)


        self.decoder_bn_1 = BatchNormalization()
        self.decoder_bn_2 = BatchNormalization()
        self.decoder_bn_3 = BatchNormalization()
        self.decoder_bn_4 = BatchNormalization()
        self.decoder_bn_5 = BatchNormalization()

        self.enable_masking = True


    def call(self, x):    
        """
        Forward pass of the model.
        Arguments:
        x: An array of the shape (batch_size, time_steps, num_PTs, 2) where the last dimension is the x and y coordinates of the PTs. Input to the model
        Outputs:
        out: An array of the same shape as x
        """
        #Apply the masking layer to the input
        encoder_input= self.masker(x, enable_masking = self.enable_masking)
        
        #Encoder
        #Each encoder layer is a dense layer followed by a dropout layer and a batch normalization layer. The output of each layer is passed to the next layer.
        #The encoder has 5 blocks, followed by a GRU layer and another GRU layer.
        #The dense blocks only processes the x-y dimensions, and the PTs aren't mixed with each other. The GRU layer mixes the PTs.
        dense_1 = self.encoder_dense_1(encoder_input)
        db_1 = self.encoder_dropout_1(dense_1)
        bn_1 = self.encoder_bn_1(db_1)

        dense_2 = self.encoder_dense_2(bn_1)
        db_2 = self.encoder_dropout_2(dense_2)
        bn_2 = self.encoder_bn_2(db_2)

        dense_3 = self.encoder_dense_3(bn_2)
        db_3 = self.encoder_dropout_3(dense_3)
        bn_3 = self.encoder_bn_3(db_3)

        dense_4 = self.encoder_dense_4(bn_3)
        db_4 = self.encoder_dropout_4(dense_4)
        bn_4 = self.encoder_bn_4(db_4)

        dense_5 = self.encoder_dense_5(bn_4)
        db_5 = self.encoder_dropout_5(dense_5)
        bn_4 = self.encoder_bn_5(db_5)
        
        #The output of the last encoder block is reshaped and passed to the GRU layers. The last axis is removed.
        #The PTs are mixed in the GRU layers.
        gru_input = tf.reshape(bn_4, (-1, bn_4.shape[-3], bn_4.shape[-1]*bn_4.shape[-2]))

        gru_encoder_1 = self.encoder_gru(gru_input)
        gru_encoder_2 = self.encoder_gru_1(gru_encoder_1)
        
        #Decoder
        #The decoder has two GRU layers, followed by 5 blocks 
        decoder_inputs = gru_encoder_2

        gru_decoder_1 = self.decoder_gru(decoder_inputs)
        gru_decoder_2 = self.decoder_gru_2(gru_decoder_1)
        #Restoring the last dimension
        dense_input = tf.reshape(gru_decoder_2, (-1, gru_decoder_2.shape[-2], self.num_PTs, gru_decoder_2.shape[-1]//self.num_PTs))

        dense_1 = self.decoder_dense_1(dense_input)
        db_1 = self.decoder_dropout_1(dense_1)
        bn_1 = self.decoder_bn_1(db_1)

        dense_2 = self.decoder_dense_2(bn_1)
        db_2 = self.decoder_dropout_2(dense_2)
        bn_2 = self.decoder_bn_2(db_2)

        
        dense_3 = self.decoder_dense_3(bn_2)
        db_3 = self.decoder_dropout_3(dense_3)
        bn_3 = self.decoder_bn_3(db_3)
        
        dense_4 = self.decoder_dense_4(bn_3)
        db_4 = self.decoder_dropout_4(dense_4)
        bn_4 = self.decoder_bn_4(db_4)
        
        bn_5 = self.decoder_bn_5(bn_4)
        out = self.decoder_dense_5(bn_5)

        return out


    def correlation(self, y_true, y_pred):
        """Calculates the correlation between the predicted and the ground truth, for validation
        Arguments:
        y_true: The ground truth
        y_pred: The predicted values
        Outputs:
        ppmc_score: The correlation between the predicted and the ground truth
        """
        
        corr_score = 0
        for i in range(self.num_PTs):
            for j in range(2):
                x = y_true[...,i, j]
                y = y_pred[...,i, j]
                mx = K.mean(x)
                my = K.mean(y)
                xm, ym = x - mx, y - my
                r_num = K.sum(tf.multiply(xm, ym))
                r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
                r = r_num / r_den
                corr_score += r/2     
        ppmc_score = corr_score/self.num_PTs
        return ppmc_score

if __name__ == '__main__':
    #Setting up control variables
    pretrain = True #whether to pretrain the model on reconstruction (no masking)
    pretrain_from_scratch = True #whether to pretrain the model from scratch or load the pretrained weights
    spkrs = open("../data/UWXRMB/speakers_list.txt", "r").readlines() #list of speakers
    sparse_speakers = open("../data/UWXRMB/sparse_speakers.txt", "r").readlines() #list of speakers with less than few utterances
    spkrs = [spkr for spkr in spkrs  if spkr not in sparse_speakers] #only train on speakers with enough utterances

    for spkr in tqdm(spkrs):
        print("Now training speaker ", spkr)
        spkr = spkr.strip("\n")
        #Loading Data
        PTs_train=np.load(f'../data/UWXRMB/trimmed/Train_files/speaker_files/resampled/{spkr}/TVs_{spkr}_train_200.npy')
        PTs_test=np.load(f'../data/UWXRMB/trimmed/Train_files/speaker_files/resampled/{spkr}/TVs_{spkr}_test_200.npy')
        print("DATA LOADED")

        model = MaskedAutoEncoder(num_PTs = 8)

        print("MODEL CREATED")

        #Callbacks
        earlystop = EarlyStopping(monitor='val_loss', patience=50)
        pretraining_checkpoints = ModelCheckpoint(f"./model_outs/AFS/trimmed/new_testset-overlap/pretrain/resampled/{spkr}/MAE_pretrained/MAE_pretrained_{spkr}.tf", monitor='val_loss', save_best_only=True, save_weights_only=True)
        tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = "./tensorboard_logs",
                                                        histogram_freq = 1,
                                                        profile_batch = '200,220')


        #pretrain by reconstruction (no masking)
        model.compile("adam",loss = "mae", metrics =  [model.correlation])   
        if pretrain:
            model.enable_masking = False
            if pretrain_from_scratch:
                print("PRETRAINING")
                model.fit(PTs_train, PTs_train, batch_size=32, epochs=10000, verbose = 0, callbacks=[earlystop,tboard_callback], validation_data=(PTs_test, PTs_test))
            else: 
                model(PTs_test)
                model.load_weights(f"././model_outs/AFS/trimmed/new_testset-overlap/pretrain/resampled/{spkr}/MAE_pretrained/MAE_pretrained_{spkr}.tf")
                
            print("TESTING PRETRAINED MODEL")
            # print(tf.reduce_mean(tf.math.abs(model(PTs_test).numpy() - PTs_test), axis = [0,1,3]))
            out = model(PTs_test).numpy().reshape(-1, 200, 8, 2)
            PTs_test = PTs_test.reshape(-1, 200, 8, 2)

            corr_avg_x, avg_corr_x = compute_corr_score(out[...,0], PTs_test[...,0])
            corr_avg_y, avg_corr_y =compute_corr_score(out[...,1], PTs_test[...,1])

            pretrain_log = f"\nCorrelation Across X axis : {corr_avg_x} Average correlation across X axis: {avg_corr_x}"
            pretrain_log += f"\nCorrelation Across Y axis : {corr_avg_y} Average correlation across Y axis: {avg_corr_y}"
            print(pretrain_log)
        for num_masked_PTs in range(1,9):
            st_time = time()
            model_name = f"MAE_Trained_on_upto_{num_masked_PTs}_AFs"
            model_dir = f"./model_outs/AFS/trimmed/new_testset-overlap/pretrain/resampled/{spkr}/{model_name}"
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)
            if pretrain:
                with open(f'{model_dir}/log.txt', 'a') as f:
                    f.write(pretrain_log)

            checkpoints = ModelCheckpoint(f"{model_dir}/{model_name}.tf", monitor='val_loss', save_best_only=True, save_weights_only=True)

            #training
            print(f"\n\n\n#######TRAINING WITH UPTO {num_masked_PTs} MASKED AFS")
            model = MaskedAutoEncoder(num_masked_PTs = num_masked_PTs, num_PTs = 8)
            model(PTs_test)
            if pretrain:
                model.load_weights(f"./model_outs/AFS/trimmed/new_testset-overlap/pretrain/resampled/{spkr}/MAE_pretrained/MAE_pretrained_{spkr}.tf")

            model.enable_masking = True    
            model.compile("adam",loss = "mae", metrics =  [model.correlation])   
            history = model.fit(PTs_train, PTs_train, batch_size=32, epochs=10000, verbose = 0, callbacks=[earlystop], validation_data=(PTs_test, PTs_test))
           
            print("TRAINING COMPLETE")
            model(PTs_test)
            model.load_weights(f"./{model_dir}/{model_name}.tf")
            # logging
            end_time = time()
            log_file=f"\n mae {history.history['val_loss'][-1]}, correlation {history.history['val_correlation'][-1]}\n"
            with open(f'{model_dir}/log.txt', 'a') as f:
                f.write(f"TRAINING DONE IN {(end_time - st_time)/60 :.0f} minutes")
                f.write(log_file)

            test_model(model, PTs_test, model_dir) #Returns the average correlation per number of PTs masked
            test_model_by_af(model, PTs_test, model_dir, is_limiting = False) #Returns the average correlation for each PT. Non limiting (all cases considered)
            test_model_by_af(model, PTs_test, model_dir, is_limiting = True) #Returns the average correlation for each PT. Limiting (corner cases ignored)