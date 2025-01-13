#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 21:51:22 2024
@author: Ruiyang Zhang (Original)
Reimplemented for TensorFlow 2.0
@author: Harrish Joseph
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import os, sys
from random import shuffle
tf.keras.mixed_precision.set_global_policy('float32')

plt.close('all')
folder_name = time.strftime('PhyLSTM3')#[0][0]
os.makedirs(f'results/{folder_name}', exist_ok=True)  
exten = 'jpg'
# Set GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class DeepPhyLSTM(tf.keras.Model):
    def __init__(self, eta, eta_t, g, ag, ag_c, r, Phi_t):
        super(DeepPhyLSTM, self).__init__()
        # Data initialization
        self.eta = tf.convert_to_tensor(eta, dtype=tf.float32)
        self.eta_t = tf.convert_to_tensor(eta_t, dtype=tf.float32)
        self.g = tf.convert_to_tensor(g, dtype=tf.float32)
        self.ag = tf.convert_to_tensor(ag, dtype=tf.float32)
        self.ag_c = tf.convert_to_tensor(ag_c, dtype=tf.float32)
        self.r = tf.convert_to_tensor(r, dtype=tf.float32)
        self.Phi_t = tf.convert_to_tensor(Phi_t, dtype=tf.float32)
        self.learning_rate = .001
        self.optimizer = Adam(learning_rate=self.learning_rate)# Optimizer

        # Enable mixed precision training for better performance
        # tf.keras.mixed_precision.set_global_policy('mixed_float16')
        tf.keras.backend.set_floatx('float32')
        # LSTM Model 1
        self.LSTM_model = tf.keras.Sequential([
            keras.layers.LSTM(100, return_sequences=True, input_shape=(None, 1), name='lstm_1'),
            keras.layers.Activation('relu', name='relu_1'),
            keras.layers.LSTM(100, return_sequences=True, name='lstm_2'),
            keras.layers.Activation('relu', name='relu_2'),
            keras.layers.LSTM(100, return_sequences=True, name='lstm_3'),
            keras.layers.Activation('relu', name='relu_3'),
            keras.layers.Dense(3 * self.eta.shape[2], name='dense_1')
        ])

        # LSTM Model F
        self.LSTM_model_f = tf.keras.Sequential([
            keras.layers.LSTM(100, return_sequences=True, input_shape=(None, 3*self.eta.shape[2]), name='lstm_4'),
            keras.layers.Activation('relu', name='relu_4'),
            keras.layers.LSTM(100, return_sequences=True, name='lstm_5'),
            keras.layers.Activation('relu', name='relu_5'),
            keras.layers.LSTM(100, return_sequences=True, name='lstm_6'),
            keras.layers.Activation('relu', name='relu_6'),
            keras.layers.Dense(self.eta.shape[2], name='dense_2')
        ])

        # LSTM Model G
        self.LSTM_model_g = tf.keras.Sequential([
            keras.layers.LSTM(100, return_sequences=True, input_shape=(None, 2*self.eta.shape[2]), name='lstm_7'),
            keras.layers.Activation('relu', name='relu_7'),
            keras.layers.LSTM(100, return_sequences=True, name='lstm_8'),
            keras.layers.Activation('relu', name='relu_8'),
            keras.layers.LSTM(100, return_sequences=True, name='lstm_9'),
            keras.layers.Activation('relu', name='relu_9'),
            keras.layers.Dense(self.eta.shape[2], name='dense_3')
        ])


    def call(self, ag):
        
        output = self.LSTM_model(ag)
        
        eta = output[:, :, 0:self.eta.shape[2]]
        eta_dot = output[:, :, self.eta.shape[2]:2*self.eta.shape[2]]
        g = output[:, :, 2*self.eta.shape[2]:]
        
        Phi_tf = tf.cast(Phi_t[:eta.shape[0], :eta.shape[1], :eta.shape[1]], tf.float32)
        eta_t = tf.matmul(Phi_tf, eta)
        eta_tt = tf.matmul(Phi_tf, eta_dot)
        g_t = tf.matmul(Phi_tf, g)

        return eta, eta_t, eta_tt, eta_dot, g, g_t

    def net_f(self, ag):
        eta, eta_t, eta_tt, eta_dot, g, g_t = self.call(ag)
        f = self.LSTM_model_f(tf.concat([eta, eta_dot, g], 2))
        r = eta_tt + f
        eta_dot1 = eta_dot[:, :, 0:1]
        r_dot = self.LSTM_model_g(tf.concat([eta_dot1, g], 2))

        return eta_t, eta_dot, g_t, r_dot, r

    # @tf.function(experimental_relax_shapes=True)
    def train_step(self, eta_tf, eta_t_tf, g_tf, ag_tf,  ag_c_tf,r_tf, Phi_tf):
        # trainable_vars = self.trainable_variables
        trainable_vars = (
            self.LSTM_model.trainable_variables+self.LSTM_model_f.trainable_variables+
            self.LSTM_model_g.trainable_variables
        )
        
        with tf.GradientTape() as tape:
            # tape.watch(trainable_vars)
            # Forward pass
            eta_pred, eta_t_pred, eta_tt_pred, eta_dot_pred, g_pred, g_t_pred = self(ag_tf)
            eta_t, eta_dot, g_t, r_dot, r = self.net_f(ag_tf)

            loss_u = tf.reduce_mean(tf.square(eta_tf - eta_pred))
            loss_udot = tf.reduce_mean(tf.square(eta_t_tf - eta_t_pred))
            loss_g = tf.reduce_mean(tf.square(g_tf - g_pred))
            loss_g_t = tf.reduce_mean(tf.square(g_t - r_dot))
            loss_ut_c = tf.reduce_mean(tf.square(eta_t_pred - eta_dot_pred))
            loss_e = tf.reduce_mean(tf.square(r_tf- r))
            total_loss = loss_u + loss_udot + loss_g + loss_ut_c + loss_e +loss_g_t

        # Compute gradients
        # trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Apply gradients
        # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        for var, grad in zip(trainable_vars, gradients):
            # print(f"Gradient for {var.name}: {grad}")
            if grad is None:
                print(f"No gradient for variable: {var.name}")
                
        return total_loss, loss_u, loss_udot, loss_g, loss_ut_c, loss_e,loss_g_t

    def train(self, num_epochs, learning_rate, bfgs=0):
        Loss_u, Loss_udot, Loss_g, Loss_ut_c, Loss_e, Loss, Loss_val,Loss_gt = [], [], [], [], [], [], [],[]
        best_loss = float('inf')
        total_samples = tf.shape(self.ag)[0]
        ratio_split = 0.8
        train_size = tf.cast(tf.cast(total_samples, tf.float32) * ratio_split, tf.int32)
        
        # Create shuffled indices
        indices = tf.random.shuffle(tf.range(total_samples))
        Ind_tr = indices[:train_size]
        Ind_val = indices[train_size:]
    
        # Split data for training and validation using TensorFlow indexing
        self.ag_tr = tf.gather(self.ag, Ind_tr)
        self.eta_tr = tf.gather(self.eta, Ind_tr)
        self.eta_t_tr = tf.gather(self.eta_t, Ind_tr)
        self.g_tr = tf.gather(self.g, Ind_tr)
        self.r_tr     = tf.gather(self.r, Ind_tr)
    
        self.ag_val = tf.gather(self.ag, Ind_val)
        self.eta_val = tf.gather(self.eta, Ind_val)
        self.eta_t_val = tf.gather(self.eta_t, Ind_val)
        self.g_val = tf.gather(self.g, Ind_val)
        self.r_val     = tf.gather(self.r, Ind_val)

        for epoch in range(num_epochs):
            start_time = time.time()


            # Train step
            train_loss, loss_u, loss_udot, loss_g, loss_ut_c, loss_e,loss_g_t = self.train_step(
                self.eta_tr, self.eta_t_tr, self.g_tr, self.ag_tr,  self.ag_c,self.r_tr, self.Phi_t)

            # Validation loss
            val_loss,_,_,_,_,_,_ = self.train_step(
                self.eta_val, self.eta_t_val, self.g_val, self.ag_val, self.ag_c,self.r_val,  self.Phi_t)

            # Append losses
            Loss_u.append(loss_u.numpy())
            Loss_udot.append(loss_udot.numpy())
            Loss_g.append(loss_g_t.numpy())
            Loss_gt.append(val_loss.numpy())
            Loss_ut_c.append(loss_ut_c.numpy())
            Loss_e.append(loss_e.numpy())
            Loss.append(train_loss.numpy())
            Loss_val.append(val_loss.numpy())

            # Update best loss and save model
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_weights(f'{folder_name}/best_model_weights.weights.h5')

            elapsed = time.time() - start_time
            print(f'Epoch: {epoch}, Train Loss: {train_loss.numpy():.3e}, '
                  f'Val Loss: {val_loss.numpy():.3e}, Best Loss: {best_loss:.3e}, Time: {elapsed:.2f}s')

        return Loss_u, Loss_udot, Loss_g, Loss_ut_c, Loss_e, Loss, Loss_val, best_loss,Loss_gt

    # @property
    # def trainable_variables(self):
    #     # Collect all trainable variables across models
    #     variables = []
    #     for model in [self.LSTM_model, self.LSTM_model_f, self.LSTM_model_g]:
    #         variables.extend(model.trainable_variables)
    #     return variables

    def predict(self, ag_star, Phi_star):
        # Prediction method with multiple return values
        eta_star = self.LSTM_model(ag_star)
        eta = eta_star[:, :, 0:self.eta.shape[2]]
        eta_dot = eta_star[:, :, self.eta.shape[2]:2*self.eta.shape[2]]
        g = eta_star[:, :, 2*self.eta.shape[2]:]
        Phi_star = tf.cast(Phi_star[:eta.shape[0], :eta.shape[1], :eta.shape[1]],tf.float32)
        eta_t = tf.matmul(tf.cast(Phi_star, tf.float32), tf.cast(eta, tf.float32))
        eta_tt = tf.matmul(Phi_star, eta_dot)
        
        f = self.LSTM_model_f(tf.concat([eta, eta_dot, g], 2))
        r = eta_tt + f

        eta_dot1 = eta_dot[:, :, 0:1]
        r_dot = self.LSTM_model_g(tf.concat([eta_dot1, g], 2))
        return eta, eta_t, eta_tt, eta_dot, g,r, r_dot


mat = scipy.io.loadmat('data_boucwen.mat')

t = mat['time']
dt = 0.02
n1 = int(dt / 0.005)
# t = t[::n1]

ag_data = mat['input_tf']#[:, ::n1]  # ag, ad, av
u_data = mat['target_X_tf']#[:, ::n1]
ut_data = mat['target_Xd_tf']#[:, ::n1]
utt_data = mat['target_Xdd_tf']#[:, ::n1]
ag_data = ag_data.reshape([ag_data.shape[0], ag_data.shape[1], 1])
u_data = u_data.reshape([u_data.shape[0], u_data.shape[1], 1])
ut_data = ut_data.reshape([ut_data.shape[0], ut_data.shape[1], 1])
utt_data = utt_data.reshape([utt_data.shape[0], utt_data.shape[1], 1])

ag_pred = mat['input_pred_tf']#[:, ::n1]  # ag, ad, av
u_pred = mat['target_pred_X_tf']#[:, ::n1]
ut_pred = mat['target_pred_Xd_tf']#[:, ::n1]
utt_pred = mat['target_pred_Xdd_tf']#[:, ::n1]
ag_pred = ag_pred.reshape([ag_pred.shape[0], ag_pred.shape[1], 1])
u_pred = u_pred.reshape([u_pred.shape[0], u_pred.shape[1], 1])
ut_pred = ut_pred.reshape([ut_pred.shape[0], ut_pred.shape[1], 1])
utt_pred = utt_pred.reshape([utt_pred.shape[0], utt_pred.shape[1], 1])

n = u_data.shape[1]
phi1 = np.concatenate([np.array([-3 / 2, 2, -1 / 2]), np.zeros([n - 3, ])])
temp1 = np.concatenate([-1 / 2 * np.identity(n - 2), np.zeros([n - 2, 2])], axis=1)
temp2 = np.concatenate([np.zeros([n - 2, 2]), 1 / 2 * np.identity(n - 2)], axis=1)
phi2 = temp1 + temp2
phi3 = np.concatenate([np.zeros([n - 3, ]), np.array([1 / 2, -2, 3 / 2])])
Phi_t0 = 1 / dt * np.concatenate(
        [np.reshape(phi1, [1, phi1.shape[0]]), phi2, np.reshape(phi3, [1, phi3.shape[0]])], axis=0)
Phi_t0 = np.reshape(Phi_t0, [1, n, n])

# ag_star = ag_data
# eta_star = u_data
# eta_t_star = ut_data
# eta_tt_star = utt_data
noPred = 2
# ag_c_star = np.concatenate([ag_data, ag_pred[0:noPred]])
# r_star = -ag_c_star
# eta_c_star = np.concatenate([u_data, u_pred[0:noPred]])
# eta_t_c_star = np.concatenate([ut_data, ut_pred[0:noPred]])
# eta_tt_c_star = np.concatenate([utt_data, utt_pred[0:noPred]])

eta = u_data
ag = ag_data
r = -ag
eta_t = ut_data
eta_tt = utt_data
g = -eta_tt - ag
ag_c = ag_data

# Training Data
eta_train = u_data
ag_train = ag_data
r_train = r
eta_t_train = eta_t
eta_tt_train = eta_tt
g_train = g
ag_c_train = ag_c

Loss_BFGS = np.empty([0])
Loss_val_BFGS = np.empty([0])
Phi_t = np.repeat(Phi_t0, ag_c.shape[0], axis=0)


# Training return Loss_u, Loss_udot, Loss_g, Loss_ut_c, Loss_e, Loss, Loss_val, best_loss,Loss_gt
model = DeepPhyLSTM(eta_train, eta_t_train, g_train, ag_train, ag_c, r, Phi_t)
# model.load_weights('PhyLSTM312181851it2500/best_model_weights')
#%%
totaltraintime = time.time()
Loss_u1, Loss_udot1, Loss_g1, Loss_ut_c1,  Loss_e1, Loss1, Loss_val, best_loss,Loss_gt_c1, = model.train(num_epochs=50, learning_rate=1e-3, bfgs=0)
# totalfinishtrain =time.time()
print(f'Total training time:{(time.time()-totaltraintime)/60:.3f}min')
train_loss = Loss1
test_loss = Loss_val
best_loss = best_loss

plt.figure()
plt.plot(np.log(train_loss), label='loss')
plt.plot(np.log(test_loss), label='loss_val')
plt.legend()
plt.savefig(os.path.join(folder_name, f'loss.{exten}'),format=f'{exten}',  dpi=300)
#%% Results
# Training performance
X_train = ag_data[0:noPred]
y_train_ref = u_data[0:noPred]
yt_train_ref = ut_data[0:noPred]
ytt_train_ref = utt_data[0:noPred]
r_train_ref = -X_train
g_train_ref = -ytt_train_ref + r_train_ref

eta, eta_t, eta_tt, eta_dot, g,r,r_dot = model.predict(X_train, np.repeat(Phi_t0, len(X_train), axis=0))
# r = model.predict(X_train, np.repeat(Phi_t0, len(X_train), axis=0))
y_train_pred = eta
yt_train_pred = eta_t
ytt_train_pred = eta_tt
g_train_pred = -eta_tt + r

dof = 0
for ii in range(len(y_train_ref)):
    plt.figure()
    plt.plot(y_train_ref[ii, :, dof], '-', label='True')
    plt.plot(y_train_pred[ii, :, dof], '--', label='Predict')
    plt.title('Training_u')
    plt.legend()
    plt.savefig(os.path.join(folder_name, f'training_u.{exten}'),format=f'{exten}',  dpi=300)

for ii in range(len(y_train_ref)):
    plt.figure()
    plt.plot(yt_train_ref[ii, :, dof], label='True')
    plt.plot(yt_train_pred[ii, :, dof], label='Predict')
    plt.title('Training_u_t')
    plt.legend()
    plt.savefig(os.path.join(folder_name, f'Training_u_t.{exten}'),format=f'{exten}',  dpi=300)
    

for ii in range(len(y_train_ref)):
    plt.figure()
    plt.plot(ytt_train_ref[ii, :, dof], label='True')
    plt.plot(ytt_train_pred[ii, :, dof], label='Predict')
    plt.title('Training_u_tt')
    plt.legend()
    plt.savefig(os.path.join(folder_name, f'Training_u_tt.{exten}'),format=f'{exten}',  dpi=300)

for ii in range(len(y_train_ref)):
    plt.figure()
    plt.plot(g_train_ref[ii, :, dof], label='True')
    plt.plot(g_train_pred[ii, :, dof], label='Predict')
    plt.title('Training_g')
    plt.legend()
    plt.savefig(os.path.join(folder_name, f'Training_g.{exten}'),format=f'{exten}',  dpi=300)

for ii in range(len(y_train_ref)):
    plt.figure()
    plt.plot(y_train_ref[ii, :, dof], g_train_ref[ii, :, dof], label='True')
    plt.plot(y_train_pred[ii, :, dof], g_train_pred[ii, :, dof], label='Predict')
    plt.title('Training_Hysteresis')
    plt.legend()
    plt.savefig(os.path.join(folder_name, f'Training_Hysteresis.{exten}'),format=f'{exten}',  dpi=300)

# Prediction performance
X_pred = ag_pred[0:noPred]
y_pred_ref = u_pred[0:noPred]
yt_pred_ref = ut_pred[0:noPred]
ytt_pred_ref = utt_pred[0:noPred]
r_pred_ref = -X_pred
g_pred_ref = -ytt_pred_ref + r_pred_ref

eta, eta_t, eta_tt, eta_dot, g,r,r_dot = model.predict(X_pred, np.repeat(Phi_t0, len(X_pred), axis=0))
# r = model.predict_c(X_pred, np.repeat(Phi_t0, len(X_pred), axis=0))
y_pred = eta
yt_pred = eta_t
ytt_pred = eta_tt
g_pred = -eta_tt + r

dof = 0
for ii in range(len(y_pred_ref)):
    plt.figure()
    plt.plot(y_pred_ref[ii, :, dof], label='True')
    plt.plot(y_pred[ii, :, dof], label='Predict')
    plt.title('Prediction_u')
    plt.legend()
    plt.savefig(os.path.join(folder_name, f'TPrediction_u.{exten}'),format=f'{exten}',  dpi=300)

for ii in range(len(y_pred_ref)):
    plt.figure()
    plt.plot(yt_pred_ref[ii, :, dof], label='True')
    plt.plot(yt_pred[ii, :, dof], label='Predict')
    plt.title('Prediction_u_t')
    plt.legend()
    plt.savefig(os.path.join(folder_name, f'Prediction_u_t.{exten}'),format=f'{exten}',  dpi=300)

for ii in range(len(y_pred_ref)):
    plt.figure()
    plt.plot(ytt_pred_ref[ii, :, dof], label='True')
    plt.plot(ytt_pred[ii, :, dof], label='Predict')
    plt.title('Prediction_u_tt')
    plt.legend()
    plt.savefig(os.path.join(folder_name, f'Prediction_u_tt.{exten}'),format=f'{exten}',  dpi=300)

for ii in range(len(y_pred_ref)):
    plt.figure()
    plt.plot(g_pred_ref[ii, :, dof], label='True')
    plt.plot(g_pred[ii, :, dof], label='Predict')
    plt.title('Prediction_g')
    plt.legend()
    plt.savefig(os.path.join(folder_name, f'Prediction_g.{exten}'),format=f'{exten}',  dpi=300)

for ii in range(len(y_pred_ref)):
    plt.figure()
    plt.plot(y_pred_ref[ii, :, dof], g_pred_ref[ii, :, dof], label='True')
    plt.plot(y_pred[ii, :, dof], g_pred[ii, :, dof], label='Predict')
    plt.title('Prediction_Hysteresis')
    plt.legend()
    plt.savefig(os.path.join(folder_name, f'Prediction_Hysteresis.{exten}'),format=f'{exten}',  dpi=300)

model.save_weights(f'{folder_name}/DeepPhyLSTM3.weights.h5')
