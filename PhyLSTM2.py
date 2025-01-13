"""
Created on Fri Dec 13 21:51:22 2024
@author: Ruiyang Zhang (Original)
Reimplemented for TensorFlow 2.0
@author: Harrish Joseph
"""
import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time
import os, sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from random import shuffle

plt.close('all')
folder_name = time.strftime(f'PhyLSTM2')#[0][0]
os.makedirs(f'results/{folder_name}', exist_ok=True)  
exten = 'jpg'
class DeepPhyLSTM(tf.keras.Model):
    def __init__(self, eta, eta_t, g, ag, ag_c, r, Phi_t):
        super(DeepPhyLSTM, self).__init__()
        # Data
        self.eta = eta
        self.eta_t = eta_t
        self.g = g
        self.ag = ag
        self.r = r
        self.ag_c = ag_c
        self.Phi_t = Phi_t

        tf.keras.backend.set_floatx('float32')
        self.learning_rate = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        self.LSTM_model = tf.keras.Sequential([
            LSTM(100, return_sequences=True, input_shape=(None, 1), stateful=False),
            Activation('relu'),
            LSTM(100, return_sequences=True, stateful=False),
            Activation('relu'),
            LSTM(100, return_sequences=True, stateful=False),
            Activation('relu'),
            Dense(100),
            Dense(3 * self.eta.shape[2])
        ])
        self.LSTM_model_f = tf.keras.Sequential([
            LSTM(100, return_sequences=True, input_shape=(None, 3*self.eta.shape[2]), stateful=False),
            Activation('relu'),
            LSTM(100, return_sequences=True, stateful=False),
            Activation('relu'),
            LSTM(100, return_sequences=True, stateful=False),
            Activation('relu'),
            Dense(100),
            Dense(self.eta.shape[2])
        ])
    def call(self, inputs):
        ag, Phi_tf = inputs
        # Forward pass through LSTM_model
        output = self.LSTM_model(ag)
        eta = output[:, :, 0:self.eta.shape[2]]
        eta_dot = output[:, :, self.eta.shape[2]:2 * self.eta.shape[2]]
        g = output[:, :, 2 * self.eta.shape[2]:]
    
        # Perform matrix multiplications
        Phi_tf = Phi_tf[:, :eta.shape[1], :eta.shape[1]]
        eta_t = tf.matmul(tf.cast(Phi_tf, tf.float32), tf.cast(eta, tf.float32))
        eta_tt = tf.matmul(tf.cast(Phi_tf, tf.float32), tf.cast(eta_dot, tf.float32))
        return eta, eta_t, eta_tt, eta_dot, g

    def net_f(self, ag, Phi_tf):
        # eta, eta_t, eta_tt, eta_dot, g = self.net_structure(ag, Phi_tf)
        eta, eta_t, eta_tt, eta_dot, g = self((ag, Phi_tf))

        g_pred = self.LSTM_model_f(tf.concat([eta, eta_dot, g], 2))
        lift_pred = eta_tt + g_pred
        return eta_t, eta_dot, lift_pred

    @tf.function
    def train_step(self, eta_tf, eta_t_tf, g_tf, ag_tf, ag_c_tf, lift, Phi_tf):
        with tf.GradientTape() as tape:
            # eta_pred, eta_t_pred, eta_tt_pred, eta_dot_pred, g_pred = self.net_structure(ag_tf, Phi_tf)
            eta_pred, eta_t_pred, eta_tt_pred, eta_dot_pred, g_pred = self((ag_tf, Phi_tf))
            eta_t_pred_c, eta_dot_pred_c, lift_pred = self.net_f(ag_c_tf, Phi_tf)

            # Compute losses
            loss_u = tf.reduce_mean(tf.square(tf.cast(eta_tf, tf.float32) - eta_pred))
            loss_udot = tf.reduce_mean(tf.square(tf.cast(eta_t_tf, tf.float32) - eta_dot_pred))
            
            loss_g = tf.reduce_mean(tf.square(tf.cast(g_tf, tf.float32) - g_pred))
            
            loss_ut_c = tf.reduce_mean(tf.square(tf.cast(eta_t_pred_c, tf.float32) - eta_dot_pred_c))
            loss_e = tf.reduce_mean(tf.square(tf.cast(lift, tf.float32)- lift_pred))
            # Total loss
            total_loss = loss_u + loss_udot + loss_ut_c + loss_e + loss_g

        # Compute gradients
        trainable_vars = (
            self.LSTM_model.trainable_variables + 
            self.LSTM_model_f.trainable_variables
        )
        gradients = tape.gradient(total_loss, trainable_vars)
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return total_loss, loss_u, loss_udot, loss_g, loss_ut_c, loss_e

    def train(self, num_epochs, learning_rate, bfgs=0):
        # Prepare training and validation data
        Ind = list(range(self.ag.shape[0]))
        shuffle(Ind)
        ratio_split = 0.8
        Ind_tr = Ind[:round(ratio_split * self.ag.shape[0])]
        Ind_val = Ind[round(ratio_split * self.ag.shape[0]):]

        self.ag_tr = self.ag[Ind_tr]
        self.eta_tr = self.eta[Ind_tr]
        self.eta_t_tr = self.eta_t[Ind_tr]
        self.g_tr = self.g[Ind_tr]
        self.ag_val = self.ag[Ind_val]
        self.eta_val = self.eta[Ind_val]
        self.eta_t_val = self.eta_t[Ind_val]
        self.g_val = self.g[Ind_val]

        # Lists to track losses
        Loss_u, Loss_udot, Loss_g, Loss_ut_c, Loss_e1, Loss, Loss_val = [], [], [], [], [], [], []
        best_loss = float('inf')

        for epoch in range(num_epochs):
            start_time = time.time()

            # Training step
            loss_value, loss_u_val, loss_udot_val, loss_g_val, loss_ut_c_val, loss_e_val = self.train_step(
                self.eta_tr, self.eta_t_tr, self.g_tr, self.ag_tr, self.ag_c[Ind_tr], self.r[Ind_tr], self.Phi_t[Ind_tr])
            # Validation loss
            val_loss = self.train_step(self.eta_val, self.eta_t_val, self.g_val, self.ag_val, self.r[Ind_val], self.ag_c[Ind_val], self.Phi_t[Ind_val])[0]
        
            # Store losses
            Loss_u.append(loss_u_val.numpy())
            Loss_udot.append(loss_udot_val.numpy())
            Loss_g.append(loss_g_val.numpy())
            Loss_ut_c.append(loss_ut_c_val.numpy())
            Loss_e1.append(loss_e_val.numpy())
            Loss.append(loss_value.numpy())
            Loss_val.append(val_loss.numpy())

            # Update best loss
            if val_loss < best_loss:
                best_loss = val_loss

            elapsed = time.time() - start_time
            print(f'Epoch: {epoch}, Loss: {loss_value.numpy():.3e}, Loss_val: {val_loss.numpy():.3e}, '
                  f'Best_loss: {best_loss:.3e}, Time: {elapsed:.2f}, Learning Rate: {learning_rate:.3e}')

        return Loss_u, Loss_udot, Loss_g, Loss_ut_c, Loss_e1, Loss, Loss_val, best_loss

    def predict(self, ag_star, Phi_star):
        eta_star, eta_t_star, eta_tt_star, eta_dot_star, g_star = self((ag_star, Phi_star))
        _, _, r_star = self.net_f(ag_star, Phi_star)
        return (eta_star.numpy(),eta_t_star.numpy(),eta_tt_star.numpy(),eta_dot_star.numpy(),g_star.numpy(),r_star.numpy())

# Set GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# Load data
mat = scipy.io.loadmat('data_boucwen.mat')

# Data preprocessing
ag_data = mat['input_tf']  # ag, ad, av
u_data = mat['target_X_tf']
ut_data = mat['target_Xd_tf']
utt_data = mat['target_Xdd_tf']

ag_all = ag_data.reshape([ag_data.shape[0], ag_data.shape[1], 1])
u_all = u_data.reshape([u_data.shape[0], u_data.shape[1], 1])
u_t_all = ut_data.reshape([ut_data.shape[0], ut_data.shape[1], 1])
u_tt_all = utt_data.reshape([utt_data.shape[0], utt_data.shape[1], 1])

t = mat['time']
dt = t[0, 1] - t[0, 0]

# Finite difference matrix construction
n = u_data.shape[1]
phi1 = np.concatenate([np.array([-3 / 2, 2, -1 / 2]), np.zeros([n - 3, ])])
temp1 = np.concatenate([-1 / 2 * np.identity(n - 2), np.zeros([n - 2, 2])], axis=1)
temp2 = np.concatenate([np.zeros([n - 2, 2]), 1 / 2 * np.identity(n - 2)], axis=1)
phi2 = temp1 + temp2
phi3 = np.concatenate([np.zeros([n - 3, ]), np.array([1 / 2, -2, 3 / 2])])
Phi_t0 = 1 / dt * np.concatenate(
        [np.reshape(phi1, [1, phi1.shape[0]]), phi2, np.reshape(phi3, [1, phi3.shape[0]])], axis=0)
Phi_t0 = np.reshape(Phi_t0, [1, n, n])

# Prepare training and prediction data
ag = ag_all[0:10]
eta = u_all[0:10]
eta_t = u_t_all[0:10]
eta_tt = u_tt_all[0:10]
ag_c = ag_all[0:10]
r = -ag_c
g = -eta_tt - ag
N_train = eta.shape[0]

# Prepare Phi_t for multiple samples
Phi_t = np.repeat(Phi_t0, ag_c.shape[0], axis=0)

# Training
model = DeepPhyLSTM(eta, eta_t, g, ag, ag_c, r, Phi_t)
model((ag[0:1],Phi_t[0:1]))
# model.load_weights('PhyLSTM212171403It5000/DeepPhyLSTM.h5')

# Train the model
Loss_u, Loss_udot, Loss_g, Loss_ut_c, Loss_e1, Loss, Loss_val, best_loss = model.train(num_epochs=100, learning_rate=1e-3, bfgs=0)

# Plot training and validation loss
plt.figure()
plt.plot(np.log(Loss), label='training loss')
plt.plot(np.log(Loss_val), label='validation loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.show()

ag_data =  mat['input_pred_tf']
u_data = mat['target_pred_X_tf']
ut_data = mat['target_pred_Xd_tf']
utt_data =  mat['target_pred_Xdd_tf']
ag_data = ag_data.reshape([ag_data.shape[0], ag_data.shape[1], 1])
u_data = u_data.reshape([u_data.shape[0], u_data.shape[1], 1])
ut_data = ut_data.reshape([ut_data.shape[0], ut_data.shape[1], 1])
utt_data = utt_data.reshape([utt_data.shape[0], utt_data.shape[1], 1])

dof = 0
# Plotting functions for training and prediction results
def plot_results(y_ref, y_pred, title):
    for ii in range(1):
        plt.figure()
        plt.plot(y_ref[ii, :, dof], label='True')
        plt.plot(y_pred[ii, :, dof], label='Predict')
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(folder_name, f'{title}.{exten}'),format=f'{exten}',  dpi=300)


# Training performance evaluation
n=2
X_train        = ag_all[0:n]
y_train_ref    = u_all[0:n]
yt_train_ref   = u_t_all[0:n]
ytt_train_ref  = u_tt_all[0:n]
lift_train_ref = -X_train
g_train_ref    = -ytt_train_ref + lift_train_ref

# Predict training data
eta, eta_t, eta_tt, eta_dot, g, lift_train_pred = model.predict(X_train, np.repeat(Phi_t0, len(X_train), axis=0))
# r = model.predict_c(X_train, np.repeat(Phi_t0, len(X_train), axis=0))

y_train_pred = eta
yt_train_pred = eta_t
ytt_train_pred = eta_tt
g_train_pred = -eta_tt + lift_train_pred

# Plot training results
plot_results(y_train_ref, y_train_pred, 'Training_u')
plot_results(yt_train_ref, yt_train_pred, 'Training_u_t')
plot_results(ytt_train_ref, ytt_train_pred, 'Training_u_tt')
plot_results(g_train_ref, g_train_pred, 'Training_g')

plt.figure(figsize=(12,10))
plt.plot(y_train_ref[0,:]-eta[0,:],label='diff_eta')
# plt.plot(yt_train_ref[0,:]-eta_t[0,:],label='diff_eta_t')
# plt.plot(ytt_train_ref[0,:]-eta_tt[0,:],label='diff_eta_tt')
plt.plot(yt_train_ref[0,:]-eta_dot[0,:],label='diff_eta_dot_t')
plt.legend()



# Prediction performance evaluation
X_pred       = ag_data[n:n+2]
y_pred_ref   = u_data[n:n+2]
yt_pred_ref  = ut_data[n:n+2]
ytt_pred_ref = utt_data[n:n+2]
r_pred_ref   = -X_pred
g_pred_ref   = -ytt_pred_ref + r_pred_ref
# Predict prediction data
# eta, eta_t, eta_tt, eta_dot, g = model.predict(X_pred, np.repeat(Phi_t0, len(X_pred), axis=0))
# eta, eta_t, eta_tt, eta_dot, g = model((X_pred, np.repeat(Phi_t0, len(X_pred), axis=0)))
eta, eta_t, eta_tt, eta_dot, g,lift_pred = model.predict(X_pred, np.repeat(Phi_t0, len(X_pred), axis=0))
y_pred = eta
yt_pred = eta_t
ytt_pred = eta_tt
g_pred = -eta_tt + lift_pred

# Prediction performance evaluation
X_predh       = ag_data[n:n+2,0:1000,:]
y_pred_refh   = u_data[n:n+2,0:1000,:]
yt_pred_refh  = ut_data[n:n+2,0:1000,:]
ytt_pred_refh = utt_data[n:n+2,0:1000,:]
r_pred_refh   = -X_pred
g_pred_refh   = -ytt_pred_ref + r_pred_ref
# Predict prediction data
# eta, eta_t, eta_tt, eta_dot, g = model((X_predh, np.repeat(Phi_t0, len(X_predh), axis=0)))
eta, eta_t, eta_tt, eta_dot, g,r = model.predict(X_predh, np.repeat(Phi_t0, len(X_predh), axis=0))
y_predh = eta
yt_predh = eta_t
ytt_predh = eta_tt
g_predh = -eta_tt + r

# Visualization and plotting logic (similar to original code)
dof = 0
# Plot prediction results
plot_results(y_pred_ref, y_pred, 'Prediction_u')
plot_results(yt_pred_ref, yt_pred, 'Prediction_u_t')
plot_results(ytt_pred_ref, ytt_pred, 'Prediction_u_tt')
plot_results(g_pred_ref, g_pred, 'Prediction_g')

# Plot prediction results for 1000 timesteps
plot_results(y_pred_refh, y_predh, 'Prediction_uh')
plot_results(yt_pred_refh, yt_predh, 'Prediction_u_th')
plot_results(ytt_pred_refh, ytt_predh, 'Prediction_u_tth')
plot_results(g_pred_refh, g_predh, 'Prediction_gh')
# Hysteresis plots
for ii in range(y_train_ref.shape[0]):
    plt.figure()
    plt.plot(y_train_ref[ii, :, dof], g_train_ref[ii, :, dof], label='True')
    plt.plot(y_train_pred[ii, :, dof], g_train_pred[ii, :, dof], label='Predict')
    plt.title('Training_Hysteresis')
    plt.legend()

for ii in range(y_pred_ref.shape[0]):
    plt.figure()
    plt.plot(y_pred_ref[ii, :, dof], g_pred_ref[ii, :, dof], label='True')
    plt.plot(y_pred[ii, :, dof], g_pred[ii, :, dof], label='Predict')
    plt.title('Prediction_Hysteresis')
    plt.legend()

model.save_weights(f'{folder_name}/DeepPhyLSTM.weights.h5')

