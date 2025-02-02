"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""

import os

import numpy as np
# Necessary Packages
import tensorflow as tf
import tensorflow.keras as keras
from utils import extract_time, rnn_cell, random_generator, batch_generator

from tsai.all import *
import pickle
import sklearn.metrics as skm
import json
from collections import Counter

def styletimegan(ori_data,label, parameters,nb_classes,style_training_data,training_label,style_training=True,only_style_training=False,save_name=None,from_join_training=False):
    """TimeGAN function.

    Use original data as training set to generater synthetic data (time-series)

    Args:
      - ori_data: original time-series data
      - parameters: TimeGAN network parameters

    Returns:
      - generated_data: generated time-series data
    """
    # Initialization on the Graph
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    tf.reset_default_graph()
    # tf.enable_eager_execution()
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    if style_training:
        style_no, style_seq_len, style_dim = np.asarray(style_training_data).shape

    # Maximum sequence length and each sequence length
    ori_time, max_seq_len = extract_time(ori_data)
    if style_training:
        style_ori_time, style_max_seq_len = extract_time(style_training_data)


    if save_name:
        save_name+='_'
    def MinMaxScaler(data):
        """Min-Max Normalizer.

        Args:
          - data: raw data

        Returns:
          - norm_data: normalized data
          - min_val: minimum values (for renormalization)
          - max_val: maximum values (for renormalization)
        """
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val

        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)

        return norm_data, min_val, max_val

    # Normalization
    ori_data, min_val, max_val = MinMaxScaler(ori_data)
    if style_training:
        style_training_data, _, _ = MinMaxScaler(style_training_data)

    ## Build a RNN networks

    # Network Parameters
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    z_dim = dim
    gamma = 1

    # Input place holders
    X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x")
    L = tf.placeholder(tf.float32, [None, nb_classes], name="myinput_L")
    Z = tf.placeholder(tf.float32, [None, max_seq_len, z_dim], name="myinput_z")
    T = tf.placeholder(tf.int32, [None], name="myinput_t")
    if style_training:
        style_T = tf.placeholder(tf.int32, [None], name="myinput_style_t")
        style_X = tf.placeholder(tf.float32, [None, style_max_seq_len, style_dim], name="myinput_style_x")
        style_L = tf.placeholder(tf.float32, [None,nb_classes], name="myinput_style_L")

    def embedder(X, T):
        """Embedding network between original feature space to latent space.

        Args:
          - X: input time-series features
          - T: input time information

        Returns:
          - H: embeddings
        """
        with tf.variable_scope("embedder", reuse=tf.AUTO_REUSE):
            e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length=T)
            H = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
        return H

    def recovery(H, T):
        """Recovery network from latent space to original space.

        Args:
          - H: latent representation
          - T: input time information

        Returns:
          - X_tilde: recovered data
        """
        with tf.variable_scope("recovery", reuse=tf.AUTO_REUSE):
            r_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
            r_outputs, r_last_states = tf.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length=T)
            X_tilde = tf.contrib.layers.fully_connected(r_outputs, dim, activation_fn=tf.nn.sigmoid)
        return X_tilde

    def generator(Z, T):
        """Generator function: Generate time-series data in latent space.

        Args:
          - Z: random variables
          - T: input time information

        Returns:
          - E: generated embedding
        """
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length=T)
            E = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
        return E

    def supervisor(H, T):
        """Generate next sequence using the previous sequence.

        Args:
          - H: latent representation
          - T: input time information

        Returns:
          - S: generated sequence based on the latent representations generated by the generator
        """
        with tf.variable_scope("supervisor", reuse=tf.AUTO_REUSE):
            e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers - 1)])
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length=T)
            S = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
        return S

    def discriminator(H, T):
        """Discriminate the original and synthetic time-series data.

        Args:
          - H: latent representation
          - T: input time information

        Returns:
          - Y_hat: classification results between original and synthetic time-series
        """
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            d_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
            d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length=T)
            Y_hat = tf.contrib.layers.fully_connected(d_outputs, 1, activation_fn=None)
        return Y_hat

    def style_discriminator(input_tensor,nb_classes=3):
        # discrimate the regime of synthetic time-series data with pre-train network
        
        
    def _inception_module(input_tensor, stride=1, activation='linear'):

    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                              padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                             strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception))

    max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                 padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = keras.layers.Concatenate(axis=2)(conv_list)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    return x

    def _shortcut_layer( input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x
            return output_layer

    def build_model(input_shape, nb_classes):
    input_layer = keras.layers.Input(input_shape)

    x = input_layer
    input_res = input_layer

    for d in range(self.depth):

        x = self._inception_module(x)

        if self.use_residual and d % 3 == 2:
            x = self._shortcut_layer(input_res, x)
            input_res = x

    gap_layer = keras.layers.GlobalAveragePooling1D()(x)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                  min_lr=0.0001)

    file_path = self.output_directory + 'best_model.hdf5'

    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                       save_best_only=True)

    self.callbacks = [reduce_lr, model_checkpoint]

    return model

    # Embedder & Recovery

    H = embedder(X, T)
    X_tilde = recovery(H, T)

    # Generator
    E_hat = generator(Z, T)
    H_hat = supervisor(E_hat, T)
    H_hat_supervise = supervisor(H, T)

    # Synthetic data
    X_hat = recovery(H_hat, T)
    generated_data_for_style_evaluation = list()
    for i in range(no):
        temp = X_hat[i, :ori_time[i], :]
        generated_data_for_style_evaluation.append(temp)

    # Discriminator
    Y_fake = discriminator(H_hat, T)
    Y_real = discriminator(H, T)
    Y_fake_e = discriminator(E_hat, T)

    # Style Discriminators

    #TODO: pretrained InceptionTime for style classification
    if style_training:
        L_x_style_training=style_discriminator(style_X)
        print('L_x_style_training',L_x_style_training.get_shape())
    L_x_style= style_discriminator(X_hat)




    #Stylized fact constrain

    #TODO: auto-correlation loss

    #TODO: leveraging effects

    # Variables
    e_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
    r_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]
    g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
    s_vars = [v for v in tf.trainable_variables() if v.name.startswith('supervisor')]
    d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    style_vars=[v for v in tf.trainable_variables() if v.name.startswith('style_discriminator')]

    #Style Discriminator loss
    if style_training:
        print('style_L',style_L.get_shape())
        Style_loss_training =tf.losses.sigmoid_cross_entropy(tf.convert_to_tensor(style_L), L_x_style_training)
    Style_loss=tf.losses.sigmoid_cross_entropy(tf.convert_to_tensor(L), L_x_style)

    # Discriminator loss
    D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
    D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
    D_loss_fake_e = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
    D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

    # Generator loss
    # 1. Adversarial loss
    G_loss_U = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
    G_loss_U_e = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)

    # 2. Supervised loss
    G_loss_S = tf.losses.mean_squared_error(H[:, 1:, :], H_hat_supervise[:, :-1, :])

    # 3. Two Momments
    G_loss_V1 = tf.reduce_mean(
        tf.abs(tf.sqrt(tf.nn.moments(X_hat, [0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X, [0])[1] + 1e-6)))
    G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat, [0])[0]) - (tf.nn.moments(X, [0])[0])))

    G_loss_V = G_loss_V1 + G_loss_V2

    # 4. Summation
    G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V + Style_loss

    # Embedder network loss
    E_loss_T0 = tf.losses.mean_squared_error(X, X_tilde)
    E_loss0 = 10 * tf.sqrt(E_loss_T0)
    E_loss = E_loss0 + 0.1 * G_loss_S + Style_loss

    # optimizer
    E0_solver = tf.train.AdamOptimizer().minimize(E_loss0, var_list=e_vars + r_vars)
    E_solver = tf.train.AdamOptimizer().minimize(E_loss, var_list=e_vars + r_vars)
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=d_vars)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=g_vars + s_vars)
    GS_solver = tf.train.AdamOptimizer().minimize(G_loss_S, var_list=g_vars + s_vars)
    if style_training:
        Style_D_solver  =tf.train.AdamOptimizer().minimize(Style_loss_training)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter('./log/style/', sess.graph)
    ## TimeGAN training
    #0. Style Discriminator training
    saver = tf.train.Saver()
    if style_training:
        print('start style training')
        model = build_model(input_shape, nb_classes)
        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                      verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
    if only_style_training:
        print("style training done, end all process")
        return
    # 1. Embedding network training
    if not from_join_training:
        print('Start Embedding Network Training')
        for itt in range(iterations):
            # Set mini-batch
            X_mb, T_mb,L_mb = batch_generator(ori_data, ori_time, batch_size,label)
            # Train embedder
            _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb,L:L_mb})
            # Checkpoint
            if itt % 1000 == 0:
                print('step: ' + str(itt) + '/' + str(iterations) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss), 4)))

        print('Finish Embedding Network Training')

        # 2. Training only with supervised loss
        print('Start Training with Supervised Loss Only')

        for itt in range(iterations):
            # Set mini-batch
            X_mb, T_mb,L_mb = batch_generator(ori_data, ori_time, batch_size,label)
            # Random vector generation
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            # style modulation
            #TODO: modulate style to Z


            # Train generator
            _, step_g_loss_s = sess.run([GS_solver, G_loss_S], feed_dict={Z: Z_mb, X: X_mb, T: T_mb,L:L_mb})
            # Checkpoint
            if itt % 1000 == 0:
                print('step: ' + str(itt) + '/' + str(iterations) + ', s_loss: ' + str(np.round(np.sqrt(step_g_loss_s), 4)))
        saver.save(sess, './model/style/before_join_training_model_' + str(save_name) + '_add_style.ckpt')
        print('Finish Training with Supervised Loss Only')
    else:
        saver.restore(sess, './model/style/before_join_training_model_' + str(save_name) + '_add_style.ckpt')
        print('restore model before joint training')


    # 3. Joint Training
    print('Start Joint Training')

    for itt in range(iterations):
        # Generator training (twice more than discriminator training)
        for kk in range(2):
            # Set mini-batch
            X_mb, T_mb,L_mb = batch_generator(ori_data, ori_time, batch_size,label)
            # Random vector generation
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            # Train generator
            _, step_g_loss_u, step_g_loss_s, step_g_loss_v,step_g_loss_style = sess.run([G_solver, G_loss_U, G_loss_S, G_loss_V,Style_loss],
                                                                      feed_dict={Z: Z_mb, X: X_mb, T: T_mb,L:L_mb})
            # Train embedder
            _, step_e_loss_t0 = sess.run([E_solver, E_loss_T0], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})

            # Discriminator training
        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        # Random vector generation
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        # Check discriminator loss before updating
        check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        # Train discriminator (only when the discriminator does not work well)
        if (check_d_loss > 0.15):
            _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})

        # Print multiple checkpoints
        if itt % 100 == 0:
            summary_writer.add_summary(step_d_loss, global_step=itt)
            summary_writer.add_summary(step_g_loss_u, global_step=itt)
            summary_writer.add_summary(step_g_loss_s, global_step=itt)
            summary_writer.add_summary(step_g_loss_v, global_step=itt)
            summary_writer.add_summary(step_e_loss_t0, global_step=itt)
            summary_writer.add_summary(step_g_loss_style, global_step=itt)
        if itt % 1000 == 0:
            print('step: ' + str(itt) + '/' + str(iterations) +
                  ', d_loss: ' + str(np.round(step_d_loss, 4)) +
                  ', g_loss_u: ' + str(np.round(step_g_loss_u, 4)) +
                  ', g_loss_s: ' + str(np.round(np.sqrt(step_g_loss_s), 4)) +
                  ', g_loss_v: ' + str(np.round(step_g_loss_v, 4)) +
                  ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_t0), 4))+
                  ', g_loss_style: ' + str(np.round(step_g_loss_style, 4))
                  )
            # Now, save the graph

            saver.save(sess, './model/style/join_training_model_'+str(save_name)+'_add_style.ckpt', global_step=itt)
    print('Finish Joint Training')

    ## Synthetic data generation
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time})

    generated_data = list()

    for i in range(no):
        temp = generated_data_curr[i, :ori_time[i], :]
        generated_data.append(temp)

    # Renormalization
    generated_data = generated_data * max_val
    generated_data = generated_data + min_val
    return generated_data
