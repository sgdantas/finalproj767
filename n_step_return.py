#IN [1]
#-*- coding: utf-8 -*-
import tensorflow as tf
distr = tf.contrib.distributions

import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

import random
from utils import embed_seq, encode_seq, full_glimpse, pointer, pointer_critic
from data_generator import DataGenerator

from copy import copy
import graph 

###############################################################
#PARAMETERS CONTROLLING THE FUNCTONS IMPLEMENTED IN THIS CODE
###############################################################
init_baseline = 0.0

n_td = 40 #Variable controlling the n-step return

use_elig_trace = True #Set to True if want to use elig_trace, False otherwise
lamda_value = 0.6
use_xp_replay = True #Set to True if want to use experience replay, False otherwise
n_replay = 10

n_reps = 3 # Number of times the training will be performed in order to generate the plots with av and std values






#DATA GENERATOR
#IN [2]
dataset = DataGenerator() # Create Data Generator

input_batch = dataset.test_batch(batch_size=128, max_length=50, dimension=2, seed=123) # Generate some data
dataset.visualize_2D_trip(input_batch[0]) # 2D plot for coord batch


#CONFIG
#IN [3]
import argparse

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []

def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg

def str2bool(v):
  return v.lower() in ('true', '1')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batch_size', type=int, default=20, help='batch size')
data_arg.add_argument('--max_length', type=int, default=50, help='number of cities') ##### #####
data_arg.add_argument('--dimension', type=int, default=2, help='city dimension')

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--input_embed', type=int, default=128, help='actor critic input embedding')
net_arg.add_argument('--num_neurons', type=int, default=512, help='encoder inner layer neurons')
net_arg.add_argument('--num_stacks', type=int, default=3, help='encoder num stacks')
net_arg.add_argument('--num_heads', type=int, default=16, help='encoder num heads')
net_arg.add_argument('--query_dim', type=int, default=360, help='decoder query space dimension')
net_arg.add_argument('--num_units', type=int, default=256, help='decoder and critic attention product space')
net_arg.add_argument('--num_neurons_critic', type=int, default=256, help='critic n-1 layer')

# Train / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--nb_steps', type=int, default=1000, help='nb steps')
train_arg.add_argument('--init_B', type=float, default=init_baseline, help='critic init baseline')
train_arg.add_argument('--lr_start', type=float, default=0.001, help='actor learning rate')
train_arg.add_argument('--lr_decay_step', type=int, default=5000, help='lr1 decay step')
train_arg.add_argument('--lr_decay_rate', type=float, default=0.96, help='lr1 decay rate')
train_arg.add_argument('--temperature', type=float, default=1.0, help='pointer initial temperature')
train_arg.add_argument('--C', type=float, default=10.0, help='pointer tanh clipping')
train_arg.add_argument('--is_training', type=str2bool, default=True, help='switch to inference mode when model is trained') 

def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed


#IN [4]
config, _ = get_config()
dir_ = str(config.dimension)+'D_'+'TSP'+str(config.max_length) +'_b'+str(config.batch_size)+'_e'+str(config.input_embed)+'_n'+str(config.num_neurons)+'_s'+str(config.num_stacks)+'_h'+str(config.num_heads)+ '_q'+str(config.query_dim) +'_u'+str(config.num_units)+'_c'+str(config.num_neurons_critic)+ '_lr'+str(config.lr_start)+'_d'+str(config.lr_decay_step)+'_'+str(config.lr_decay_rate)+ '_T'+str(config.temperature)+ '_steps'+str(config.nb_steps)+'_i'+str(config.init_B) 
print(dir_)


#MODEL
       
   
        
class Actor(object):
    
    def __init__(self):
        
        # Data config
        self.batch_size = config.batch_size # batch size
        self.max_length = config.max_length # input sequence length (number of cities)
        self.dimension = config.dimension # dimension of a city (coordinates)
        
        # Network config
        self.input_embed = config.input_embed # dimension of embedding space
        self.num_neurons = config.num_neurons # dimension of hidden states (encoder)
        self.num_stacks = config.num_stacks # encoder num stacks
        self.num_heads = config.num_heads # encoder num heads
        self.query_dim = config.query_dim # decoder query space dimension
        self.num_units = config.num_units # dimension of attention product space (decoder and critic)
        self.num_neurons_critic = config.num_neurons_critic # critic n-1 layer num neurons
        self.initializer = tf.contrib.layers.xavier_initializer() # variables initializer
        
        # Training config (actor and critic)
        self.global_step = tf.Variable(0, trainable=False, name="global_step") # actor global step
        self.global_step2 = tf.Variable(0, trainable=False, name="global_step2") # critic global step
        self.init_B = config.init_B # critic initial baseline
        self.lr_start = config.lr_start # initial learning rate
        self.lr_decay_step = config.lr_decay_step # learning rate decay step
        self.lr_decay_rate = config.lr_decay_rate # learning rate decay rate
        self.is_training = config.is_training # swith to False if test mode

        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.input_ = tf.placeholder(tf.float32, [None, self.max_length, self.dimension], name="input_coordinates")
        
        ############################################################################################################################################################################################## 
        ############################################################################################################################################################################################## 
        ##############################################################################################################################################################################################
        actor_embedding = embed_seq(input_seq=self.input_, from_=self.dimension, to_= self.input_embed, is_training=self.is_training, BN=True, initializer=self.initializer)
        actor_encoding = encode_seq(input_seq=actor_embedding, input_dim=self.input_embed, num_stacks=self.num_stacks, num_heads=self.num_heads, num_neurons=self.num_neurons, is_training=self.is_training)

        if self.is_training == False:
            actor_encoding = tf.tile(actor_encoding,[self.batch_size,1,1])
        n_hidden = actor_encoding.get_shape().as_list()[2] # input_embed
        idx_list_previous, log_probs_previous, entropies_previous = [], [], [] # tours index, log_probs, entropies
        mask_previous = tf.zeros((self.batch_size, self.max_length))
        query1_previous = tf.zeros((self.batch_size, n_hidden)) # initial state
        query2_previous = tf.zeros((self.batch_size, n_hidden)) # previous state
        query3_previous = tf.zeros((self.batch_size, n_hidden)) # previous previous state
        ############################################################################################################################################################################################## 
        ############################################################################################################################################################################################## 
        ############################################################################################################################################################################################## 
        
        """
        Build a graph looping untill all TD steps are taken
        untill the end of the episode 
        """
        flag = False
        idx_ = []
        for i in range (self.max_length):
            print()
            print('Current on step ', i)
            with tf.variable_scope("actor"+str(i)):
                # Call enconde decode TD
                idx_list_previous, log_probs_previous, entropies_previous, mask_previous, query1_previous, query2_previous, query3_previous, idx_ = self.encode_decode_TD( n_td,idx_list_previous, log_probs_previous, entropies_previous, mask_previous, query1_previous, query2_previous, query3_previous, idx_ )            
            
            with tf.variable_scope("critic"+str(i)): self.build_critic()
            with tf.variable_scope("environment"+str(i)): self.build_reward()            
            
            if(len(idx_list_previous) == self.max_length):
                flag = True            
            with tf.variable_scope("optimizer"+str(i)): self.build_optim_reiforce(i)
            if(flag):
                print("end of the episode")
                break
            
        self.merged = tf.summary.merge_all()    
        
    ############################################################################################################################################################################################## 
    ##############################################################################################################################################################################################
    ##############################################################################################################################################################################################
    def _update_e_trace(self, trace, gradients, lamda_value, index, flag_grad2 = True ):
        """
        update the eligibility trace vector
        """
        trace_output = []
        
        if (flag_grad2): #If related to the weights of the Actor
            grad_init_index1 = (index - 1) * len(trace)
            grad_init_index2 = (index) * len(trace)           
            
            print("len trace = ",len(trace))
            print("len gradients = ", len(gradients))
            step_size = len(trace)

            for trace_index in range(len(trace)):                
                trace_output.append( (lamda_value*trace[trace_index][0] + gradients[trace_index][0], gradients[trace_index][1]) )

            for trace_index in range(len(gradients) - step_size):
                trace_output.append( (gradients[ step_size + trace_index][0], gradients[step_size + trace_index][1] ))
                
        else:   #If related to the weights of the Critic
            for trace_index in range(len(trace)):
                
                trace_output.append( (lamda_value*trace[trace_index][0]  + gradients[trace_index][0], gradients[trace_index][1]) )
        return trace_output

    def encode_decode_TD(self, n_step, idx_list_previous, log_probs_previous, entropies_previous, mask_previous, query1_previous, query2_previous, query3_previous, idx_ ):
        """
        Modified to accomodate tours 
        idx_list_previous is a list with partial choices made by the actor
        The actor takes n-steps
        The critic tells the rest of the path
        """

        actor_embedding = embed_seq(input_seq=self.input_, from_=self.dimension, to_= self.input_embed, is_training=self.is_training, BN=True, initializer=self.initializer)
        actor_encoding = encode_seq(input_seq=actor_embedding, input_dim=self.input_embed, num_stacks=self.num_stacks, num_heads=self.num_heads, num_neurons=self.num_neurons, is_training=self.is_training)
        
        if self.is_training == False:
            actor_encoding = tf.tile(actor_encoding,[self.batch_size,1,1])
        
        idx_list = copy(idx_list_previous)
        log_probs = copy(log_probs_previous)
        entropies = copy(entropies_previous)
        

        mask = copy(mask_previous)
        
        n_hidden = actor_encoding.get_shape().as_list()[2] # input_embed
        W_ref = tf.get_variable("W_ref",[1, n_hidden, self.num_units],initializer=self.initializer)
        W_q = tf.get_variable("W_q",[self.query_dim, self.num_units],initializer=self.initializer)
        v = tf.get_variable("v",[self.num_units],initializer=self.initializer)
        
        encoded_ref = tf.nn.conv1d(actor_encoding, W_ref, 1, "VALID") # actor_encoding is the ref for actions [Batch size, seq_length, n_hidden]
        
        query1 = copy( query1_previous)
        query2 = copy( query2_previous)
        query3 = copy( query3_previous)
        idx_copy = copy(idx_)
            
        W_1 =tf.get_variable("W_1",[n_hidden, self.query_dim],initializer=self.initializer) # update trajectory (state)
        W_2 =tf.get_variable("W_2",[n_hidden, self.query_dim],initializer=self.initializer)
        W_3 =tf.get_variable("W_3",[n_hidden, self.query_dim],initializer=self.initializer)
        
        
        """
        # sample from POINTER from the perspective of the Actor
        """
        for step in range(n_step + 1 ): 
            query = tf.nn.relu(tf.matmul(query1, W_1) + tf.matmul(query2, W_2) + tf.matmul(query3, W_3))
            logits = pointer(encoded_ref=encoded_ref, query=query, mask=mask, W_ref=W_ref, W_q=W_q, v=v, C=config.C, temperature=config.temperature)
            prob = distr.Categorical(logits) # logits = masked_scores
            idx = prob.sample()

            idx_list.append(idx) # tour index
            idx_list_previous.append(idx)
                
            log_probs.append(prob.log_prob(idx)) # log prob
            log_probs_previous.append(prob.log_prob(idx))
                
            entropies.append(prob.entropy()) # entropies
            entropies_previous.append(prob.entropy())
                
            mask = mask + tf.one_hot(idx, self.max_length) # mask
            mask_previous = mask_previous + tf.one_hot(idx, self.max_length)

            idx_copy =  tf.stack([tf.range(self.batch_size,dtype=tf.int32), idx],1) # idx with batch   
            idx_ = tf.stack([tf.range(self.batch_size,dtype=tf.int32), idx],1) # idx with batch   
            query3 = query2
            query2 = query1
            query1 = tf.gather_nd(actor_encoding, idx_) # update trajectory (state)
                
            query3_previous = query2_previous
            query2_previous = query1_previous
            query1_previous = tf.gather_nd(actor_encoding, idx_) # update trajectory (state)                

            if (len(idx_list) >= self.max_length): break #leave the loop if reach the end of the episode

        """
        # sample from POINTER from the perspective of the Critic
        make q_t vector = 0
        """
        while(len(idx_list) < self.max_length): 
                                                         
            logits = pointer_critic(encoded_ref=encoded_ref, mask=mask, W_ref=W_ref, v=v, C=config.C, temperature=config.temperature)
            prob = distr.Categorical(logits) # logits = masked_scores
            idx = prob.sample()

            idx_list.append(idx) # tour index
            log_probs.append(prob.log_prob(idx)) # log prob
            entropies.append(prob.entropy()) # entropies
            mask = mask + tf.one_hot(idx, self.max_length) # mask

            idx_copy = tf.stack([tf.range(self.batch_size,dtype=tf.int32), idx],1) # idx with batch   
            #idx_ = tf.stack([tf.range(self.batch_size,dtype=tf.int32), idx],1) # idx with batch   
            query3 = query2
            query2 = query1
            query1 = tf.gather_nd(actor_encoding, idx_copy) # update trajectory (state)
                                                        
        idx_list.append(idx_list[0]) # return to start
        self.tour =tf.stack(idx_list, axis=1) # permutations
        self.log_prob = tf.add_n(log_probs) # corresponding log-probability for backprop
        self.entropies = tf.add_n(entropies)
        tf.summary.scalar('log_prob_mean', tf.reduce_mean(self.log_prob))
        tf.summary.scalar('entropies_mean', tf.reduce_mean(self.entropies))
        
        return idx_list_previous, log_probs_previous, entropies_previous, mask_previous, query1_previous, query2_previous, query3_previous, idx_    #returns variables necessary for the next loop
    ############################################################################################################################################################################################## 
    ##############################################################################################################################################################################################
    ##############################################################################################################################################################################################              
        
    def build_reward(self): # reorder input % tour and return tour length (euclidean distance)
        self.permutations = tf.stack([tf.tile(tf.expand_dims(tf.range(self.batch_size,dtype=tf.int32),1),[1,self.max_length+1]),self.tour],2)
        if self.is_training==True:
            self.ordered_input_ = tf.gather_nd(self.input_,self.permutations)
        else:
            self.ordered_input_ = tf.gather_nd(tf.tile(self.input_,[self.batch_size,1,1]),self.permutations)
        self.ordered_input_ = tf.transpose(self.ordered_input_,[2,1,0]) # [features, seq length +1, batch_size]   Rq: +1 because end = start    
        
        ordered_x_ = self.ordered_input_[0] # ordered x, y coordinates [seq length +1, batch_size]
        ordered_y_ = self.ordered_input_[1] # ordered y coordinates [seq length +1, batch_size]          
        delta_x2 = tf.transpose(tf.square(ordered_x_[1:]-ordered_x_[:-1]),[1,0]) # [batch_size, seq length]        delta_x**2
        delta_y2 = tf.transpose(tf.square(ordered_y_[1:]-ordered_y_[:-1]),[1,0]) # [batch_size, seq length]        delta_y**2

        inter_city_distances = tf.sqrt(delta_x2+delta_y2) # sqrt(delta_x**2 + delta_y**2) this is the euclidean distance between each city: depot --> ... ---> depot      [batch_size, seq length]
        self.distances = tf.reduce_sum(inter_city_distances, axis=1) # [batch_size]
        self.reward = tf.cast(self.distances,tf.float32) # define reward from tour length  
        tf.summary.scalar('reward_mean', tf.reduce_mean(self.reward))
            
    def build_critic(self):
        critic_embedding = embed_seq(input_seq=self.input_, from_=self.dimension, to_= self.input_embed, is_training=self.is_training, BN=True, initializer=self.initializer)
        critic_encoding = encode_seq(input_seq=critic_embedding, input_dim=self.input_embed, num_stacks=self.num_stacks, num_heads=self.num_heads, num_neurons=self.num_neurons, is_training=self.is_training)
        frame = full_glimpse(ref=critic_encoding, from_=self.input_embed, to_=self.num_units, initializer=tf.contrib.layers.xavier_initializer()) # Glimpse on critic_encoding [Batch_size, input_embed]
        
        with tf.variable_scope("ffn"): #  2 dense layers for predictions
            h0 = tf.layers.dense(frame, self.num_neurons_critic, activation=tf.nn.relu, kernel_initializer=self.initializer)
            w1 = tf.get_variable("w1", [self.num_neurons_critic, 1], initializer=self.initializer)
            b1 = tf.Variable(self.init_B, name="b1")
            self.predictions = tf.squeeze(tf.matmul(h0, w1)+b1)
            tf.summary.scalar('predictions_mean', tf.reduce_mean(self.predictions))
                
    def build_optim_reiforce(self, i):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops): # Update moving_mean and moving_variance for BN
            
            with tf.name_scope('reinforce'+str(i)):
                lr1 = tf.train.natural_exp_decay(learning_rate=self.lr_start, global_step=self.global_step, decay_steps=self.lr_decay_step, decay_rate=self.lr_decay_rate, staircase=False, name="learning_rate1") # learning rate actor
                tf.summary.scalar('lr', lr1)
                opt1 = tf.train.AdamOptimizer(learning_rate=lr1) # Optimizer
                self.loss = tf.reduce_mean(tf.stop_gradient(self.reward-self.predictions)*self.log_prob, axis=0) # loss actor
                gvs1 = opt1.compute_gradients(self.loss) # gradients
                capped_gvs1 = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs1 if grad is not None] # L2 clip
                # IF ELIG TRACE
                if(use_elig_trace):
                    """
                    If first iteration create the elig trace vector
                    """
                    if(i==0): 
                        self.elig_trace1 = [(0*grad,var) for grad, var in capped_gvs1 if grad is not None ]                    
                        self.trn_op1 = opt1.apply_gradients(grads_and_vars=capped_gvs1, global_step=self.global_step) # minimize op actor                   
                    #"""
                    #If needed to update the elig trace vector
                    #"""
                    else:
                        self.elig_trace1 = self._update_e_trace(self.elig_trace1, capped_gvs1, lamda_value, i )                 
                        self.trn_op1 = opt1.apply_gradients(grads_and_vars=self.elig_trace1, global_step=self.global_step) # minimize op actor               
                # NO ELIG TRACE
                else:
                    self.trn_op1 = opt1.apply_gradients(grads_and_vars=capped_gvs1, global_step=self.global_step) # minimize op actor
            
            with tf.name_scope('state_value'+str(i)):
                lr2 = tf.train.natural_exp_decay(learning_rate=self.lr_start, global_step=self.global_step2, decay_steps=self.lr_decay_step, decay_rate=self.lr_decay_rate, staircase=False, name="learning_rate2") # learning rate critic
                opt2 = tf.train.AdamOptimizer(learning_rate=lr2) # Optimizer
                loss2 = tf.losses.mean_squared_error(self.reward, self.predictions) # loss critic
                gvs2 = opt2.compute_gradients(loss2) # gradients
                capped_gvs2 = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs2 if grad is not None] # L2 clip
                if(use_elig_trace):
                    if(i==0):
                        """
                        If first iteration create the elig trace vector
                        """
                        self.elig_trace2 = [(0*grad,var) for grad, var in capped_gvs2 if grad is not None ]
                        self.trn_op2 = opt2.apply_gradients(grads_and_vars=capped_gvs2, global_step=self.global_step) # minimize op actor
                    else:
                        self.elig_trace2 = self._update_e_trace(self.elig_trace2, capped_gvs2, lamda_value, i, flag_grad2 = False )
                        self.trn_op2 = opt2.apply_gradients(grads_and_vars=self.elig_trace2, global_step=self.global_step) # minimize op actor

                else:
                    self.trn_op2 = opt2.apply_gradients(grads_and_vars=capped_gvs2, global_step=self.global_step2) # minimize op critic


#IN [6]
tf.reset_default_graph()
actor = Actor() # Build graph


#IN [7]
variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name] # Save & restore all the variables.
saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)   


#IN [8]
with tf.Session() as sess: # start session
    sess.run(tf.global_variables_initializer()) # Run initialize op
    variables_names = [v.name for v in tf.trainable_variables() if 'Adam' not in v.name]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        #print("Variable: ", k, "Shape: ", v.shape) # print all variables
        pass


reward_per_step = np.zeros(shape = (config.nb_steps,n_reps)) #################################################################
baseline_per_step = np.zeros(config.nb_steps) #################################################################
iterations = np.arange(config.nb_steps) #################################################################

#TRAIN

np.random.seed(123) # reproducibility
tf.set_random_seed(123)

for rep in range(n_reps):
    print ('step - ', rep)

    with tf.Session() as sess: # start session
        sess.run(tf.global_variables_initializer()) # run initialize op
        writer = tf.summary.FileWriter('summary/'+dir_, sess.graph) # summary writer
        buffer_list = []  
        for i in tqdm(range(config.nb_steps)): # Forward pass & train step
            input_batch2 = dataset.train_batch(actor.batch_size, actor.max_length, actor.dimension)

            buffer_list.extend(input_batch2) # store samples in this list

            feed2 = {actor.input_: input_batch2} # get feed dict
            reward, predictions, summary, _, _ = sess.run([actor.reward, actor.predictions, actor.merged, actor.trn_op1, actor.trn_op2], feed_dict=feed2)

            reward_per_step[i][rep] = np.mean(reward) #################################################################
            baseline_per_step[i] += np.mean(predictions)/n_reps                                   
            
            if i % 50 == 0: 
                print('reward',np.mean(reward))
                print('predictions',np.mean(predictions))
                writer.add_summary(summary,i)          
            """
            Very simple implementation of Experience Replay
            where, at every n_replay iterations, 
            n_replay random replays are sampled from buffer_list
            """
            if(use_xp_replay):
                if(i>= n_replay):
                    for j in range(n_replay):
                        input_batch_replay = random.sample(buffer_list, k = actor.batch_size)
                        feed_replay = {actor.input_: input_batch_replay} # get feed dict
                        reward, predictions, summary, _, _ = sess.run([actor.reward, actor.predictions, actor.merged, actor.trn_op1, actor.trn_op2], feed_dict=feed2)
                    n_replay += n_replay
        
        save_path = "save/"+dir_
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saver.save(sess, save_path+"/actor.ckpt") # save the variables to disk
        print("Training COMPLETED! Model saved in file: %s" % save_path)


np.savetxt('iterations_1000it.out',iterations, delimiter= ' ')
np.savetxt('reward_per_step_v1_1000it_TD'+str(n_td)+'.out',reward_per_step, delimiter = ',')
graph.plot_graph2(iterations,
                   reward_per_step,
                   "average reward - TD"+str(n_td),
                   "iterations","av. tour lenght",
                   "tour_lenght_traing_TD"+str(n_td)+".png",plot_graph_= True)
graph.plot_graph3(iterations,
                   [reward_per_step,baseline_per_step],
                   ["average reward - TD"+str(n_td),"baseline"],
                   "iterations","av. tour lenght",
                   "tour_lenght_traing_"+str(n_td)+"Baseline_"+str(init_baseline)+".png",plot_graph_= True)



# TEST

config.is_training = False
#config.batch_size = 10 ##### #####
config.max_length = 50 ##### #####
config.temperature = 1.2 ##### #####

tf.reset_default_graph()
actor = Actor() # Build graph
r1,r2 = test_1()

##########################
config.is_training = True
#config.batch_size = 256 ##### #####
config.max_length = 50 ##### #####
config.temperature = 1.0 ##### #####


variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name] # Save & restore all the variables.
saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)   


with tf.Session() as sess:  # start session
    sess.run(tf.global_variables_initializer()) # Run initialize op
    
    save_path = "save/"+dir_
    saver.restore(sess, save_path+"/actor.ckpt") # Restore variables from disk.
    
    predictions_length, predictions_length_w2opt = [], []

    #######################################################
    nb_test = (10)
    rw1 = np.zeros(nb_test)
    rw2 = np.zeros(nb_test)

    #######################################################
    for i in tqdm(range(nb_test)): # test instance
        seed_ = 1+i
        input_batch = dataset.test_batch(1, actor.max_length, actor.dimension, seed=seed_, shuffle=False)
        feed = {actor.input_: input_batch} # Get feed dict
        tour, reward = sess.run([actor.tour, actor.reward], feed_dict=feed) # sample tours
        
        j = np.argmin(reward) # find best solution
        best_permutation = tour[j][:-1]
        predictions_length.append(reward[j])
        print('reward (before 2 opt)',reward[j])
        #dataset.visualize_2D_trip(input_batch[0][best_permutation])
        #dataset.visualize_sampling(tour)
        
        opt_tour, opt_length = dataset.loop2opt(input_batch[0][best_permutation])
        predictions_length_w2opt.append(opt_length)
        print('reward (with 2 opt)', opt_length)
        #dataset.visualize_2D_trip(opt_tour)
        rw1[i] = np.mean(reward)
        rw2[i] = np.opt_length

        
    predictions_length = np.asarray(predictions_length) # average tour length
    predictions_length_w2opt = np.asarray(predictions_length_w2opt)
    print("Testing COMPLETED ! Mean length1:",np.mean(predictions_length), "Mean length2:",np.mean(predictions_length_w2opt))

    n1, bins1, patches1 = plt.hist(predictions_length, 50, facecolor='b', alpha=0.75) # Histogram
    n2, bins2, patches2 = plt.hist(predictions_length_w2opt, 50, facecolor='g', alpha=0.75) # Histogram
    plt.xlabel('Tour length')
    plt.ylabel('Counts')
    plt.axis([3., 9., 0, 250])
    plt.grid(True)
    plt.show()

