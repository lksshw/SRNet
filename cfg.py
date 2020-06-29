gpu = 0

lt = 1.
lt_alpha = 1.
lb = 1.
lb_beta = 10.
lf = 1.
lf_theta_1 = 10.
lf_theta_2 = 1.
lf_theta_3 = 500.
epsilon = 1e-5

# train
learning_rate = 1e-4 
decay_rate = 0.9
beta1 = 0.9
beta2 = 0.999 
max_iter = 500000
show_loss_interval = 50
write_log_interval = 50
save_ckpt_interval = 1000
gen_example_interval = 1000
checkpoint_savedir = 'logs/'
ckpt_path = '/content/train_step-4000.model'

# data
batch_size = 8
data_shape = [64, None]
data_dir = './datasets/srnet_data'
i_t_dir = 'i_t'
i_s_dir = 'i_s'
t_sk_dir = 't_sk'
t_t_dir = 't_t'
t_b_dir = 't_b'
t_f_dir = 't_f'
mask_t_dir = 'mask_t'
example_data_dir = 'custom_feed/labels'
example_result_dir = 'custom_feed/gen_logs'

# predict
predict_ckpt_path = None
predict_data_dir = None
predict_result_dir = 'custom_feed/result'