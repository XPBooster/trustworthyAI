import logging
import numpy as np
import pandas as pd
from pytz import timezone
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from torch.utils.data import DataLoader
from data_loader.dataset_read_data import CausalDataSet
from models.actor_graph import Actor
from rewards import Reward
from helpers.config_graph import get_config, print_config
from helpers.dir_utils import create_dir
from helpers.log_helper import LogHelper, VisualLogger
from helpers.tf_utils import set_seed
from helpers.analyze_utils import convert_graph_int_to_adj_mat, graph_prunned_by_coef, \
                                  count_accuracy, graph_prunned_by_coef_2nd
# from helpers.cam_with_pruning_cam import pruning_cam
from helpers.lambda_utils import BIC_lambdas
import matplotlib
matplotlib.use('Agg')


def main():
    # Setup for output directory and logging
    output_dir = 'output/{}'.format(datetime.now(timezone('Asia/Hong_Kong')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
    create_dir(output_dir)
    LogHelper.setup(log_path=f'{output_dir}/training.log', level_str='INFO')
    _logger = logging.getLogger(__name__)
    visual_logger = VisualLogger(f'{output_dir}')
    config, _ = get_config()
    create_dir(output_dir)
    set_seed(config.seed)

    # Log the configuration parameters
    _logger.info(f'Configuration parameters: {vars(config)}')    # Use vars to convert config to dict for logging

    file_path = f'{config.data_path}/real_data'
    solution_path = f'{config.data_path}/true_graph'
    training_set = CausalDataSet(config)
    training_loader = DataLoader(dataset=training_set, batch_size=config.batch_size, shuffle=True)
    # set penalty weights
    score_type = config.score_type
    reg_type = config.reg_type
    
    if config.lambda_flag_default:
        
        sl, su, strue = BIC_lambdas(training_set.inputdata, None, None, training_set.true_graph.T, reg_type, score_type)
        lambda1, lambda1_upper, lambda1_update_add = 0, 5, 1
        lambda2, lambda2_upper, lambda2_update_mul = 1/(10**(np.round(config.num_nodes/3))), 0.01, 10
        lambda_iter_num = config.lambda_iter_num
        _logger.info(f'Original sl: {sl}, su: {su}, strue: {strue}')
        _logger.info(f'Transformed sl: {sl}, su: {su}, lambda2: {lambda2}, true: {(strue-sl)/(su-sl)*lambda1_upper}')
        
    else:

        sl, su = config.score_lower, config.score_upper # test choices for the case with manually provided bounds

        if config.score_bd_tight:
            lambda1, lambda1_upper = 2, 2
        else:
            lambda1, lambda1_upper, lambda1_update_add = 0, 5, 1
        lambda2, lambda2_upper, lambda2_update_mul = 1 / (10 ** (np.round(config.num_nodes/3))), 0.01, config.lambda2_update
        lambda_iter_num = config.lambda_iter_num
        
    # actor
    actor = Actor(config)
    reward = Reward(config, training_set.inputdata, sl, su, lambda1_upper, False)
    _logger.info('RL Model: Finished creating training dataset, actor model and reward class')

    # Saver to save & restore all the variables.
    saver = tf.train.Saver(var_list=[v for v in tf.global_variables() if 'Adam' not in v.name], keep_checkpoint_every_n_hours=1.0)

    _logger.info('Starting Training...')
    sess_config = tf.ConfigProto(log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        # Run initialize op
        sess.run(tf.global_variables_initializer())

        # Test tensor shape
        _logger.info('Shape of actor.input: {}'.format(sess.run(tf.shape(actor.input_))))
        # _logger.info('training_set.true_graph: {}'.format(training_set.true_graph))
        # _logger.info('training_set.b: {}'.format(training_set.b))

        # Initialize useful variables
        rewards_avg_baseline, rewards_batches, reward_max_per_batch = [], [], []
        lambda1s, lambda2s = [], []
        graphss, probsss = [], []
        max_reward, max_rewards = float('-inf'), []
        image_count = 0
        accuracy_res, accuracy_res_pruned = [], []
        max_reward_score_cyc = (lambda1_upper+1, 0)

        # Summary writer
        visual_logger = VisualLogger(output_dir)
        # writer = tf.summary.FileWriter(output_dir, sess.graph)
        _logger.info('Starting training.')
            
        for i in (range(1, config.nb_epoch + 1)):

            if config.verbose:
                _logger.info(f'Start training for {i}-th epoch')
            for batch_x in training_loader:

                batch_x = batch_x.numpy()
                graphs_feed = sess.run(actor.graphs, feed_dict={actor.input_: batch_x})
                reward_feed = reward.cal_rewards(graphs_feed, lambda1, lambda2)

                max_reward = -reward.update_scores([max_reward_score_cyc], lambda1, lambda2)[0]
                max_reward_batch = float('inf')
                max_reward_batch_score_cyc = (0, 0)

                for reward_, score_, cyc_ in reward_feed:
                    if reward_ < max_reward_batch:
                        max_reward_batch = reward_
                        max_reward_batch_score_cyc = (score_, cyc_)

                max_reward_batch = -max_reward_batch

                if max_reward < max_reward_batch:
                    max_reward = max_reward_batch
                    max_reward_score_cyc = max_reward_batch_score_cyc

                # for average reward per batch
                reward_batch_score_cyc = np.mean(reward_feed[:,1:], axis=0)

                if config.verbose:
                    _logger.info('Finish calculating reward for current batch of graph')

                # Get feed dict
                feed = {actor.input_: batch_x, actor.reward_: -reward_feed[:,0], actor.graphs_:graphs_feed}

                summary, base_op, score_test, probs, graph_batch, \
                    reward_batch, reward_avg_baseline, train_step1, train_step2 = sess.run([actor.merged, actor.base_op,
                    actor.test_scores, actor.log_softmax, actor.graph_batch, actor.reward_batch, actor.avg_baseline, actor.train_step1,
                    actor.train_step2], feed_dict=feed)

                if config.verbose:
                    _logger.info('Finish updating actor and critic network using reward calculated')

                lambda1s.append(lambda1)
                lambda2s.append(lambda2)

                rewards_avg_baseline.append(reward_avg_baseline)
                rewards_batches.append(reward_batch_score_cyc)
                reward_max_per_batch.append(max_reward_batch_score_cyc)

                graphss.append(graph_batch)
                probsss.append(probs)
                max_rewards.append(max_reward_score_cyc)

                # logging
                if i == 1 or i % 5 == 0:
                    # if i >= 500:
                        # writer.add_summary(summary,i)

                    _logger.info('[iter {}] reward_batch: {}, max_reward: {}, max_reward_batch: {}'.format(i,
                                 reward_batch, max_reward, max_reward_batch))
                    # # other logger info; uncomment if you want to check
                    # # _logger.info('graph_batch_avg: {}'.format(graph_batch))
                    # # _logger.info('graph true: {}'.format(training_set.true_graph))
                    # # _logger.info('graph weights true: {}'.format(training_set.b))
                    # # _logger.info('=====================================')
                    #
                    # plt.figure(1)
                    # plt.plot(rewards_batches, label='reward per batch')
                    # plt.plot(max_rewards, label='max reward')
                    # plt.legend()
                    # plt.savefig('{}/reward_batch_average.png'.format(output_dir))
                    # plt.close()
                    #
                    # image_count += 1
                    # # this draw the average graph per batch.
                    # # can be modified to draw the graph (with or w/o pruning) that has the best reward
                    # fig = plt.figure(2)
                    # fig.suptitle('Iteration: {}'.format(i))
                    # ax = fig.add_subplot(1, 2, 1)
                    # ax.set_title('recovered_graph')
                    # ax.imshow(np.around(graph_batch.T).astype(int),cmap=plt.cm.gray)
                    # ax = fig.add_subplot(1, 2, 2)
                    # ax.set_title('ground truth')
                    # ax.imshow(training_set.true_graph, cmap=plt.cm.gray)
                    # plt.savefig(f'{output_dir}/recovered_graph_iteration_{image_count}.png')
                    # plt.close()

                # update lambda1, lamda2
                if (i+1) % lambda_iter_num == 0:
                    ls_kv = reward.update_all_scores(lambda1, lambda2)
                    # np.save(f'{output_dir}/solvd_dict_epoch_{i}.npy', np.array(ls_kv))
                    max_rewards_re = reward.update_scores(max_rewards, lambda1, lambda2)
                    rewards_batches_re = reward.update_scores(rewards_batches, lambda1, lambda2)
                    reward_max_per_batch_re = reward.update_scores(reward_max_per_batch, lambda1, lambda2)

                    # saved somewhat more detailed logging info
                    np.save(f'{output_dir}/solvd_dict.npy', np.array(ls_kv))
                    pd.DataFrame(np.array(max_rewards_re)).to_csv(f'{output_dir}/max_rewards.csv')
                    pd.DataFrame(rewards_batches_re).to_csv(f'{output_dir}/rewards_batch.csv')
                    pd.DataFrame(reward_max_per_batch_re).to_csv(f'{output_dir}/reward_max_batch.csv')
                    pd.DataFrame(lambda1s).to_csv(f'{output_dir}/lambda1s.csv')
                    pd.DataFrame(lambda2s).to_csv(f'{output_dir}/lambda2s.csv')

                    graph_int, score_min, cyc_min = np.int32(ls_kv[0][0]), ls_kv[0][1][1], ls_kv[0][1][-1]

                    if cyc_min < 1e-5:
                        lambda1_upper = score_min
                    lambda1 = min(lambda1+lambda1_update_add, lambda1_upper)
                    lambda2 = min(lambda2*lambda2_update_mul, lambda2_upper)
                    _logger.info('[iter {}] lambda1 {}, upper {}, lambda2 {}, upper {}, score_min {}, cyc_min {}'.format(i+1,
                                 lambda1, lambda1_upper, lambda2, lambda2_upper, score_min, cyc_min))

                    graph_batch = convert_graph_int_to_adj_mat(graph_int)

                    if reg_type == 'LR':
                        graph_batch_pruned = np.array(graph_prunned_by_coef(graph_batch, training_set.inputdata))
                    elif reg_type == 'QR':
                        graph_batch_pruned = np.array(graph_prunned_by_coef_2nd(graph_batch, training_set.inputdata))
                    elif reg_type == 'GPR':
                        # The R codes of CAM pruning operates the graph form that (i,j)=1 indicates i-th node-> j-th node
                        # so we need to do a tranpose on the input graph and another tranpose on the output graph
                        graph_batch_pruned = np.transpose(pruning_cam(training_set.inputdata, np.array(graph_batch).T))

                    # estimate accuracy
                    acc_est = count_accuracy(training_set.true_graph, graph_batch.T)
                    acc_est2 = count_accuracy(training_set.true_graph, graph_batch_pruned.T)

                    fdr, tpr, fpr, shd, nnz = acc_est['fdr'], acc_est['tpr'], acc_est['fpr'], acc_est['shd'], \
                                              acc_est['pred_size']
                    fdr2, tpr2, fpr2, shd2, nnz2 = acc_est2['fdr'], acc_est2['tpr'], acc_est2['fpr'], acc_est2['shd'], \
                                                   acc_est2['pred_size']

                    accuracy_res.append((fdr, tpr, fpr, shd, nnz))
                    accuracy_res_pruned.append((fdr2, tpr2, fpr2, shd2, nnz2))

                    np.save('{}/accuracy_res.npy'.format(output_dir), np.array(accuracy_res))
                    np.save('{}/accuracy_res2.npy'.format(output_dir), np.array(accuracy_res_pruned))

                    _logger.info('before pruning: fdr {}, tpr {}, fpr {}, shd {}, nnz {}'.format(fdr, tpr, fpr, shd, nnz))
                    _logger.info('after  pruning: fdr {}, tpr {}, fpr {}, shd {}, nnz {}'.format(fdr2, tpr2, fpr2, shd2, nnz2))

                # Save the variables to disk
                if i % max(1, int(config.nb_epoch / 5)) == 0 and i != 0:
                    curr_model_path = saver.save(sess, f'{output_dir}/tmp.ckpt', global_step=i)
                    _logger.info('Model saved in file: {}'.format(curr_model_path))

        _logger.info('Training COMPLETED !')
        saver.save(sess, f'{output_dir}/actor.ckpt')


if __name__ == '__main__':
    main()

