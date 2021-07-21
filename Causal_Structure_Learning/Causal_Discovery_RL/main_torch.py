import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from Causal_Structure_Learning.Causal_Discovery_RL.models.critic_torch import Critic
from Causal_Structure_Learning.Causal_Discovery_RL.models.decoder.single_layer_decoder_torch import SingleLayerDecoder
from Causal_Structure_Learning.Causal_Discovery_RL.models.encoder.transformer_encoder_torch import TransformerEncoder
from data_loader.dataset_read_data import CausalDataSet
from helpers.config_graph import get_config
from helpers.dir_utils import create_dir
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# from helpers.cam_with_pruning_cam import pruning_cam
from helpers.lambda_utils import BIC_lambdas
from helpers.log_helper import LogHelper, ResultWriter
from helpers.tf_utils import set_seed
from rewards import Reward
from itertools import chain


class ActorCritic(object):
    _logger = logging.getLogger(__name__)

    def __init__(self, config, result_writer):

        self.config = config
        self.result_writer = result_writer
        self.is_train = True
        # Data config
        self.num_nodes = config.num_nodes
        self.input_dimension = config.input_dimension

        # Reward config
        self.avg_baseline = config.init_baseline  # moving baseline for Reinforce
        self.alpha = config.alpha  # moving average update

        # Training config
        self.global_step = 0  # global step
        self.global_step2 = 0  # global step
        self.train_set = CausalDataSet(config)

        if self.config.encoder_type == 'TransformerEncoder':
            self.encoder = TransformerEncoder(self.config, self.is_train)
        # elif self.config.encoder_type == 'GATEncoder':
        #     encoder = GATEncoder(self.config, self.is_train)
        else:
            raise NotImplementedError('Current encoder type is not implemented yet!')

        if self.config.decoder_type == 'SingleLayerDecoder':
            self.decoder = SingleLayerDecoder(self.config, self.is_train)
        # elif self.config.decoder_type == 'TransformerDecoder':
        #     self.decoder = TransformerDecoder(self.config, self.is_train)
        # elif self.config.decoder_type == 'BilinearDecoder':
        #     self.decoder = BilinearDecoder(self.config, self.is_train)
        # elif self.config.decoder_type == 'NTNDecoder':
        #     self.decoder = NTNDecoder(self.config, self.is_train)
        # else:
        #     raise NotImplementedError('Current decoder type is not implemented yet!')
        self.critic = Critic(self.config, self.is_train)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.config.lr1, betas=(0.9, 0.999),
                                                 eps=1e-8)
        self.optimizer_actor = torch.optim.Adam(params=chain(self.encoder.parameters(), self.decoder.parameters()),
                                                lr=self.config.lr2, betas=(0.9, 0.999), eps=1e-8)
        self.scheduler_actor = CosineAnnealingWarmRestarts(self.optimizer_actor, T_0=self.config.lr1_tinit,
                                                           T_mult=self.config.lr1_tmult)
        self.scheduler_critic = CosineAnnealingWarmRestarts(self.optimizer_critic, T_0=self.config.lr2_tinit,
                                                            T_mult=self.config.lr2_tmult)
        self.mse = torch.nn.MSELoss(reduction=True, size_average=True)

    def fit(self):

        # lambda initialization
        if self.config.lambda_flag_default:

            sl, su, strue = BIC_lambdas(self.train_set.inputdata, None, None, self.train_set.true_graph.T,
                                        self.config.reg_type, self.config.score_type)
            lambda1, lambda1_upper, lambda1_update_add = 0, 5, 1
            lambda2, lambda2_upper, lambda2_update_mul = 1 / (10 ** (np.round(self.config.num_nodes / 3))), 0.01, 10
            
            _logger.info(f'Original sl: {sl}, su: {su}, strue: {strue}')
            _logger.info(f'Transformed sl: {sl}, su: {su}, strue: {(strue - sl) / (su - sl) * lambda1_upper}')

        else:

            sl, su = self.config.score_lower, self.config.score_upper  # test choices for the case with manually provided bounds
            if self.config.score_bd_tight:
                lambda1, lambda1_upper = 2, 2
            else:
                lambda1, lambda1_upper, lambda1_update_add = 0, 5, 1
            lambda2, lambda2_upper, lambda2_update_mul = 1 / (
                    10 ** (np.round(self.config.num_nodes / 3))), 0.01, self.config.lambda2_update
        lambda_iter_num = self.config.lambda_iter_num
        lambda1s, lambda2s = [], []
        
        rewards_avg_baseline, rewards_batches, reward_max_per_batch = [], [], []
        graph_batch_list, probsss = [], []
        max_reward, max_rewards = float('-inf'), []
        image_count = 0
        accuracy_res, accuracy_res_pruned = [], []
        max_reward_score_cyc = (lambda1_upper + 1, 0)
        reward = Reward(self.config, self.train_set.inputdata, sl, su, lambda1_upper, verbose_flag=False)
        train_loader = DataLoader(dataset=self.train_set, batch_size=self.config.batch_size, shuffle=True)
        iter_num = len(self.train_set) // self.config.batch_size

        for epoch in (range(1, self.config.nb_epoch + 1)):

            for idx, batch_x in enumerate(train_loader):

                encoder_output = self.encoder(batch_x)
                samples, logits_for_rewards, entropy_for_rewards = self.decoder(encoder_output)
                samples = torch.transpose(torch.stack(samples), 1, 0) # (batch, config.num_nodes, config.d_model)
                logits_for_rewards = torch.transpose(torch.stack(logits_for_rewards), 1, 0)
                entropy_for_rewards = torch.transpose(torch.stack(entropy_for_rewards), 1, 0)

                graph_batch = torch.mean(samples, dim=0)
                log_prob = samples * torch.log2(torch.sigmoid(logits_for_rewards) + 1e-5) + (
                            1 - samples) * torch.log2(1 - torch.sigmoid(logits_for_rewards)) # 1e-5 for numerical stability
                self.log_softmax = torch.mean(log_prob, dim=[1, 2])
                self.entropy_regularization = torch.mean(entropy_for_rewards, dim=[0, 1, 2])
                result_writer.add_scalar('train/entropy', self.entropy_regularization)
                result_writer.add_scalar('train/log_prob', torch.mean(self.log_softmax))
                critic_output = self.critic(encoder_output)
                reward_feed = reward.cal_rewards(samples, lambda1, lambda2)

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

                reward_batch_score_cyc = np.mean(reward_feed[:, 1:], axis=0)

                self.reward = -reward_feed[:, 0]
                self.reward_batch = self.reward.mean()
                self.graphs = samples

                # self.reward = reward.cal_rewards(graphs_gen, lambda1, lambda2)
                # Update moving_mean and moving_variance for batch normalization layers
                # Update baseline
                self.avg_baseline = self.config.alpha * self.avg_baseline + (
                        1.0 - self.config.alpha) * self.reward_batch
                # tf.summary.scalar('average baseline', self.avg_baseline)

                # Actor learning
                self.advantages = torch.tensor(self.reward - self.avg_baseline) - critic_output  # [Batch size, 1]
                self.optimizer_actor.zero_grad()
                self.loss_actor = torch.mean(self.advantages.detach() * self.log_softmax, dim=0) - self.entropy_regularization
                self.loss_actor.backward(retain_graph=True)
                # for name, parms in self.encoder.named_parameters():
                #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                #           ' -->grad_value:', parms.grad.shape)
                # for name, parms in self.decoder.named_parameters():
                #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                #           ' -->grad_value:', parms.grad.shape)
                torch.nn.utils.clip_grad_norm_(chain(self.encoder.parameters(), self.decoder.parameters()), max_norm=1)
                self.optimizer_actor.step()
                # variable_summaries('advantages', self.advantages, with_max_min=True)
                # tf.summary.scalar('loss1', self.loss1)
                # Minimize step

                # Critic learning rate
                # weights_ = 1.0  # weights_ = tf.exp(self.log_softmax-tf.reduce_max(self.log_softmax)) # probs / max_prob
                # critic_output.type
                self.loss_critic = self.mse(critic_output.type(torch.float64),
                                            torch.tensor(self.reward - self.avg_baseline, dtype=torch.float64))
                self.loss_critic.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
                self.optimizer_critic.step()
                self.scheduler_actor.step(epoch + idx / iter_num)
                self.scheduler_critic.step(epoch + idx / iter_num)
                result_writer.add_scalar('train/lr_actor', self.optimizer_actor.param_groups[-1]['lr'])
                result_writer.add_scalar('train/lr_critic', self.optimizer_critic.param_groups[-1]['lr'])
                result_writer.add_scalar('train/loss_actor', self.loss_actor)
                result_writer.add_scalar('train/loss_critic', self.loss_critic)

                lambda1s.append(lambda1)
                lambda2s.append(lambda2)

                rewards_avg_baseline.append(self.avg_baseline)
                rewards_batches.append(reward_batch_score_cyc)
                reward_max_per_batch.append(max_reward_batch_score_cyc)

                graph_batch_list.append(graph_batch)
                probsss.append(self.log_softmax)
                max_rewards.append(max_reward_score_cyc)


if __name__ == "__main__":
    output_dir = f'output/{time.time()}'
    create_dir(output_dir)
    LogHelper.setup(log_path=f'{output_dir}/training.log', level_str='INFO')
    _logger = logging.getLogger(__name__)
    result_writer = ResultWriter(project='1269547421/reinforment-graph',
                                 api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ODRiZmY5MC1hOGU0LTQwNDEtYmNmYS02YmRjYmZjNGE5NWIifQ==',
                                 verbose=False)
    config, _ = get_config()
    create_dir(output_dir)
    set_seed(config.seed)
    result_writer.add_hparam({'opt/lr': config.lr1,
                              'data/input_len': config.input_dimension,
                              'rl/alpha': config.alpha,
                              'rl/l1_graph': config.l1_graph_reg,
                              'actor/encoder': config.encoder_type,
                              'actor/d_model': config.d_model,
                              'actor/d_attn': config.d_model_attn,
                              'actor/num_heads': config.num_heads,
                              'actor/num_stacks': config.num_stacks,
                              'actor/decoder': config.decoder_type,
                              'actor/decoder_d_model': config.decoder_d_model,
                              'critic/score_type': config.score_type,
                              'critic/reg_type': config.reg_type,
                              'critic/lambda_iter_num': config.lambda_iter_num,
                              'critic/lambda1_update': config.lambda1_update,
                              'critic/lambda2_update': config.lambda2_update})
    _logger.info(f'Configuration parameters: {vars(config)}')  # Use vars to convert config to dict for logging
    actor = ActorCritic(config, result_writer)
    actor.fit()
