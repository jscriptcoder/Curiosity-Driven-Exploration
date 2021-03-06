import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from agent import Agent
from common import CategoricalPolicy, Critic
from common.utils import device, soft_update, make_experience, from_experience


class SACAgent(Agent):
    """Soft Actor-Critic for Discrete Action Settings
    https://arxiv.org/abs/1910.07207

    Soft Actor-Critic
    https://arxiv.org/abs/1801.01290
    https://spinningup.openai.com/en/latest/algorithms/sac.html
    https://www.youtube.com/watch?v=CLZkpo8rEGg
    https://towardsdatascience.com/soft-actor-critic-demystified-b8427df61665
    """

    name = 'SAC'

    def __init__(self, config):
        super().__init__(config)

        self.policy = CategoricalPolicy(config.state_size,
                                        config.action_size, 
                                        config.hidden_actor, 
                                        config.activ_actor)

        self.policy_optim = config.optim_actor(self.policy.parameters(),
                                               lr=config.lr_actor)

        self.Q1_local = Critic(config.state_size,
                               config.action_size,
                               config.hidden_critic,
                               config.activ_critic)

        self.Q1_target = Critic(config.state_size,
                                config.action_size,
                                config.hidden_critic,
                                config.activ_critic)

        self.Q1_target.load_state_dict(self.Q1_local.state_dict())

        self.Q1_optim = config.optim_critic(self.Q1_local.parameters(),
                                            lr=config.lr_critic)

        self.Q2_local = Critic(config.state_size,
                               config.action_size,
                               config.hidden_critic,
                               config.activ_critic)

        self.Q2_target = Critic(config.state_size,
                                config.action_size,
                                config.hidden_critic,
                                config.activ_critic)

        self.Q2_target.load_state_dict(self.Q2_local.state_dict())

        self.Q2_optim = config.optim_critic(self.Q2_local.parameters(),
                                            lr=config.lr_critic)

        if config.alpha_auto_tuning:
            # temperature variable to be learned, and its target entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.detach().exp()

            # Target entropy is -|A|
            self.target_entropy = -config.action_size

            self.alpha_optim = config.optim_alpha([self.log_alpha], lr=config.lr_alpha)
        else:
            self.alpha = config.alpha
    
    def act(self, state, train=True):
        # Since there is only one state we're gonna insert a new dimension
        # so we make it as if it was batch_size=1
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        if train:
            action, _, _ = self.policy.sample_action(state)
        else:
            action = self.policy.greedy_action(state)

        return action.item()

    def update_Q(self,
                 states,
                 actions,
                 next_states,
                 rewards,
                 dones):

        grad_clip_critic = self.config.grad_clip_critic
        use_huber_loss = self.config.use_huber_loss
        gamma = self.config.gamma
        tau = self.config.tau

        _, next_action_probs, next_log_probs = self.policy.sample_action(next_states)

        with torch.no_grad():
            Q1_targets_next = self.Q1_target(next_states)
            Q2_targets_next = self.Q2_target(next_states)

            Q_targets_next = torch.min(Q1_targets_next, Q2_targets_next)
            Q_targets_next = Q_targets_next - self.alpha * next_log_probs

            # Expectation of Q target
            Q_targets_next = torch.sum(next_action_probs * Q_targets_next, dim=1, keepdim=True)

            Q_targets = rewards + (gamma * Q_targets_next) * (1 - dones)

        Q1_expected = self.Q1_local(states).gather(1, actions)
        Q2_expected = self.Q2_local(states).gather(1, actions)

        # Compute critic loss
        if use_huber_loss:
            Q1_loss = F.smooth_l1_loss(Q1_expected, Q_targets)
            Q2_loss = F.smooth_l1_loss(Q2_expected, Q_targets)
        else:
            Q1_loss = F.mse_loss(Q1_expected, Q_targets)
            Q2_loss = F.mse_loss(Q2_expected, Q_targets)

        self.value_losses.append(torch.max(Q1_loss, Q2_loss).item())

        # Minimize the loss
        self.Q1_optim.zero_grad()
        Q1_loss.backward()

        if grad_clip_critic is not None:
            clip_grad_norm_(self.Q1_local.parameters(), grad_clip_critic)

        self.Q1_optim.step()

        self.Q2_optim.zero_grad()
        Q2_loss.backward()

        if grad_clip_critic is not None:
            clip_grad_norm_(self.Q2_local.parameters(), grad_clip_critic)

        self.Q2_optim.step()

        soft_update(self.Q1_local, self.Q1_target, tau)
        soft_update(self.Q2_local, self.Q2_target, tau)

    def update_policy(self, states):
        grad_clip_actor = self.config.grad_clip_actor

        _, action_probs, log_props = self.policy.sample_action(states)

        # Expectations of entropies
        log_props = torch.sum(action_probs * log_props, dim=1, keepdim=True)

        Q1_pred = self.Q1_local(states)
        Q2_pred = self.Q2_local(states)

        Q_pred = torch.min(Q1_pred, Q2_pred)

        # Expectations of Q
        Q_pred = torch.sum(action_probs * Q_pred, dim=1, keepdim=True)

        policy_loss = self.alpha * log_props - Q_pred
        policy_loss = policy_loss.mean()

        self.policy_losses.append(policy_loss.item())

        self.policy_optim.zero_grad()
        policy_loss.backward()

        if grad_clip_actor is not None:
            clip_grad_norm_(self.policy.parameters(),
                            grad_clip_actor)

        self.policy_optim.step()

        return log_props.detach()

    def try_update_alpha(self, log_probs):
        alpha_auto_tuning = self.config.alpha_auto_tuning

        if alpha_auto_tuning:
            alpha_loss = -self.log_alpha * (log_probs + self.target_entropy)
            alpha_loss = alpha_loss.mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.detach().exp()

    def learn(self, experiences):
        (states, 
         actions, 
         rewards, 
         next_states, 
         dones) = from_experience(experiences)
        
        self.update_Q(states, 
                      actions, 
                      next_states, 
                      rewards, 
                      dones)
        
        log_props = self.update_policy(states)
        
        self.try_update_alpha(log_props)

    def save_weights(self, path='weights'):
        torch.save(self.policy.state_dict(),
                   '{}/{}_policy_checkpoint.ph'.format(path, self.name))

    def load_weights(self, path='weights'):
        self.policy.\
            load_state_dict(
                torch.load('{}/{}_policy_checkpoint.ph'.
                           format(path, self.name),
                           map_location='cpu'))

    def summary(self, agent_name='SAC [discrete] Agent'):
        print('{}:'.format(agent_name))
        print('==========')
        print('')
        print('Policy Network:')
        print('--------------')
        print(self.policy)
        print('')
        print('Q Network:')
        print('----------')
        print(self.Q1_local)