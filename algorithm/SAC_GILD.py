import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch import autograd
import numpy as np
import copy
from utils import soft_update, hard_update, Hot_Plug, update_model
from learn2learn import clone_module
from model import GaussianPolicy, Critic_DoubleQ, DeterministicPolicy, GILD_Network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAC_GILD(object):
    def __init__(self, state_dim, action_space, demo_traj, args):
        self.critic = Critic_DoubleQ(state_dim, action_space.shape[0], 256).to(device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=args.critic_lr)

        self.critic_target = Critic_DoubleQ(state_dim, action_space.shape[0], 256).to(device)
        hard_update(self.critic_target, self.critic)

        self.policy_type = args.policy_type
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.alpha = args.alpha
        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
                self.alpha_optimizer = Adam([self.log_alpha], lr=args.actor_lr)

            self.actor = GaussianPolicy(state_dim, action_space.shape[0], 256, action_space).to(device)
            self.actor_optimizer = Adam(self.actor.parameters(), lr=args.actor_lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.actor = DeterministicPolicy(state_dim, action_space.shape[0], 256, action_space).to(device)
            self.actor_optimizer = Adam(self.actor.parameters(), lr=args.actor_lr)
        
        self.gild_network = GILD_Network(action_space.shape[0] + state_dim + action_space.shape[0]).to(device)
        self.meta_optimizer = torch.optim.Adam(self.gild_network.parameters(), lr=args.gild_lr, weight_decay=1e-4)
        
        self.hotplug = Hot_Plug(self.actor)
        self.lr_actor = args.actor_lr

        self.policy_freq = args.policy_freq
        self.gamma = args.discount
        self.tau = args.tau
        self.total_it = 0
        self.loss_store = []

        self.demo_states = torch.tensor(demo_traj[:, 0 : state_dim]).float().to(device)
        self.demo_actions = torch.tensor(demo_traj[:, state_dim : state_dim+action_space.shape[0]]).float().to(device)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Optimize the Critic
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state)
            qf1_next_target, qf2_next_target = self.critic_target(next_state, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward + not_done * self.gamma * (min_qf_next_target)
            # Compute critic loss
        qf1, qf2 = self.critic(state,
                               action)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()
            
        # Sample replay buffer for meta test
        state_val, _, _, _, _ = replay_buffer.sample(batch_size)
        
        # Sample demonstration data
        ind = np.random.randint(low=0, high=self.demo_states.shape[0], size=batch_size)
        demo_state = self.demo_states[ind]
        demo_action = self.demo_actions[ind]

        # RL + BC
        actor_tmp = clone_module(self.actor)
            # Compute actor_tmp's RL loss
        pi_tmp, log_pi_tmp, _ = actor_tmp.sample(state)
        qf1_pi, qf2_pi = self.critic(state, pi_tmp)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        loss_rl_tmp = ((self.alpha * log_pi_tmp) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            # Compute actor_tmp's BC loss
        if self.policy_type == "Gaussian":
            bc_log_probs = actor_tmp.log_prob(demo_state, demo_action) # for stochastic policy
            loss_bc = -torch.mean(bc_log_probs)
        else: 
            loss_bc = F.mse_loss(actor_tmp(demo_state), demo_action) # for deterministic policy
            # Compute actor_tmp 's total loss
        actor_tmp_loss = loss_rl_tmp + loss_bc
            # Optimize actor_tmp
        gradients_actor_tmp = autograd.grad(actor_tmp_loss,
                                            actor_tmp.parameters(),
                                            )
        update_model(actor_tmp, self.lr_actor, gradients_actor_tmp)

        # RL + GILD
            # Compute actor's RL loss
        pi, log_pi, _ = self.actor.sample(state)
        qf1_pi, qf2_pi = self.critic(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        loss_rl = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            # Compute actor's GILD loss
        concat_output = torch.cat([pi, demo_state, demo_action],1)
        loss_gild = self.gild_network(concat_output).mean()
            # Compute actor's total loss
        actor_loss = loss_rl + loss_gild
            # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward(create_graph=True)
        self.hotplug.update(self.lr_actor)

        # compute meta loss
            # RL loss of actor_tmp (RL+BC)
        pi_val_bc, log_pi_val_bc, _ = actor_tmp.sample(state_val)
        qf1_pi_val_bc, qf2_pi_val_bc = self.critic(state_val, pi_val_bc)
        min_qf_pi_val_bc = torch.min(qf1_pi_val_bc, qf2_pi_val_bc)
        policy_loss_val_bc = ((self.alpha * log_pi_val_bc) - min_qf_pi_val_bc).mean()
            # RL loss of actor(RL+GILD)
        pi_val_gild, log_pi_val_gild, _ = self.actor.sample(state_val)
        qf1_pi_val_gild, qf2_pi_val_gild = self.critic(state_val, pi_val_gild)
        min_qf_pi_val_gild = torch.min(qf1_pi_val_gild, qf2_pi_val_gild)
        policy_loss_val_gild = ((self.alpha * log_pi_val_gild) - min_qf_pi_val_gild).mean()
            # meta loss
        utility = torch.tanh(policy_bc_loss_val - policy_loss_val_gild)
        loss_meta = -utility

        # Meta optimization of GILD network
        self.meta_optimizer.zero_grad()
        grad_omega = torch.autograd.grad(loss_meta, self.gild_network.parameters())
        for gradient, variable in zip(grad_omega, self.gild_network.parameters()):
            variable.grad.data = gradient
        self.meta_optimizer.step()
        self.actor_optimizer.step()
        self.hotplug.restore()

        del actor_tmp

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        else:
            alpha_loss = torch.tensor(0.).to(device)

        # soft update
        if self.total_it % self.policy_freq == 0:
            soft_update(self.critic_target, self.critic, self.tau)


    def save(self, directory, model_period):
        torch.save(self.critic.state_dict(), directory + ("/trained_models/%s/critic.pth" % (model_period)))
        torch.save(self.critic_optimizer.state_dict(), directory + ("/trained_models/%s/critic_optimizer.pth" % (model_period)) )
        torch.save(self.actor.state_dict(), directory + ("/trained_models/%s/actor.pth" % (model_period)))
        torch.save(self.actor_optimizer.state_dict(), directory + ("/trained_models/%s/actor_optimizer.pth"  % (model_period)))
        torch.save(self.gild_network.state_dict(),  (directory + "/trained_models/%s/gild_network.pth" % (model_period)) )
        torch.save(self.meta_optimizer.state_dict(),  (directory + "/trained_models/%s/meta_optimizer.pth" % (model_period)) )

    def load(self, directory):
        self.critic.load_state_dict(torch.load(directory + "/critic.pth"))
        self.critic_optimizer.load_state_dict(torch.load(directory + "/critic_optimizer.pth"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(directory + "/actor.pth"))
        self.actor_optimizer.load_state_dict(torch.load(directory + "/actor_optimizer.pth"))
        self.actor_target = copy.deepcopy(self.actor)

        self.gild_network.load_state_dict(torch.load(directory + "/gild_network.pth"))
        self.meta_optimizer.load_state_dict(torch.load(directory + "/meta_optimizer.pth"))