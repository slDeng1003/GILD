import torch
import torch.nn.functional as F
from torch import autograd
import numpy as np
import copy
from utils import Hot_Plug
from utils import update_model
from learn2learn import clone_module
from model import Actor, Critic, GILD_Network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
class DDPG_GILD(object):
    def __init__(self, state_dim, action_dim, max_action, demo_traj, args):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.gild_network =  GILD_Network(action_dim+state_dim+action_dim).to(device)
        self.meta_optimizer = torch.optim.Adam(self.gild_network.parameters(), lr=args.gild_lr, weight_decay=1e-4)
        self.hotplug = Hot_Plug(self.actor)
        self.lr_actor = args.actor_lr

        self.max_action = max_action
        self.loss_store = []
        self.total_it = 0
        self.discount = args.discount
        self.tau = args.tau

        self.demo_states = torch.tensor(demo_traj[:, 0 : state_dim]).float().to(device)
        self.demo_actions = torch.tensor(demo_traj[:, state_dim : state_dim+action_dim]).float().to(device)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        next_a = self.actor_target(next_state)
        target_Q = self.critic_target(next_state, next_a)
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # Sample demonstration data
        ind = np.random.randint(low=0, high=self.demo_states.shape[0], size=batch_size)
        demo_state = self.demo_states[ind]
        demo_action = self.demo_actions[ind]

        # Sample replay buffer for meta test
        state_val, _, _, _, _ = replay_buffer.sample(batch_size)
        
        # RL + BC
        actor_tmp = clone_module(self.actor)
            # Compute actor_tmp's RL loss
        loss_rl_tmp = -self.critic(state, actor_tmp(state)).mean()
            # Compute actor_tmp's BC loss
        loss_bc = F.mse_loss(actor_tmp(demo_state), demo_action)
            # Compute actor_tmp 's total loss
        actor_tmp_loss = loss_rl_tmp + loss_bc
            # Optimize actor_tmp
        gradients_actor_tmp = autograd.grad(actor_tmp_loss,
                                            actor_tmp.parameters(),
                                            )
        update_model(actor_tmp, self.lr_actor, gradients_actor_tmp)

        # RL + GILD
            # Compute actor's RL loss
        loss_rl = -self.critic(state, self.actor(state)).mean()
            # Compute actor's GILD loss
        concat_output = torch.cat([self.actor(demo_state), demo_state, demo_action], 1)
        loss_gild = self.gild_network(concat_output).mean()
            # Compute actor's total loss
        actor_loss = loss_rl + loss_gild
            # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward(create_graph=True)
        self.hotplug.update(self.lr_actor)

        # compute meta loss
            # RL loss of actor_tmp (RL+BC)
        action_val_bc = actor_tmp(state_val)
        policy_bc_loss_val = -self.critic(state_val, action_val_bc).mean().detach() # -Q_bc
            # RL loss of actor(RL+GILD)
        action_val_gild = self.actor(state_val)
        policy_loss_val_gild = -self.critic(state_val, action_val_gild).mean() # -Q_gild
            # meta loss
        utility = torch.tanh(policy_bc_loss_val - policy_loss_val_gild) # tanh(Q_gild-Q_bc)
        loss_meta = -utility # maximize utility

        # Meta optimization of gild network
        self.meta_optimizer.zero_grad()
        grad_omega = autograd.grad(loss_meta, self.gild_network.parameters())
        for gradient, variable in zip(grad_omega, self.gild_network.parameters()):
            variable.grad.data = gradient
        self.meta_optimizer.step()
        self.actor_optimizer.step()
        self.hotplug.restore()

        del actor_tmp


        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


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
