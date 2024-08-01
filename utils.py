import numpy as np
import torch
from collections import OrderedDict
from learn2learn.utils import update_module
import math


# Expect tuples of (state, action, next_state, reward, done)
# Sample tuples of (state, action, next_state, reward, 1-done)
class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

def write_log(logtitle, log, log_path):
    f = open(log_path, mode='a')
    f.write(str(logtitle))
    f.write(str(log))
    f.write('\n')
    f.close()


def l1_penalty(var):
    return torch.abs(var).sum()

class Hot_Plug(object):
    def __init__(self, model):
        self.model = model
        self.params = OrderedDict(self.model.named_parameters())
    def update(self, lr=0.1):
        for param_name in self.params.keys():
            path = param_name.split('.')
            cursor = self.model
            for module_name in path[:-1]:
                cursor = cursor._modules[module_name]
            if lr > 0:
                cursor._parameters[path[-1]] = self.params[param_name] - lr*self.params[param_name].grad
            else:
                cursor._parameters[path[-1]] = self.params[param_name]
    def restore(self):
        self.update(lr=0)

def update_model(model, lr, grads=None):
    """
        Performs an update on model using grads and lr.
    """
    if grads is not None:
        params = list(model.parameters())
        for p, g in zip(params, grads):
            if g is not None:
                p.update = - lr * g
    return update_module(model)


# For SAC
def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)




# plot learning curve
def plot_results(path):
    model_path_1 = '%s/evaluation/evaluation_reward.npy' % (path)
    plot_path = '%s/evaluation/evaluation_reward.jpg' % (path)

    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def sliding_mean(data_array, window=5):
        data_array = np.array(data_array)
        new_list = []
        for i in range(len(data_array)):
            indices = list(range(max(i - window + 1, 0),
                                 min(i + window + 1, len(data_array))))
            avg = 0
            for j in indices:
                avg += data_array[j]
            avg /= float(len(indices))
            new_list.append(avg)

        return np.array(new_list)

    smooth_curve = True
    # smooth_curve = False

    def get_data(model_path):
        average_reward_train = np.load(model_path)

        # uniformly smoothing learning curve for clarity
        if smooth_curve:
            clean_statistics_train = sliding_mean(average_reward_train, window=30)
        else:
            clean_statistics_train = average_reward_train

        return clean_statistics_train

    model_path = []
    model_path.append(model_path_1)
    results_all = []
    for file in model_path:
        results_all.append(get_data(file))

    for rel in results_all:
        if smooth_curve:
            results_tmp = rel.tolist()
        else:
            results_tmp = rel
        frame_id = results_tmp.index(max(results_tmp))
        print("Max Evaluation reward Index: ", frame_id)
        print("Max Evaluation reward: ", max(results_tmp))
        print('******************************')

    plt.title('Learning curve')
    plt.xlabel('Episode'), plt.ylabel('Average Reward'), plt.legend(loc='best')

    plt.plot(results_all[0], color='#FFA500', label="method")

    plt.tight_layout()
    plt.legend()
    plt.savefig(plot_path)
    plt.close('all')

