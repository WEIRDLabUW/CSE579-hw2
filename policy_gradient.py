import torch
import numpy as np
from torch import optim
from utils import log_density
from rollouts import rollout


def train_model(policy, baseline, trajs, policy_optim, baseline_optim, device, gamma=0.99, baseline_train_batch_size=64,
                baseline_num_epochs=5):
    states_all = []
    actions_all = []
    returns_all = []
    for traj in trajs:
        states_singletraj = traj['observations']
        actions_singletraj = traj['actions']
        rewards_singletraj = traj['rewards']
        returns_singletraj = np.zeros_like(rewards_singletraj)
        #========== TODO: start ==========
        # Compute the return-to-go for this trajectory.
        # Hint: G_t = r[t] + gamma * r[t+1] + gamma^2 * r[t+2] + ... iterate t backwards as a running sum.



        #========== TODO: END ==========
        states_all.append(states_singletraj)
        actions_all.append(actions_singletraj)
        returns_all.append(returns_singletraj)
    states = np.concatenate(states_all)
    actions = np.concatenate(actions_all)
    returns = np.concatenate(returns_all)

    # Normalize the returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)
    for epoch in range(baseline_num_epochs):
        np.random.shuffle(arr)
        for i in range(n // baseline_train_batch_size):
            batch_index = arr[baseline_train_batch_size * i: baseline_train_batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index).to(device)
            inputs = torch.Tensor(states).to(device)[batch_index]
            target = torch.Tensor(returns).to(device)[batch_index]
            
            #========== TODO: start ==========
            # Train baseline by regressing onto returns.
            # Hint: Regress the baseline from each state onto the above
            # computed return to go. You can use similar code to behavior cloning to do so. This should be
            # 2 lines of code



            #========== TODO: END ==========
            baseline_optim.zero_grad()
            loss.backward()
            baseline_optim.step()

    action, std, logstd = policy(torch.Tensor(states).to(device))
    log_policy = log_density(torch.Tensor(actions).to(device), policy.mu, std, logstd)
    baseline_pred = baseline(torch.from_numpy(states).float().to(device))
    #========== TODO: start ==========
    # Train policy by optimizing surrogate objective: -log prob * (return - baseline)
    # Return is computed above, as well as log_probs and baselines.
    # Hint: Policy gradient is given by: \grad log prob(a|s)* (return - baseline)
    # Hint: Then simply compute the surrogate objective by taking the objective as -log prob * (return - baseline)
    # Hint: You can then use standard pytorch machinery to take *one* gradient step on the policy


    
    #========== TODO: END ==========
    policy_optim.zero_grad()
    loss.backward()
    policy_optim.step()

    del states, actions, returns, states_all, actions_all, returns_all


# Training loop for policy gradient
def simulate_policy_pg(env, policy, baseline, num_epochs=200, batch_size=100,
                       gamma=0.99, baseline_train_batch_size=64, baseline_num_epochs=5, print_freq=10, device = "cuda", render=False):
    policy_optim = optim.Adam(policy.parameters())
    baseline_optim = optim.Adam(baseline.parameters())

    for iter_num in range(num_epochs):
        sample_trajs = []

        # Sampling trajectories
        for it in range(batch_size):
            sample_traj = rollout(
                env,
                policy,
                render=False)
            sample_trajs.append(sample_traj)

        # Logging returns occasionally
        if iter_num % print_freq == 0:
            rewards_np = np.mean(np.asarray([traj['rewards'].sum() for traj in sample_trajs]))
            path_length = np.max(np.asarray([traj['rewards'].shape[0] for traj in sample_trajs]))
            print("Episode: {}, reward: {}, max path length: {}".format(iter_num, rewards_np, path_length))

        # Training model
        train_model(policy, baseline, sample_trajs, policy_optim, baseline_optim, device, gamma=gamma,
                    baseline_train_batch_size=baseline_train_batch_size, baseline_num_epochs=baseline_num_epochs)
