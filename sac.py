import torch
import torch.nn.functional as F

from agents import GenericACAgent


class SACAgent(GenericACAgent):
    def update_actor(self, obs):        
        #========== TODO: start ==========
        # Sample actions and the log_prob of the actions from the actor given obs. Hint: This is the same as AC agent.
        # Get the two Q values from the double Q function critic and take the minimum value. Then calculate the actor loss which
        # is defined by self.alpha * log_prob - actor_Q. Make sure that gradient does not flow through the alpha parameter.

        
        #========== TODO: end ==========
        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the temperature parameter to achieve entropy close to the target entropy
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach())
        alpha_loss = alpha_loss.mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        return actor_loss.item(), -log_prob.mean().item(), alpha_loss.item()
    
    def update_critic(self, obs, action, reward, next_obs, not_done_no_max):
        #========== TODO: start ==========
        # Train the double Q function:
        # Hint step 1: Sample the next_action and log_prob of the next action using the self.actor and the next_obs. Use the code 
        # below in update_actor as a reference on how to do this
        
        # Hint step 2: Sample the two target Q values from the critic_target using next_obs and the sampled next_action. 
        # Calculate the target value by taking the min of the values and then subtracting self.alpha * log_prob
        # The target Q is the reward + (not_done_no_max * discount * target_value)
        # Hint: make sure gradients don't flow through the target_Q

        
        # Hint step 3:
        # Sample the current Q1 and Q2 values of the current state using the critic, and regress onto the target Q.
        # The loss is mse(Q1, targetQ) + mse(Q2 + target Q)

        
        #========== TODO: end ==========
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()