import math
import numpy as np

class UCB():   
    def __init__(self, arms, means=None):
        self.arms = arms # Key is genre name, value is number of pulls
        self.true_means = means # true means are actual rewards from training
        self.emp_means = {key: 0.0 for key in arms} # empirical means are obtained via testing
        self.emp_cumulative_means = {key: 0.0 for key in arms}
        self.alpha = 1
        self.sigma = 1
        self.cumulative_reward = 0
        self.regret = []
    
    def reset(self):
        self.emp_means = {key: 0.0 for key in self.arms}
        self.arms = {key: 0.0 for key in self.arms}
        self.cumulative_reward = 0
        self.regret.clear()
    
    def select_arm(self, round):
        mean_interval = {}
        pulls_list = np.array([])
        for _, p in self.arms.items():
            pulls_list = np.append(pulls_list, p)
        
        for genre, pulls in self.arms.items():
            if pulls == 0:
                return genre
            
            mu_k_hat = self.emp_means[genre]
            interval = math.sqrt( ( 2 * self.alpha * math.pow(self.sigma, 2) * math.log10(round) ) / pulls )
            #interval = math.sqrt( ( 2 * self.alpha * math.pow(self.sigma, 2) * math.log10(np.sum(pulls_list)) ) / pulls )
            mean_interval[genre] = mu_k_hat + interval
        
        max_genre = max(mean_interval, key=mean_interval.get)
        return max_genre
    
    def update(self, arm, reward, round, meta_user=None):
        self.arms[arm] += 1 # increment pulls of the chosen arm
        self.emp_cumulative_means[arm] += reward
        self.emp_means[arm] = self.emp_cumulative_means[arm] / self.arms[arm] # update emp. mean
        self.cumulative_reward += reward
        
        optimal_arm = 5
        #if (meta_user is not None):
        #    optimal_arm = max(self.true_means[meta_user].values())
        #else:
        #    optimal_arm = self.true_means[arm]
        optimal_reward = optimal_arm * round
        self.regret.append(optimal_reward - self.cumulative_reward)
        return
    
    def get_regret(self):
        return self.regret
    
    def get_arms(self):
        return self.arms
    
    def get_emp_means(self):
        return self.emp_means
