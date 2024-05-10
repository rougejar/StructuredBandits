import math
import random
import numpy as np
from ucb import UCB

class UCBC(UCB):
    def __init__(self, means, arms):
        self.conf_set = []
        self.comp_set = []
        self.gamma = 1
        self.d = 2
        super().__init__(means, arms)

    def construct_conf_set(self, t):
        self.conf_set.clear()
        
        theta_sets = {arm: set() for arm in self.arms}
        
        for arm_k, n_k in self.arms.items():
            for theta, genre_ratings in self.true_means.items():
                mu_k = genre_ratings[arm_k]
                mu_k_hat = self.emp_means[arm_k]
                interval = math.sqrt( ( 2 * self.alpha * math.pow(self.sigma, 2) * math.log10(t) ) / n_k )
                
                if (abs(mu_k - mu_k_hat) < interval):
                    theta_sets[arm_k].add(theta)
                
        theta_intersection = set.intersection(*theta_sets.values())
        self.conf_set = list(theta_intersection)
        return
    
    def construct_comp_set(self):
        self.comp_set.clear()
        if len(self.conf_set) == 0:
            self.comp_set = list(self.arms)
        else:
            conf_thetas = {key: value for key, value in self.true_means.items() if key in self.conf_set}
            for arm_k, n_k in self.arms.items():
                for theta, genre_ratings in conf_thetas.items():
                    mu_k = genre_ratings[arm_k]
                    mu_l = [value for key, value in genre_ratings.items() if key != arm_k]
                    if (mu_k >= max(mu_l)) and (arm_k not in self.comp_set):
                        self.comp_set.append(arm_k)
        
        return
    
    def select_arm(self, round, informativeness, informativeness_arm):
        mean_interval = {}
        for genre, pulls in self.arms.items():
            if pulls == 0:
                return genre
        
        comp_arms = {key: value for key, value in self.arms.items() if key in self.comp_set}
        
        for genre, pulls in comp_arms.items():
            mu_k_hat = self.emp_means[genre]
            interval = math.sqrt( ( 2 * self.alpha * math.pow(self.sigma, 2) * math.log10(round) ) / pulls )
            mean_interval[genre] = mu_k_hat + interval
        
        max_genre = max(mean_interval, key=mean_interval.get)
        
        inform_prob = self.gamma / math.pow(round, self.d)
        
        selected_arm = max_genre
        if informativeness:
            probs = [inform_prob, 1 - inform_prob]
            selected_arm = random.choices([max_genre, informativeness_arm], weights=probs)[0]
        
        return selected_arm
    
    def reset(self):
        self.emp_means = {key: 0.0 for key in self.arms}
        self.arms = {key: 0.0 for key in self.arms}
        self.cumulative_reward = 0
        self.regret.clear()
        self.conf_set.clear()
        self.comp_set.clear()
    
    def guess_theta(self):
        mu_list = {theta: {arm : 0.0 for arm in self.arms.keys()} for theta in self.true_means.keys()}
        
        for genre, pulls in self.arms.items():
            for theta, genre_ratings in self.true_means.items():
                mu = self.true_means[theta][genre]
                mu_hat = self.emp_means[genre]
                mu_list[theta][genre] = abs(mu - mu_hat)
        
        sums = {key: sum(inner_dict.values()) for key, inner_dict in mu_list.items()}
        
        min_theta_key = min(sums, key=sums.get)
        min_theta_val = mu_list[min_theta_key]
        
        correct_theta_key = (45,3)
        correct_theta_val = mu_list[correct_theta_key]
        return
    
    def calc_informativeness(self):
        
        genre_theta_ratings = {genre: np.array([], dtype=float) for genre in self.arms.keys()}
        conf_thetas = {key: value for key, value in self.true_means.items() if key in self.conf_set}
        
        for genre, ratings in genre_theta_ratings.items():
            for theta, genre_ratings in conf_thetas.items():
                genre_theta_ratings[genre] = np.append(genre_theta_ratings[genre], genre_ratings[genre])

        genre_entropies = {genre: np.array([], dtype=float) for genre in self.arms.keys()}
        
        num_bins = 10
        for genre, ratings in genre_theta_ratings.items():
            hist, bin_edges = np.histogram(genre_theta_ratings[genre], bins=num_bins, density=True)
            entropy = np.sum((hist * np.log2(hist + 1e-10)))
            genre_entropies[genre] = np.append(genre_entropies[genre], entropy)
        
        max_entropy = max(genre_entropies, key=genre_entropies.get)
        return max_entropy