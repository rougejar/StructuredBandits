import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

from ucbc import UCBC
from ucb import UCB



def select_movie(genre_movies, chosen_genre):
    random_movie = np.random.choice(genre_movies[chosen_genre])
    return random_movie


def choose_rating_ucb1(ucb, round, genre_movies, ratings):
    tries = 0
    movie_found = False
    while (movie_found == False):
        if (tries > 1000):
            raise Exception("Could not find movie.")
        chosen_arm = ucb.select_arm(round)
        chosen_movie = select_movie(genre_movies, chosen_arm)
        chosen_movie_ratings = ratings[ratings['movieId'] == chosen_movie]
        
        if len(chosen_movie_ratings) > 0:
            movie_found = True
        tries += 1
    
    chosen_rating = np.random.choice(chosen_movie_ratings['rating'])
    return chosen_arm, chosen_rating


def choose_rating_ucbc(ucbc, chosen_users, round, genre_movies, test_ratings, informative, informative_arm):
    tries = 0
    user_found = False
    while (user_found == False):
        if (tries > 1000):
            raise Exception("Could not find movie.")
        chosen_arm = ucbc.select_arm(round, informative, informative_arm)
        chosen_movie = select_movie(genre_movies, chosen_arm)
        chosen_movie_ratings = test_ratings[test_ratings['movieId'] == chosen_movie]
        chosen_user_ratings = chosen_movie_ratings[chosen_movie_ratings['userId'].isin(chosen_users)]
        if len(chosen_user_ratings) > 0:
            user_found = True
        tries += 1
    
    chosen_rating = np.random.choice(chosen_user_ratings['rating'])
    return chosen_arm, chosen_rating


def playUCB(ucb1, rounds, genre_movies, ratings, cumulative_regret):
    ucb1.reset()
    for round in range(rounds):
        #print("<----- Round (UCB): ", round, " ----->")
        chosen_arm, chosen_rating = choose_rating_ucb1(ucb1, round, genre_movies, ratings)
        ucb1.update(chosen_arm, chosen_rating, round)
    ucb1.get_emp_means()
    ucb1.get_arms()
    regret = ucb1.get_regret()
    cumulative_regret.extend(regret)
    ucb1.reset()
    return cumulative_regret


def playUCBC(ucbc, meta_user_ratings, chosen_meta_users, rounds, genres, genre_movies, test_ratings, cumulative_regret, informative=False):
    ucbc.reset()
    informative_arm = 'Animation'
    for chosen_meta_user, chosen_users in chosen_meta_users.items():
        for round in range(rounds):
            #print("<----- Round (UCB-C): ", round, " ----->")
            if (round >= len(genres)):
                ucbc.construct_conf_set(round)
                ucbc.construct_comp_set()
                
                if informative:
                    informative_arm = ucbc.calc_informativeness()
                
            chosen_arm, chosen_rating = choose_rating_ucbc(ucbc, chosen_users, round, genre_movies, test_ratings, informative, informative_arm)
            ucbc.update(chosen_arm, chosen_rating, round, chosen_meta_user)
            ucbc.guess_theta()
        ucbc.get_arms()
        regret = ucbc.get_regret()
        cumulative_regret.extend(regret)
        ucbc.reset()
    return cumulative_regret


def main():
    
    # Command line parser
    parser = argparse.ArgumentParser(description='Structured Bandit Simulator')
    
    def restricted_subset(value):
        allowed_letters = set('uci')
        if not set(value).issubset(allowed_letters):
            raise argparse.ArgumentTypeError(f"Invalid value: {value}. Must be a subset of 'u', 'c', 'i'")
        return value
    
    # Command line arguments
    parser.add_argument('-r', type=int, default=1000, help='Number of rounds to run the algorithm(s). Default is 1000.')
    parser.add_argument('-algo', type=restricted_subset, default='uci', help='The algorithm(s) to run. Type u for UCB, c for UCB-C, i for Informative-UCB-C. Can run multiple with multiple inputs. Default enables all.')
    parser.add_argument('-age', type=int, default=0, help='Age of meta user to test UCB-C on (if enabled). Default is random.')
    parser.add_argument('-occ', type=int, default=0, help='Occupation of meta user to test UCB-C on (if enabled). Default is random.')
    
    args = parser.parse_args()
    arg_rounds = args.r
    arg_algo = args.algo
    arg_age = args.age
    arg_occ = args.occ
    
    # Load datasets
    movies = pd.read_csv("Datasets/movies.csv", sep='::', engine='python')
    users = pd.read_csv("Datasets/users.csv", sep='::', engine='python')
    ratings = pd.read_csv("Datasets/ratings.csv", sep='::', engine='python')
    
    # Split ratings dataset into train and test sets
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.5, random_state=42)   
    
    # Create dictionary of meta-users
    meta_users = {}
    
    # Iterate through users to create metausers
    for _, row in users.iterrows():
        age_occ = (row['age'], row['occupation'])
        user_id = row['userId']
        if age_occ in meta_users:
            meta_users[age_occ].append(user_id)
        else:
            meta_users[age_occ] = [user_id]
    
    # Create dictionary of genres and corresponding ratings (initialized to 0)
    genres = {}
    for movie in movies['genres'].str.split('|'):
        for genre in movie:
            genres[genre] = 0.0
    
    # Create list of movies for each genre
    genre_movies = {genre: [] for genre in genres}
    for _, movie in movies.iterrows():
        movie_id = movie['movieId']
        movie_genres = movie['genres'].split('|')
        for g in movie_genres:
            genre_movies[g].append(movie_id)
    
    # Create dictionary of genre ratings (UCB only)
    genre_movie_ratings = {genre: np.array([]) for genre in genres}
    for genre, movieIds in genre_movies.items():
        movie_ratings = ratings[ratings['movieId'].isin(movieIds)]['rating']
        genre_movie_ratings[genre] = movie_ratings.tolist()
        genre_movie_ratings[genre] = np.mean(genre_movie_ratings[genre])
        if np.isnan(genre_movie_ratings[genre]):
            genre_movie_ratings[genre] = 0.0
    
    # Create dictionary of meta-user ratings (UCB-C only)
    meta_user_ratings = {key: {genre: np.array([]) for genre in genres} for key in meta_users}
    for ageocc, userIds in meta_users.items():
        filtered_ratings = train_ratings[train_ratings['userId'].isin(userIds)]
        for genre, movieIds in genre_movies.items():
            movie_ratings = filtered_ratings[filtered_ratings['movieId'].isin(movieIds)]['rating']
            meta_user_ratings[ageocc][genre] = movie_ratings.tolist()
            meta_user_ratings[ageocc][genre] = np.mean(meta_user_ratings[ageocc][genre])
            if np.isnan(meta_user_ratings[ageocc][genre]):
                meta_user_ratings[ageocc][genre] = 0.0
    
    regretUCB, regretUCBC, regretUCBCi = None, None, None
    ucb1 = UCB(genres, genre_movie_ratings)
    ucbc = UCBC(genres, meta_user_ratings)
    ucbci = UCBC(genres, meta_user_ratings)
    
    rounds = arg_rounds
    cumulative_regret_ucb = []
    cumulative_regret_ucbc = []
    cumulative_regret_ucbci = []
    
    # <-------------------- UCB -------------------->
    if 'u' in arg_algo:
        regretUCB = playUCB(ucb1, rounds, genre_movies, ratings, cumulative_regret_ucb)

    # <-------------------- UCB-C -------------------->
    chosen_meta_users = {(1,1): meta_users[(1,1)]}
    
    if 'c' or 'i' in arg_algo:
        if (arg_age, arg_occ) in meta_users:
            chosen_meta_users = {(arg_age,arg_occ): meta_users[(arg_age,arg_occ)]}
        else:
            chosen_meta_users = dict(random.sample(meta_users.items(), 1))
    
    if 'i' in arg_algo:
        regretUCBCi = playUCBC(ucbci, meta_user_ratings, chosen_meta_users, rounds, genres, genre_movies, test_ratings, cumulative_regret_ucbci, informative=True)
    if 'c' in arg_algo:
        regretUCBC = playUCBC(ucbc, meta_user_ratings, chosen_meta_users, rounds, genres, genre_movies, test_ratings, cumulative_regret_ucbc, informative=False)
    
    if regretUCB is not None:
        plt.plot(range(rounds), regretUCB, label='UCB')
    if regretUCBC is not None:
        plt.plot(range(rounds), regretUCBC, label='UCB-C')
    if regretUCBCi is not None:
        plt.plot(range(rounds), regretUCBCi, label='UCB-C-Entropy')
        
    plt.xlabel('Rounds')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret vs Total Rounds')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
