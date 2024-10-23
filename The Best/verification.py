## Question : did everyone post ? 

import pandas as pd 

posts_1 = pd.read_csv("instagram_posts_0911_1111.csv", delimiter=',') 
posts_2 = pd.read_csv("instagram_posts_1211_1611.csv", delimiter=',')

posts = pd.concat([posts_1, posts_2],axis = 0)

accounts = pd.read_csv("instagram_accounts.csv")

# If no assertion error, then the answer is yes, furthermore the accounts and posts have the same length, meaning that every user posted once and only once.

assert posts.shape[0] == accounts.shape[0], "The accounts and posts datasets have the same lengths"
assert set(accounts['id_user']) - set(posts['id_user'])  == set(), "There are people who never posted"
 