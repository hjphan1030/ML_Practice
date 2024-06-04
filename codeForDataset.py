import tweepy
import csv

# Authenticate to Twitter
auth = tweepy.OAuthHandler("", "")
auth.set_access_token("", "")

# Create API object
api = tweepy.API(auth)

# Define gender hatred keywords
keywords = ["misandrist", "misogynist", "한남", "한녀", "페미"]

# Collect tweets
tweets = []
for keyword in keywords:
    tweets.extend(tweepy.Cursor(api.search_tweets, q=keyword, lang="en", tweet_mode="extended").items(1000))

# Extract tweet text and keyword
tweet_data = [(tweet.full_text, keyword) for tweet in tweets for keyword in keywords if keyword in tweet.full_text.lower()]

# Save to CSV file
with open('tweets.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Tweet", "Keyword"])
    writer.writerows(tweet_data)