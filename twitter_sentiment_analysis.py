from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import json
import keys as k
import sentiment_mod as s

#consumer key, consumer secret, access token, access secret.
# make your own keys from dev.twitter.com
ckey=k.ckey
csecret=k.csecret
atoken=k.atoken
asecret=k.asecret

# tweepy listener 
class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        # gets only text from tweet data
        tweet = all_data["text"]
        # gets sentiment value and confidence from the trained model
        sent_value,confidence = s.sentiment(tweet)
        print(tweet,sent_value,confidence)

        # if confidence is greater than 85 write it to a file 
        if(confidence*100 >=85):
            output = open("twitter-out.txt","a")
            output.write(sent_value)
            output.write("\n")
            output.close()
        return True

    def on_error(self, status):
        print(status)

# tweepy stuff
auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
twitterStream = Stream(auth, listener())
twitterStream.filter(track=["trump"])
# ^tracks a specific keyword
