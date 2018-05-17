# Script for sentiment analysis on some messages

import nltk
import nltk.sentiment
nltk.download('vader_lexicon')

# analyzer = nltk.sentiment.sentiment_analyzer.SentimentAnalyzer()
nltk.sentiment.util.demo_vader_instance("greetings ! i the great mauve power ranger come to request of you one hat and two balls , so i can set off in my ventures to save the world !")
nltk.sentiment.util.demo_vader_instance("that won't work at all , i need the book and two hats or the book and the ball .")
nltk.sentiment.util.demo_vader_instance("okay then atleast give me one ball with one hat - its a fair deal come on")
nltk.sentiment.util.demo_vader_instance("that's a pretty unfair agreement in points distribution and quantity of items. ")
nltk.sentiment.util.demo_vader_instance("ok . . . so how about a counter offer instead of being rude ?")
nltk.sentiment.util.demo_vader_instance("good afternoon !")
nltk.sentiment.util.demo_vader_instance("hello there . so , i'd really love to get the hat .")
nltk.sentiment.util.demo_vader_instance("no can do buckerino")
nltk.sentiment.util.demo_vader_instance("greetings ! i the great mauve power ranger come to request of you one hat and two balls , so i can set off in my ventures to save the world !")