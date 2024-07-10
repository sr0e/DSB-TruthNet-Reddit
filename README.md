# TruthNet: Developing a Natural Language Model for Fake News Detection

The ability to recognize factual news from misinformation is crucial. The objective of this project is to develop a natural language processing model able to effectively predict whether a reddit post is from the subreddit of World News or from a satirical news subreddit, The Onion, based on its textual content. 

|Feature|Type|Description|
|---|---|---|
|create_utc| float | Unique identifier for each post|
|title| object| Title of reddit news post|
|subreddit|object|Name of subreddit|
|is_onion| int| Binarized subreddit column: 0=World News, 1=The Onion|
|sentiment| float|Polarity Score for each Title|


## **Executive Summary**

### **Introduction**

The rise of technology and the subsequent spread of information on the internet has made it challenging for users to discriminate between factual news and satire. Misinformation can have grave consequences, affecting public perception and general trust in media sources. Therefore, creating a classification model that identifies real from ‘fake’ news based on its textual content can assist in combating the spread of misinformation. In this project, optimizing for recall was the most important as it would reduce the amount of false negatives and prevent any satirical posts being misclassified as factual news. 

### **Method**

To analyze the data, python was used with a multitude of imported libraries: pandas, numpy, matplotlib, sklearn (model selection, pipeline, linear  model, compose, ensemble, metrics), and nltk (corpus, sentiment). The Praw API was used to scrape both r/TheOnion and r/worldnews subreddits. Both subreddits were scraped 5 times, collecting the newest 1000 post ID, post title, and subreddit name.  

#### **Data Cleaning**

As the data was coming from reddit, there wasn’t much cleaning to do. Most of the posts that were collected didn’t have any subtext, so the only columns that were analyzed were the title and subreddit. After each scrape, the duplicate posts were dropped so that only unique posts were in the data frame. An additional column was added that binarized the subreddits into 1 (The Onion) and 0 (World News). In looking at the most common words, “onion”, “worldnews”, “thread”, and “live” showed up as often as other common stop words. They were added to the list of stop words labeled ‘sw’. 

#### **Instantiating Tf-IDF + Random Forest Classifier** 

A pipeline with a Tfidf Vectorizer (stop words = sw) and Random Forest Classifier (n_estimators = 300, oob_score=True) was fitted on the training data. Next the specific parameters of Tf-IDF ngram range [(1, 1), (2, 2)], stop words [None, ‘english’, sw], Random Forest max features (np.arange(1, 21)), and max depth [None, 1, 2, 3, 4] was again fitted on the training data. A grid search was performed to find the best parameters for the training data. 

#### **Instantiating Count Vectorizer + Logistic Regression + Sentiment Analysis**

##### **Sentiment Analysis**

A new column was added to the dataframe named ‘sentiment’ that contained the polarity scores of each title. This column was used in addition to the ‘title’ column for the following analysis.

##### **Logistic Regression**
A pipeline with a preprocessor (Count Vectorizer (stop words = sw)) and Logistic Regression (max_iter=1000) was fitted on the training data. Next, a parameter grid was set up with two dictionaries of parameters. Both contained count vectorizer stopwords [None, ‘english’, sw], ngram range [(1, 1), (2, 2)], min df [1, 2, 3, 4] and a logistic regression penalty parameter. One contained the l2 penalty with the logistic regression C parameter (np.logspace(-2, 1, 100)), the second contained a logistic regression penalty of ‘None’ without the C parameter. A penalty of None ignores the C parameter and causes fittings to fail. A grid search was performed to find the best parameters for the training data. 

### **Results**

#### **Baseline**

There was a class imbalance of 59% World News to The Onion. 

Both the developed Random Forest model and Logistic Regression model showed promise in distinguishing between r/worldnews and r/TheOnion.

#### **Tf-IDF + Random Forest Classifier**

The best parameters were Tf-IDF stop words of ‘sw’, Tf-IDF ngram-range of (1, 1), a Random Forest max depth of None and a Random Forest max features of 1. With this, the model had an accuracy score of 0.934 and a recall of 0.92. It committed 37 Type II (false negative) errors and 33 Type I (false positive) errors. Compared to the baseline, this model performs significantly better than a random guess. It will correctly predict the class 93% of the time and identify the true positives 92% of the time. 

#### **Count Vectorizer (cvec) + Logistic Regression + Sentiment Analysis**

The best parameters were cvec stop words of ‘sw’, cvec ngram-range of (1,1), cvec minimum df of 1, Logistic Regression penalty of l2, and Logistic Regression C score of 1.072. With this, the model had an accuracy score of 0.942 and a recall of 0.94. It committed 29 Type II (false negative) errors and 33 Type I (false positive) errors. Compared to baseline, this model performs significantly better than a random guess. Compared to the Random Forest Classifier model, it performs slightly better on both accuracy and recall. It will correctly predict the class 94% of the time and identify the true positives 94% of the time.

### **Discussion/Conclusion/Next Steps**

In this project, we developed two natural language processing (NLP) models designed to differentiate between factual news collected from r/worldnews and satirical news collected from r/TheOnion. Using a sentiment analysis, count vectorizer and logistic regression, our best model successfully optimized for recall to minimize false negatives. This is crucial in the effort to combat misinformation online. Furthermore, this may enhance media literacy and promote responsible consumption of information in the digital age.

Additionally, while the models both performed better than baseline, our best model still failed at accurately classifying the news 6% of the time. This reflects a broader societal challenge that we find ourselves at a moment in time where the divide between satire and reality have become increasingly blurred. The rapid increase of online content, along with semantic and cultural nuances, has made this line more blurred. While it is nice to rely only on technology to assist in determining fact from fiction, it is imperative that users still develop and improve critical thinking skills. 

Moving forward, we would benefit from not only taking just the title of these news articles, but also delving into the actual text. This will assist in providing more context to the titles themselves. Exploring more advanced NLP methodologies will also further hone the models accuracy, recall and precision. Real-world deployment of this model will need continuous adaptation to grow with the ever evolving semantic and cultural nuances. 

This project underlines the vast potential of NLP in protecting information integrity and encouraging informed digital discourse. Ongoing research and development will be paramount to continue and cultivate the effectiveness of classification systems. 

#### **References**

reddit.com/r/TheOnion
reddit.com/r/worldnews

General Assembly DSB Unit 3 Lessons