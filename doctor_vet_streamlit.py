# import modules
import re             # for regrex operations
import pickle
import numpy as np    # for mathematical calculations
import pandas as pd   # for working with structured data (dataframes)
import streamlit as st
from doctor_vet_module import get_overall_prediction
from doctor_vet_module import get_prediction_per_comment

# Title of the app
st.title("Doctor and Veterinary Classification")

# Display text
st.write("This app is to give little explanation on building a model which will correctly classify a number of given reddit users as practicing doctors, practicng veterinary or others based on each user's comments I did")

st.write("For more in-depth view of the model building processes, check the 'Doctor-vet-classification-model-bulding' notebook")

# Display text
st.write("The dataset for this task was sourced from a database whose link is given as")

st.code("[postgresql://niphemi.oyewole:W7bHIgaN1ejh@ep-delicate-river-a5cq94ee-pooler.us-east-2.aws.neon.tech/Vetassist?statusColor=F8F8F8&env=&name=redditors%20db&tLSMode=0&usePrivateKey=false&safeModeLevel=0&advancedSafeModeLevel=0&driverVersion=0&lazyload=false](postgresql://niphemi.oyewole:W7bHIgaN1ejh@ep-delicate-river-a5cq94ee-pooler.us-east-2.aws.neon.tech/Vetassist?statusColor=F8F8F8&env=&name=redditors%20db&tLSMode=0&usePrivateKey=false&safeModeLevel=0&advancedSafeModeLevel=0&driverVersion=0&lazyload=false)")

st.write("The link given raises error when trying to connect, so I used a modified version of the link shown below")

st.code("postgresql://niphemi.oyewole:endpoint=ep-delicate-river-a5cq94ee-pooler;W7bHIgaN1ejh@ep-delicate-river-a5cq94ee-pooler.us-east-2.aws.neon.tech/Vetassist?sslmode=allow")

# define the connection link to database
# conn_str = "postgresql://niphemi.oyewole:endpoint=ep-delicate-river-a5cq94ee-pooler;W7bHIgaN1ejh@ep-delicate-river-a5cq94ee-pooler.us-east-2.aws.neon.tech/Vetassist?sslmode=allow"

# create connection to the databse
# engine =  create_engine(conn_str)

# st.write("First, lets take a look at the tables in the database")

# define sql query for retrieving the tables in the database
sql_for_tables = """
SELECT
    table_schema || '.' || table_name
FROM
    information_schema.tables
WHERE
    table_type = 'BASE TABLE'
AND
    table_schema NOT IN ('pg_catalog', 'information_schema');
"""

# retrieve the tables in a dataframe
# tables_df = pd.read_sql_query(sql_for_tables, engine)
# st.write(tables_df)

st.write("""There are two tables in the database

Each table would be saved in a pandas dataframe""")

sql_for_table1 = """
SELECT
    *
FROM
    public.reddit_usernames_comments;
"""
# if the table cannot be rerieved from the database, use the csv file saved
# user_comment_df = pd.read_sql_query(sql_for_table1, engine)
user_comment_df = pd.read_csv("https://raw.githubusercontent.com/usmanadaudu/Doctor-Vet-Classification/main/reddit_usernames_comments.csv")

sql_for_table2 = """
SELECT
    *
FROM
    public.reddit_usernames;
"""
# if the table cannot be rerieved from the database, use the csv file saved
# user_info_df = pd.read_sql_query(sql_for_table2, engine)
user_info_df = pd.read_csv("https://raw.githubusercontent.com/usmanadaudu/Doctor-Vet-Classification/main/reddit_usernames.csv")

st.write("Lets take a look at the tables one after the other")
st.write("First Table")
st.write(user_comment_df.head())
st.write("Shape: ", user_comment_df.shape)
st.write("Second Table")
st.write(user_info_df.head())
st.write("Shape: ", user_info_df.shape)

st.header("Data Exploration")
st.write("""
This table (now dataframe) contains usernames of users and their comments

Lets look at a comment in order to understand how it is structured
""")
# print all comments by first user
st.write("comments by first user")
st.markdown(f"```\n{user_comment_df['comments'][0]}\n```")
# split comments into individual comments
first_comments = [user_comment_df["comments"][0].split("|")]

# print the number of comments for first user
st.write("Number of comments by first user: ", len(first_comments[0]))

# remove repeated comments
unique_comment = []
for comment in first_comments[0]:
    if comment in unique_comment:
        continue
    else:
        unique_comment.append(comment)
        
st.write("Length of unique comments for first user", len(unique_comment))
st.markdown(f"```\n{' | '.join(unique_comment)}\n```")

st.write("""It can be seen that the comment column contains multiple comments separated with "|"

It can also be seen that there are repeated comments
""")

st.write("checking for missing and duplicated values shows that there are neither missing nor duplicated values in the data")

st.write("Lets explore the second dataframe also")
st.write(user_info_df.head())
st.write("There are no missing values in this dataset also")

st.header("Data Preprocessing")
st.write("""
for the preprocessing, the various steps that would be done are:\n
1.  Removing web links\n
2.  emoving file directories\n
3.  Removing deleted comments indicated as'[deleted]'
4.  Removing stopwords. Stopwords are:""")
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
"you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its',
'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
st.code(stop_words, language="text")
st.write("""
5.  Removing punctuations""")
punctuations = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
st.write("""
6.  Removing non-alphabetic characters
7.  Reducing multiple adjacent spaces to a single space
8.  Reducing repeated comments to one
""")
def remove_web_link(text):
    """
    This function removes web links from texts
    
    Arg
    text (string): text to be worked on
    
    output
    (string): text where all web links (if any) have been removed
    """
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                  "", text.strip())
    return text

def remove_directories(text):
    """
    This function removes file directories from texts
    
    Arg
    text (string): text to be worked on
    
    output
    (string): text where all file directories (if any) have been removed
    """
    text = re.sub(r"(/[a-zA-Z0-9_]+)+(/)*(.[a-zA-Z_]+)*",
                  "", text).strip()
    return text

def remove_deleted_comments(text):
    """
    This function removes deleted comments indicated as "[deleted]"
    
    Arg
    text (string): text to be worked on
    
    output
    (string): text where all deleted comments (if any) have been removed
    """
    text = re.sub(r"\[deleted\]", "", text).strip()
    return text

def remove_stopwords(text):
    """
    This function removes stopwords from text"
    
    Arg
    text (string): text to be worked on
    
    output
    (string): text where all stopwords (if any) have been removed
    """
    # split the group of comments into separate comments
    text_list = text.split("|")
    
    # loop over each comment and remove any of the stopwords found
    for i in range(len(text_list)):
        text_list[i] = " ".join([word for word in text_list[i].split() if word.lower() not in stop_words])
        
    # merge the comments together using "|"
    return " | ".join(text_list)

def remove_punctuations(text):
    """
    This function removes punctuations from text"
    
    Arg
    text (string): text to be worked on
    
    output
    (string): text where all punctuations (if any) have been removed
    """
    # split the group of comments into separate comments
    text_list = text.split("|")
    
    # loop over each comment and remove any of the punctuations found    
    for i in range(len(text_list)):
        text_list[i] = "". join([l if l not in punctuations else " " for l in text_list[i]])
        
    # merge the comments together using "|"
    return " | ".join(text_list)

def remove_non_alphabets(text):
    """
    This function removes non-alphabetic characters from text"
    
    Arg
    text (string): text to be worked on
    
    output
    (string): text where all non-alphabetic characters (if any) have been removed
    """
    # split the group of comments into separate comments
    text_list = text.split("|")
    
    # loop over each comment and remove any of the non-alphabetic characters found 
    for i in range(len(text_list)):
        text_list[i] = re.sub(r"[^a-zA-Z ]", "", text_list[i].strip())
        
    # merge the comments together using "|"
    return " | ".join(text_list)

def remove_unneeded_spaces(text):
    """
    This function reduces multiple adjacent spaces to a single space in text"
    
    Arg
    text (string): text to be worked on
    
    output
    (string): text where all multiple spaces (if any) have been reduced to a single space
    """
    text = re.sub(r"(\s)+", " ", text).strip()
    return text

def remove_repeated_sentence(text):
    """
    This function removes repeated comments and preserves only the first one of them"
    
    Arg
    text (string): text to be worked on
    
    output
    (string): text where all multiple comments (if any) have been reduced to a single comment
    """    
    # split the group of comments into separate comments
    text_list = text.split("|")
    
    # create a variable to store unique comments found
    unique_comment = []
    
    # loop over the comments and only store the first of any kind of comment
    for comment in text_list:
        # if the current comment has been seen earlier or is empty
        # or just contains a single space
        if (comment.strip() in unique_comment) or (comment == "") or (comment == " "):
            # ignore the comment
            continue
        else:
            # if it has not been seen add it to the list of unique comments
            unique_comment.append(comment.strip())
        
    # merge the comments together using "|"
    return " | ".join(unique_comment)

def nlp_preprocessing(text):
    """
    This function applies preprocesses texts"
    
    Arg
    text (string): text to be worked on
    
    output
    (string): preprocessed text
    """ 
    # remove web links from text
    text = remove_web_link(text)
    
    # remove file directories from text
    text = remove_directories(text)
    
    # remove deleted comments in text
    text = remove_deleted_comments(text)
    
    # remove english stopwords in texts
    text = remove_stopwords(text)
    
    # remove punctuations in text
    text = remove_punctuations(text)
    
    # remove non-alphabetic characters in text
    text = remove_non_alphabets(text)
    
    # reduce multiple adjacent spaces to a single space
    text = remove_unneeded_spaces(text)
    
    # remove repeated comments
    text = remove_repeated_sentence(text)
    
    # convert all characters to lower case
    text = text.lower()
    
    # return preprocessed text
    return text



st.write("Lets take a look at the text below and how it would look like after preprocessing")
txt = """[deleted] | [deleted] | Got it. But why can I only select $1.99, $3.99 or $7.99 worth of MYST? Seems pretty strange imo. Why can’t we just send however much we like? Is there a way to just see our address and send whatever amount we choose? ---> /r/MysteriumNetwork/comments/zk6hag/how_to_send_myst_to_application/izy2cgw/ | You’re a legend bro. Wonder why tf they don’t make this accessible, seems like a no brainer! ---> /r/MysteriumNetwork/comments/zk6hag/how_to_send_myst_to_application/j0uh08x/ | Same problem here. WTF ---> /r/MysteriumNetwork/comments/zfm5ll/anyone_else_stuck_permanently_downloading_an/izxz9s6/ | UPDATE:

Wow so i fixed it guys. I deleted the DNS in my Wi-Fi settings and after a new one was generated I hit apply. Then all of the sudden I was back online! My question is what in the world caused that to happen? I love Mysterium but feel very sketched out, yet idk if it was even their fault. What do you guys think? I’m skeptical of reinstalling it again. I had a some MYST in my account prior to uninstalling, I doubt I would get it back if I reinstalled right? Not the end of the world, I’m just happy to be online again although I missed a lot of meetings this morning. Really curious how this happened after enabling the kill switch, quitting and uninstalling. Lmk what you think. ---> /r/MysteriumNetwork/comments/xx6kgk/need_help_ever_since_i_enabled_kill_switch_i_cant/iraks5t/ | Feels like I’ve won the lottery. Now back to my minimum wage misery. ---> /r/MysteriumNetwork/comments/xx6kgk/need_help_ever_since_i_enabled_kill_switch_i_cant/irbgtx8/ | I did, no reply and it’s been a week ---> /r/MysteriumNetwork/comments/txk7ao/cant_connect_to_any_nodes_for_more_than_30/i3vsj5p/ | Wow that's so sketchy.. wtf? Now I'm really not touching this project. ---> /r/MysteriumNetwork/comments/twpgt1/psa_moderators_are_censoring_posts_on/i3kenty/ | Okay good to know, I jumped on a Russian node and started thinking twice lol. You think it’s safer going with residential nodes or no difference? 🥂 ---> /r/MysteriumNetwork/comments/tsbqed/security/i2wtq00/ | YES. Any more of a liability compared to something like ExpressVPN or any other popular centralized vpn? Given the nature of both i feel like it would be more prevalent with dVPNs but really don’t know ---> /r/MysteriumNetwork/comments/tsbqed/security/i2wu0ju/ | Wow! Yeah seems the returns with Mysterium aren't great, I would have to enable whitelisting as well. Helium is a cool project. I wanted to buy a miner and antenna but all of the miners were sold out with a 6 month waiting period. I talked to this guy who does free installs but takes like 80% of the rewards which seemed pointless. Monero mining I don't know anything about. I don't have a mining rig either, just an iMac but I have no issue buying the equipment if the returns are steady ---> /r/MysteriumNetwork/comments/tijvgl/best_ways_to_earn_supplemental_income_aside_from/i1ervof/ |  Yeah I think I am. Do you have any tips? I would be considered a noob but have good internet speeds and hardware. What kind of returns do you get? What are your thoughts on enabling whitelisting? Is it worthy the drop in MYST? ---> /r/MysteriumNetwork/comments/thwllq/mysterium_vs_orchid/i1cbhz0/ | Lol fair enough. Hows the returns with presearch? ---> /r/MysteriumNetwork/comments/thwllq/mysterium_vs_orchid/i1cvidl/ | Yeah it's bad. What's up with that? I hope they're at least working on something. They have no presence at all it's creepy. ---> /r/MysteriumNetwork/comments/thwllq/mysterium_vs_orchid/i2nxv92/ | What type of returns do you see? Do you have whitelisting enabled? ---> /r/MysteriumNetwork/comments/thwllq/mysterium_vs_orchid/i1e5qgv/


Comment of user with index 306 [deleted]
View in your timezone:  
[March 21, 3 PM UTC][0]  

[0]: https://timee.io/20240321T1500?tl=%F0%9F%8F%86%20Win%20a%20free%20SenseCAP%20M4%20Square%20in%20our%20Myst%20Nodes%20x%20SenseCAP%20AMA!%20%F0%9F%93%85%20March%2021%20%40%203%20PM%20UTC%20%F0%9F%93%8DMysterium%20Network%20Discord.%20Set%20your%20reminder%20now.|View in your timezone:  
[18.12.2023 at 3 PM UTC][0]  [deleted]

[0]: https://timee.io/20231218T1500?tl=%5BAMA%5D%20Kryptex%20X%20MystNodes%20partnership.%20Earn%20crypto%20passive%20income%2C%20simply.%20Join%20us%20live%20on%2018.12.2023%20at%203%20PM%20UTC%20and%20ask%20your%20questions%20now!|View in your timezone:  
[23.08.2023 at 12 PM UTC][0]  [deleted]

[0]: https://timee.io/20230823T1200?tl=%5BAMA%5D%20Meet%20the%20Future%20of%20VPN!%20We're%20Savannah%20and%20Furkan%20from%20MysteriumVPN.%20The%20People-Powered%20Alternative%20with%20More%20IPs%20Than%20Many%20Legacy%20VPNs%20Combined.%20Join%20Us%20Live%20on%2023.08%20%40%2012%20PM%20UTC%20and%20Ask%20Your%20Questions%20Now!|View in your timezone:  
[23.08.2023 at 12 PM UTC][0][deleted]"""
st.code(txt, language="text")
st.write("After preprocessing")
st.code(nlp_preprocessing(txt), language="text")

st.header("Hand Engineering")
st.write("I labelled a number of samples by hand to form my training set")
st.write("The way I went about this was that I first merged the two tables together as shown below")
reddit_user_df = pd.merge(user_comment_df, user_info_df,
                          on="username", how="left")
st.write(reddit_user_df.head())
st.write("Then I splitted all groups of comments into separate comments")
reddit_user_df_processed = reddit_user_df.copy()
reddit_user_df_processed["comments"] = reddit_user_df["comments"].apply(nlp_preprocessing)

# create dictionary to store values for the new dataframe
user_separated_comment_dict = {
    "username" : [],
    "comment" : [],
    "subreddit" : [],
    "former_index" : []
}

# loop through the first 5 data and save each comment as a separate entry
for i in reddit_user_df_processed.index:
    for comment in reddit_user_df_processed.iloc[i]["comments"].split("|"):
        user_separated_comment_dict["username"].append(reddit_user_df_processed.iloc[i]["username"])
        user_separated_comment_dict["comment"].append(comment.strip())
        user_separated_comment_dict["subreddit"].append(reddit_user_df_processed.iloc[i]["subreddit"])
        user_separated_comment_dict["former_index"].append(i)
        
# convert the dictionary above to pandas dataframe
user_separated_comment_df = pd.DataFrame(user_separated_comment_dict)

st.write(user_separated_comment_df.head(10))
st.write("""
'former_index' which is the index of corresponding sample in the unprocessed dataframe is added in order to aid during labelling

Processed comments usually loose meaning or context when read by humans. Therefore the unprocessed form would be used for labelling
""")
st.write("Looking at the number comments in each of the subreddits as shown below")
st.write(user_separated_comment_df['subreddit'].value_counts())
st.write("I chose all the medicine, vet, HeliumNetwork and orchid subreddits because there are less than 30 comments in these subreddits. Then, I chose 30 comments in each of the MysteriumNetwork subreddit and Veterinary")

st.write("The first 5 entries in the training set after labelling are shown below")
train_set_df = pd.read_csv("train_set.csv")
st.write(train_set_df.head())

st.header("Model Building")
st.write("**My Approach to Building the Model**")
st.write("""
The following are the approaches used to solve this problem

1.  All users would be categorized as others unless proven otherwise from the comments
2.  Comments are independent of each other (meaning a comment is not continued in another comment)
3.  Comments made by a user would be splitted and considered separate data to capture the independence among comments
4.  When there is indication of user's category in a comment, other comments do not matter (i.e. when users state that they are doctors in a comment, even if other comment are not related to this, the user is still a doctor
5.  Any user automatically found from above to not be a doctor or veterinarian would be automatically classified as Others
""")

st.write("**Preprocessing**")
st.write("""Just before building my model, another round of preprocessing (apart from the ones above) was done. These are:
1.  Vectorization: The comments in each dataset were vectorized using 'TfidfVectorizer' 
2.  The target label was encoded using LabelEncoder as shown below""")
st.markdown(f"""```
.    Label       ->  Encoding \n
Medical Doctor  ->     0 \n
Other           ->     1 \n
Veterinarian    ->     2
\n```""")

st.write("The training set was splitted into 80% for training set and 20% for validation set")
# import the training dataset
data = pd.read_csv("train_set.csv")
# add the preprocessed comments as a new column in the dataframe
data["processed_comment"] = data["comment"].apply(nlp_preprocessing)
# initialize vactorizer for the comments
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)
    
with open("encoder.pkl", "rb") as file:
    encoder = pickle.load(file)
    
with open("doctor_vet_model.pkl", "rb") as file:
    model = pickle.load(file)
    
X = vectorizer.fit_transform(data["processed_comment"].values).toarray()
labels = data["Label"]
y = encoder.fit_transform(labels)

st.write("The various model built and their performances are summarized below")
st.write("**XGBoost**")
st.write("Predictions: ")
st.code("[1 0 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1]")

st.write("Accuracy: ", 0.75)

st.write("Classification Report")
st.code(f""".                precision    recall  f1-score   support

Medical Doctor       1.00      0.50      0.67         2
         Other       0.79      0.92      0.85        25
  Veterinarian       0.00      0.00      0.00         5

      accuracy                           0.75        32
     macro avg       0.60      0.47      0.51        32
  weighted avg       0.68      0.75      0.71        32""")

st.write("**MultinomialNB**")
st.write("Predictions: ")
st.code("[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]")
st.write("Accuracy: ", 0.78125)

st.write("Classification Report")
st.code(f""".                precision    recall  f1-score   support

Medical Doctor       0.00      0.00      0.00         2
         Other       0.78      1.00      0.88        25
  Veterinarian       0.00      0.00      0.00         5

      accuracy                           0.78        32
     macro avg       0.26      0.33      0.29        32
  weighted avg       0.61      0.78      0.69        32""")

st.write("**kNN Model**")
st.write("Predictions: ")
st.code("[1 0 1 1 1 1 1 1 1 0 1 1 0 1 1 1 0 1 1 0 0 1 1 2 1 1 1 1 1 1 1 1]")
st.write("Accuracy: ", 0.78125)

st.write("Classification Report")
st.code(f""".                precision    recall  f1-score   support

Medical Doctor       0.17      0.50      0.25         2
         Other       0.92      0.92      0.92        25
  Veterinarian       1.00      0.20      0.33         5

      accuracy                           0.78        32
     macro avg       0.70      0.54      0.50        32
  weighted avg       0.89      0.78      0.79        32""")

st.write("**AdaBoost Model**")
st.write("Predictions: ")
st.code("[1 0 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1]")
st.write("Accuracy: ", 0.75)

st.write("Classification Report")
st.code(f""".                precision    recall  f1-score   support

Medical Doctor       1.00      0.50      0.67         2
         Other       0.79      0.92      0.85        25
  Veterinarian       0.00      0.00      0.00         5

      accuracy                           0.75        32
     macro avg       0.60      0.47      0.51        32
  weighted avg       0.68      0.75      0.71        32""")

st.write("**Stacking (Combination of kNN and MultinomialNB)**")
st.write("Predictions: ")
st.code("[1 0 0 1 1 1 1 1 0 0 1 1 2 1 2 1 2 1 1 2 2 1 2 2 1 2 1 1 1 0 1 1]")
st.write("Accuracy: ", 0.71875)

st.write("Classification Report")
st.code(f""".                precision    recall  f1-score   support

Medical Doctor       0.20      0.50      0.29         2
         Other       0.95      0.72      0.82        25
  Veterinarian       0.50      0.80      0.62         5

      accuracy                           0.72        32
     macro avg       0.55      0.67      0.57        32
  weighted avg       0.83      0.72      0.75        32""")

st.write("This result is the best so far comparing the performance on all the classes")
st.write("Checking the classifcation of each model, It is seen that the stacking classifier which makes use of Multinomial Naive Bayes and kNN Classifier performed best")
st.write("It was also found that the performance of the Multinomial Model was extremely poor despite having high accuracy")
st.write("I will be going with the **Stacking Classifier**")

st.header("Making Predictions")
# get comment
user_comments = st.text_area('Enter the user comments here')

filepath = user_comments
comment_header=None
file_type = "text"



if st.button("Make Prediction", type="primary"):
    prediction = get_overall_prediction(filepath, np, pd, re, punctuations,
                                         stop_words, vectorizer, encoder,
                                         model, comment_header, file_type)
    st.success(f"User category: {prediction}")
    