
def nlp_preprocessing(text):
    """
    This function applies preprocesses texts"
    
    Arg
    text (string): text to be worked on
    
    output
    (string): preprocessed text
    """ 
    # remove web links from text
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                  "", text.strip())
    
    # remove file directories from text
    text = re.sub(r"(/[a-zA-Z0-9_]+)+(/)*(.[a-zA-Z_]+)*",
                  "", text).strip()
    
    # remove deleted comments in text
    text = re.sub(r"\[deleted\]", "", text).strip()
    
    # remove english stopwords in texts
    text_list = text.split("|")    # split the group of comments into separate comments
    stop_words = set(stopwords.words('english'))    # get the stopwords for in English language
    for i in range(len(text_list)):    # loop over each comment and remove stopwords
        text_list[i] = " ".join([word for word in text_list[i].split() if word.lower() not in stop_words])
    text = " | ".join(text_list)
    
    # remove punctuations in text
    text_list = text.split("|")
    for i in range(len(text_list)):    # loop over each comment and remove punctuations
        text_list[i] = "". join([l if l not in string.punctuation else " " for l in text_list[i]])
    text = " | ".join(text_list)
    
    # remove non-alphabetic characters in text
    text_list = text.split("|")    # split the group of comments into separate comments
    for i in range(len(text_list)): # remove non-alphabetic characters from each comment
        text_list[i] = re.sub(r"[^a-zA-Z ]", "", text_list[i].strip())
    text = " | ".join(text_list)
    
    # reduce multiple adjacent spaces to a single space
    text = re.sub(r"(\s)+", " ", text).strip()
    
    # remove repeated comments
    text_list = text.split("|")    # split the group of comments into separate comments
    unique_comment = []    # create a variable to store unique comments found
    for comment in text_list: # loop over the comments store unique comments
        # if the current comment has been seen earlier or is empty
        # or just contains a single space
        if (comment.strip() in unique_comment) or (comment == "") or (comment == " "):
            # ignore the comment
            continue
        else:
            # if it has not been seen add it to the list of unique comments
            unique_comment.append(comment.strip())
    text = " | ".join(unique_comment)
    
    # convert all characters to lower case
    text = text.lower()
    
    # return preprocessed text
    return text

def get_prediction_per_comment(comment):
    """
    This function will make prediction on single comment
    """
    # preprocess the comment
    comment = nlp_preprocessing(comment)
    
    # vectorize the preprocessed comment
    X = vectorizer.transform([comment]).toarray()
    
    # get prediction from the stacking classifier in integer form
    y_pred = s_class.predict(X)
    
    # get the class name of the prediction
    y_pred_class = encoder.classes_[y_pred][0]
    
    # return the class name
    return y_pred_class

def get_overall_prediction(file, comment_header="comments", csv=True):
    """
    This function takes in a link to a csv file ordataframe and predict users as either medical doctor,veterinarian or other based on their comments
    
    Args
    df (csv file path or pandas dataframe): file path to csv or dataframe containing comments in column named 'comments', multiple comments separated with '|'
    comment_header (string): Name of the column containig the comments
    csv (bool): If True (default), indicates the file is a csv
                If False, indicates the file is not a csv
    
    
    Output
    (pandas dataframe): a copy of the datafrme containing the predictions in column named 'Predicted Label'
 """
    if csv:
        # create a dtaframe using the csv file
        dataset = pd.read_csv(file)
    else:
        # make a copy of the dataframe
        dataset = file.copy()
    
    # initialize predictions as empty string
    dataset.loc[:, "Predicted Label"] = ""
    
    # loop through each group of comments to make prediction
    for i in dataset.index:
        # extract comments of current user
        comments = dataset.loc[i, comment_header]
        
        # create variable to store the orediction of each comment by the current user
        predictions = np.array([])
        
        # loop through comments made by the user user and make prediction on each
        for comment in comments.split("|"):
            # get prediction for the current comment
            pred = get_prediction_per_comment(comment)
            
            # add the prediction to the list of predictions for comments made by current user
            predictions = np.append(predictions, pred)
            
            # if any of the prediction is Veterinarian, ignore the rest
            if any(predictions == "Veterinarian"):
                dataset.loc[i, "Predicted Label"] = "Veterinarian"
                
            # else if any of the prediction is Medical Doctor, ignore the rest
            elif any(predictions == "Medical Doctor"):
                dataset.loc[i, "Predicted Label"] = "Medical Doctor"
                
            # else predict the user as Other
            else:
                dataset.loc[i, "Predicted Label"] = "Other"
    
    # return the dataframe containing the predictions
    return dataset