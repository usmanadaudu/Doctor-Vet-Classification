
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