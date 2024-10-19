#!/usr/bin/env python
# coding: utf-8

# # Programming Assignment 5: Embeddings! 
# 
# Can you write a program to answer quiz questions?  
# 
# Do you ever wish you could write a program to take quizzes or tests for you? In this assignment, you’ll do just that! In particular, you’ll leverage word embeddings to write a program that can answer various multiple choice and true/false quiz questions.

# # The Embeddings
# You’ll be using subset of ~4k 50-dimensional GloVe embeddings trained on Wikipedia articles. The GloVe (Global Vectors) model learns vector representations for words by looking at global word-word co-occurrence statistics in a body of text and learning vectors such that their dot product is proportional to the probability of the corresponding words co-occuring in a piece of text. The GloVe model was developed right here at Stanford, and if you’re curious you can read more about it [here](https://nlp.stanford.edu/projects/glove/)!

# In[1]:


# Do not modify this cell, please just run it!
import quizlet


# # Your Mission
# The assignment consists of five parts.

# # Part 1: Synonyms
# For this section, your goal is to answer questions of the form:
# 
# ```
#   What is a synonym for warrior?  
#   a) soldier  
#   b) sailor  
#   c) pirate  
#   d) spy  
# ```
# 
# You are given as input a word and a list of candidate choices. Your goal is to return the choice you think is the synonym. You’ll first implement two similarity metrics - euclidean distance and cosine similarity - then leverage them to answer the multiple choice questions!
# 
# Specifically, you will implement the following 4 functions:
# 
# * **euclidean_distance()**: calculate the euclidean distance between two vectors. Note: you’ll only use this metric in Part 1. For the rest of the assignment, you'll only use cosine similarity.
# * **cosine_similarity()**: calculate the cosine similarity between two vectors. You’ll be using this helper function throughout the other parts of the assignment as well, so you’ll want to get it right!
# * **find_synonym()**: given a word, a list of 4 candidate choices, and which similarity metric to use, return which word you think is the synonym! The function takes in `comparison_metric` as a parameter: if its value is `euc_dist`, you'll use Euclidean distance as the similarity metric; if its value is `cosine_sim`, you'll use cosine similarity as the metric.
# * **part1_written()**: you’ll find that finding synonyms with word embeddings works quite well, especially when using cosine similarity as the metric. However, it’s not perfect. In this function, you’ll look at a question that your `find_synonyms()` function (using cosine similarity) gets wrong, and answer why you think this might be the case. Please return your answer as a string in this function.
# 
# Note: for the rest of the assignment, you'll only use cosine similarity as the comparison metric. You won't use the euclidean distance function anymore.
# 
# 

# In[15]:


import numpy as np

def cosine_similarity(v1, v2):
    '''
    Calculates and returns the cosine similarity between vectors v1 and v2
    Arguments:
        v1 (np.array), v2 (np.array): vectors
    Returns:
        cosine_sim (float): the cosine similarity between v1, v2
    '''
  
    dot_product = np.dot(v1, v2)
    
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0 
    else:
        cosine_sim = dot_product / (norm_v1 * norm_v2)
    
    return cosine_sim
def euclidean_distance(v1, v2):
    '''
    Calculates and returns the euclidean distance between v1 and v2

    Arguments:
        v1 (np.array), v2 (np.array): vectors

    Returns:
        euclidean_dist (float): the euclidean distance between v1, v2
    '''
   
    difference = v1 - v2
    
    squared_difference = difference ** 2
    
    sum_squared_difference = np.sum(squared_difference)
    
    euclidean_dist = np.sqrt(sum_squared_difference)
    
    return euclidean_dist
                

def find_synonym(word, choices, embeddings, comparison_metric):
    '''
    Answer a multiple choice synonym question! Namely, given a word w 
    and list of candidate answers, find the word that is most similar to w.
    Similarity will be determined by either euclidean distance or cosine
    similarity, depending on what is passed in as the comparison_metric.

    Arguments:
        word (str): word
        choices (List[str]): list of candidate answers
        embeddings (Dict[str, np.array]): map of words to their embeddings
        comparison_metric (str): either 'euc_dist' or 'cosine_sim'. 
            This indicates which metric to use - either euclidean distance or cosine similarity.
            With euclidean distance, we want the word with the lowest euclidean distance.
            With cosine similarity, we want the word with the highest cosine similarity.

    Returns:
        answer (str): the word in choices most similar to the given word
    '''
    answer = None

    word_embedding = embeddings[word]

    if word_embedding is None:
        return answer 

    best_similarity = -np.inf if comparison_metric == 'cosine_sim' else np.inf

    for choice in choices:
        choice_embedding = embeddings[choice]
        
        if choice_embedding is None:
            continue  
        
        if comparison_metric == 'euc_dist':
            similarity = euclidean_distance(word_embedding, choice_embedding)
            if similarity < best_similarity:
                best_similarity = similarity
                answer = choice
        elif comparison_metric == 'cosine_sim':
            similarity = cosine_similarity(word_embedding, choice_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                answer = choice

    return answer

def part1_written():
    '''
    Finding synonyms using cosine similarity on word embeddings does fairly well!
    However, it's not perfect. In particular, you should see that it gets the last
    synonym quiz question wrong (the true answer would be positive):

    30. What is a synonym for sanguine?
        a) pessimistic
        b) unsure
        c) sad
        d) positive

    What word does it choose instead? In 1-2 sentences, explain why you think 
    it got the question wrong.
    
    See the cell below for the code to run for this part
    '''
    #########################################################
    ## TODO: replace string with your answer               ##
    ######################################################### 
    answer = """
    The model may have chosen a word like 'pessimistic', 'unsure', or 'sad' instead of 'positive' because it could not capture the subtle semantic differences between these words and 'sanguine'. 
        Additionally, the embeddings might not have been trained on a sufficiently diverse dataset to understand the positive connotation of 'sanguine' accurately.
    
    """
    #########################################################
    ## End TODO                                            ##
    ######################################################### 
    return answer


# In[16]:


"""
DO NOT CHANGE
otherwise autograder may fail to evaluate
"""

"""This will create a class to test the functions you implemented above. If you are curious,
you can see the code for this in quizlet.py but it is not required. If you run this cell,
we will load the test data for you and run it on your functions to test your implementation.

You should get an accuracy of 66% with euclidean distance and 83% with cosine distance
"""

part1 = quizlet.Part1_Runner(find_synonym, part1_written)
part1_euc, part1_cosine = part1.evaluate(True)  # To only print the scores, pass in False as an argument
score_a = 0
if part1_euc >= 0.66:
    score_a += 50
if part1_cosine >= 0.83:
    score_a += 50
(part1_euc, part1_cosine)


# # Part 2: Analogies
# For this section, your goal is to answer questions of the form:
# 
# ```
#   man is to king as woman is to ___?  
#   a) princess  
#   b) queen  
#   c) wife  
#   d) ruler   
# ```
# 
# Namely, you are trying to find the word `bb` that completes the analogy `a:b → aa:bb`. You will take as input three words, `a`, `b`, `aa`, and a list of candidate choices and return the choice you think completes the analogy.
# 
# One of the neat properties of embeddings is their ability to capture relational meanings. In fact, for the analogy **man:king → woman:queen** above, we have that the vector:
# 
# `vector('king') - vector('man') + vector('woman')`
# 
# is a vector close to  `vector('queen')`. Make sure that when completing these analogies, you are following the **same logical order** as the example above in order to align with our test set. You’ll leverage this pattern to try to answer the quizlet questions!
# 
# Specifically, you will implement the following function:
# 
# * **find_analogy_word()**: given a, b, and aa, find the best word in a list of candidate choices that completes the analogy (leveraging cosine similarity as your similarity metric).

# In[17]:


def find_analogy_word(a, b, aa, choices, embeddings):
    '''
    Find the word bb that completes the analogy: a:b -> aa:bb
    A classic example would be: man:king -> woman:queen

    Note: use cosine similarity as your similarity metric

    Arguments:
        a, b, aa (str): words in the analogy described above
        choices (List[str]): list of strings for possible answer
        embeddings (Dict[str, np.array]): map of words to their embeddings

    Returns:
        answer: the word bb that completes the analogy
    '''
    answer = None
    
   
    try:
        a_embedding = embeddings[a]
        b_embedding = embeddings[b]
        aa_embedding = embeddings[aa]
        
        analogy_vector = b_embedding - a_embedding + aa_embedding
    except KeyError:
        return answer 

    best_similarity = -np.inf 

    for choice in choices:
        try:
            choice_embedding = embeddings[choice]
            similarity = cosine_similarity(analogy_vector, choice_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                answer = choice
        except KeyError:
            continue  

    return answer


# In[18]:


"""
DO NOT CHANGE
otherwise autograder may fail to evaluate
"""
# You should get an accuracy of 64%

part2 = quizlet.Part2_Runner(find_analogy_word)
part2_auc = part2.evaluate(True) # To only print the scores, pass in False as an argument
part2_score = 0
if part2_auc >= 0.63999:
    part2_score = 100
score_b = int(part2_score * 0.5)
part2_auc


# # Part 3: Sentence Similarity
# For this section, your goal is to answer questions of the form:
# 
# ```
#     True/False: the following two sentences are semantically similar:
#       1. he later learned that the incident was caused by the concorde's sonic boom
#       2. he later found out the alarming incident had been caused by concorde's powerful sonic boom
# ```
# 
# Namely, you take in 2 sentences as input, and output either true or false (ie, a label of 1 or 0). To do this, you will create a sentence embedding that represents each sentence in vector form, then apply cosine similarity to compute the similarity between the two sentence embeddings. If they have a high enough similarity, you’ll guess "True" and otherwise "False".
# 
# To accomplish this, you’ll first turn each sentence into a single vector embedding. There are a few different ways you can do this. For this assignment, we’ll look at two approaches:
# 
# * **Simple sum**: Sum the word embeddings of each individual word in the sentence. This resulting vector is the sentence embedding vector.
# * **Sum with POS weighting**: Take a weighted sum of the individual word vectors, where the weighting depends on the part of speech (POS) of that given word. Each POS (ie, verb, noun, adjective, etc) has a different scalar weight associated with it. We multiply each word vector by the scalar weight associated with its part of speech, then sum these weighted vectors.
# 
# Specifically, you will implement the following 2 functions:
# 
# * **get_embedding()**: given a sentence (string), return the sentence embedding (vector). The function also takes in the parameter `use_POS`:
#     * if `use_POS` is false (regular case), leverage method 1 above - simply the sum of the word embeddings for each word in the sentence (ignoring words that don’t appear in our vocabulary).
#     * if `use_POS` is true, leverage method 2 - use a weighted sum, where we weight each word by a scalar that depends on its part of speech tag.
# * **get_similarity()**: given two sentences, find the cosine similarity between their corresponding sentence embeddings.
# 
# Helpful hints:
# 
# * We’ve given you a map `POS_weights` that maps part of speech tags to their associated weight. For example, `POS_weights['NN'] = 0.8` (where NN is the POS tag for noun).
# * You may skip words that either (1) are not in our embeddings or (2) have a POS tag that is not in `POS_weights` .
# * To get a list of all the words in the sentence, use nltk's word_tokenize function.
# 
#   ```
#   >>> sentence = "this is a sentence"
#   >>> word_tokens = word_tokenize(sentence)
#   >>> word_tokens
#   ['this', 'is', 'a', 'sentence']
#   ```
#   
# * To get the POS tags for each word in a sentence, you can use nltk.pos_tag. To use it, you provide a list of words in a sentence, and it returns a list of tuples, where the first element is the word and the second is its corresponding POS tag. **For this PA, make sure that you pass in the entire sentence to a single call to nltk.pos_tag; do not call  nltk.pos_tag separately on each word in the sentence.** This is because some words can be multiple parts of speech (for example, "back" can be a noun or a verb). Passing in the entire sentence allows for more context to figure out what POS tag a word should have.
# 
# ```
#     >>> tagged_words = nltk.pos_tag(word_tokens)
#     >>> tagged_words
#     [('this', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NN')]`
# ```

# In[19]:


"""
DO NOT CHANGE
"""
# You will use nltk for tokenizing and tagging!
import nltk
from nltk.tokenize import word_tokenize


# In[20]:


"""
DO NOT CHANGE
"""
# Run this cell to download the nltk tagger
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# In[23]:


def get_embedding(s, embeddings, use_POS=False, POS_weights=None):
    '''
    Returns vector embedding for a given sentence.

    Hint:
    - to get all the words in the sentence, you can use nltk's `word_tokenize` function
        >>> list_of_words = word_tokenize(sentence_string)
    - to get part of speech tags for words in a sentence, you can use `nltk.pos_tag`
        >>> tagged_tokens = nltk.pos_tag(list_of_words)
    - you can read more here: https://www.nltk.org/book/ch05.html

    Arguments:
        s (str): sentence
        embeddings (Dict[str, np.array]): map of words (strings) to their embeddings (np.array)
        use_POS (bool): flag indicating whether to use POS weightings when
            calculating the sentence embedding
        POS_weights (Dict[str, float]): map of part of speech tags (strings) to their weights (floats),
            it is only to be used if the use_POS flag is true

    Returns:
        embed (np.array): vector embedding of sentence s
    '''
    embed = np.zeros(embeddings.vector_size)  
    list_of_words = word_tokenize(s) 
    tagged_tokens = nltk.pos_tag(list_of_words) 

    for word, tag in tagged_tokens:
        if word in embeddings: 
            if use_POS and tag in POS_weights: 
                embed += embeddings[word] * POS_weights[tag]
            else:  
                embed += embeddings[word]

    return embed

def get_similarity(s1, s2, embeddings, use_POS, POS_weights=None):
    '''
    Given 2 sentences and the embeddings dictionary, convert the sentences
    into sentence embeddings and return the cosine similarity between them.

    Arguments:
        s1, s2 (str): sentences
        embeddings (Dict[str, np.array]): map of words (strings) to their embeddings (np.array)
        use_POS (bool): flag indicating whether to use POS weightings when
            calculating the sentence embedding
        POS_weights (Dict[str, float]): map of part of speech tags (strings) to their weights (floats),
            it is only to be used if the use_POS flag is true

    Returns:
        similarity (float): cosine similarity of the two sentence embeddings
    '''
   
    embed_s1 = get_embedding(s1, embeddings, use_POS, POS_weights)
    embed_s2 = get_embedding(s2, embeddings, use_POS, POS_weights)

    dot_product = np.dot(embed_s1, embed_s2)
    norm_s1 = np.linalg.norm(embed_s1)
    norm_s2 = np.linalg.norm(embed_s2)

    if norm_s1 == 0 or norm_s2 == 0:
        similarity = 0.0 
    else:
        similarity = dot_product / (norm_s1 * norm_s2)

    return similarity


# In[24]:


"""
DO NOT CHANGE
otherwise autograder may fail to evaluate
"""

# You should get an accuracy of 78% without POS weights, and 88% with.

part3 = quizlet.Part3_Runner(get_similarity)
part3_base, part3_pos = part3.evaluate(True) # To only print the scores, pass in False as an argument

part3_score = 0
if part3_base >= 0.7799999:
    part3_score += 50
if part3_pos >= 0.8799999:
    part3_score += 50

score_b = int(part2_score * 0.5 + part3_score * 0.5)

(part3_base, part3_pos)


# # Part 4: Exploration
# In this section, you'll do an exploration question. Specifically, you'll implement the following 2 functions:
# 
# * **occupation_exploration()**: given a list of occupations, find the top 5 occupations with the highest cosine similarity to the word "man", and the top 5 occupations with the highest cosine similarity to the word "woman".
# * **part4_written()**: look at your results from the previous exploration task. What do you observe, and why do you think this might be the case? Write your answer within the function by returning a string.
# 

# In[27]:


def occupation_exploration(occupations, embeddings):
    '''
    Given a list of occupations, return the 5 occupations that are closest
    to 'man', and the 5 closest to 'woman', using cosine similarity between
    corresponding word embeddings as a measure of similarity.

    Arguments:
        occupations (List[str]): list of occupations
        embeddings (Dict[str, np.array]): map of words (strings) to their embeddings (np.array)

    Returns:
        top_man_occs (List[str]): list of 5 occupations closest to 'man'
        top_woman_occs (List[str]): list of 5 occuptions closest to 'woman'
            note: both lists should be sorted, with the occupation with highest
                  cosine similarity first in the list
    '''
    top_man_occs = []
    top_woman_occs = []
   
    man_embedding = embeddings['man']
    woman_embedding = embeddings['woman']

    if man_embedding is not None:
        man_similarities = []
        for occupation in occupations:
            if occupation in embeddings:
                occupation_embedding = embeddings[occupation]
                similarity = cosine_similarity(man_embedding, occupation_embedding)
                man_similarities.append((occupation, similarity))
        top_man_occs = sorted(man_similarities, key=lambda x: x[1], reverse=True)[:5]
    
    if woman_embedding is not None:
        woman_similarities = []
        for occupation in occupations:
            if occupation in embeddings:
                occupation_embedding = embeddings[occupation]
                similarity = cosine_similarity(woman_embedding, occupation_embedding)
                woman_similarities.append((occupation, similarity))
        top_woman_occs = sorted(woman_similarities, key=lambda x: x[1], reverse=True)[:5]

    return [occ[0] for occ in top_man_occs], [occ[0] for occ in top_woman_occs]

def part4_written():
    '''
    Take a look at what occupations you found are closest to 'man' and
    closest to 'woman'. Do you notice anything curious? In 1-2 sentences,
    describe what you find, and why you think this occurs.
    '''
    #########################################################
    ## TODO: replace string with your answer               ##
    ######################################################### 
    answer = """
   ghvjhebfk
    """
    #########################################################
    ## End TODO                                            ##
    ######################################################### 
    return answer


# In[28]:


"""
DO NOT CHANGE
otherwise autograder may fail to evaluate
"""

part4 = quizlet.Part4_Runner(occupation_exploration, part4_written)
part4_man, part4_woman = part4.evaluate()

expected_men = {'teacher', 'actor', 'worker', 'lawyer', 'warrior'}
expected_women = {'nurse', 'teacher', 'worker', 'maid', 'waitress'}

part4_score = 0
if len(expected_men - set(part4_man)) == 0:
    part4_score += 50
if len(expected_women - set(part4_woman)) == 0:
    part4_score += 50

score_c = int(part4_score * 0.5)

(part4_man, part4_woman)


# # Part 5: Entity Representation
# 
# Entities can be detected in text using a Named Entity Recognition (NER) Model. Luckily many such models exist and can be employed off-the-shelf with decent performance. In this section, we will take a corpus of documents from Wikipedia and extract named entities from them by using the NER model from SpaCy.
# 
# Once we've extracted entities, we might be interested in finding which ones are similar to each other. 
# 
# Like any other words, entities have have vector representations. However, since we are working with a fixed vocabulary, it is likely that we won't find all our entities in the vocabulary.
#     
# One simple way to create entity representations is to take the mean of the text description of the entity. In this assignment, for each entity we have a description from Wikipedia. You will implement a function that computes an entity representation by taking the mean of the embeddings of the description. Note that not all the words in the description are in our vocabulary, so skip those. Also, using embeddings of stop words might add noise to averaged embeddings, so let's skip those words as well. _Note: SpaCy token objects have a_ `Token.is_stop` _field that you can use for this._
# 
# Once we've computed embeddings for each entity, we might be interested in finding entities that are similar to each other. For each entity, let's find the top 5 most similar entities. _Note: For fast computation, you might want to vectorize your cosine similarity computation._
# 
# We have a dataset of annotated similar entities. Let's see how well we do on this benchmark and then let's visually inspect our similar entities to see how coherent they seem. 
# 
# _Question: Do you see any patterns in entities that are similar? Do you see any systematic mistakes?_
# 

# In[29]:


# You will use SpaCy to extact Named Entities
import spacy


# In[30]:


"""
Note: if you don't have installed en_core_web_sm uncomment the below line to install.
But after you have installed comment it again to make autograder work.
"""
# !python -m spacy download en_core_web_sm


# In[31]:


import en_core_web_sm
nlp = en_core_web_sm.load()


# In[35]:


def extract_named_entities(paragraph):
    '''
    This function will be fed 1  paragraph at a time.
    See the documentation for using the SpaCy API (https://spacy.io/) to convert 
    the paragraphs to Spacy Documents. The processing automatically runs an NER model.
    You should be able to use a simple field on the Document object to return
    the  entities in the paragraph.
    https://spacy.io/usage/linguistic-features#named-entities
    Note: nlp is defined in the previous cell.
    
    Arguments:
        paragraph (str): The paragraph of text from Wikipedia
        
    Returns:
        named_entities (List[str]): A list with the named entities for each token in the paragraph
        spacy_paragraph (spacy.Doc): The paragraph after it has been converted to a Spacy Document
    '''
    spacy_paragraph = nlp(paragraph)
    named_entities = [ent for ent in spacy_paragraph.ents]
    
    return named_entities, spacy_paragraph
    
def compute_entity_representation(description, embeddings):
    '''
    For each entity, we will use the description to build an entity representation.
    Use the embeddings as before to get an emebedding for each word in the description.
    Take the mean of all words from the description that appear in the vocabulary and 
    that are NOT stop words.
    Note: you want to make sure to use use lowercase tokens to lookup the embddings.
    Note: the description is a Spacy.Doc so access the raw text of its tokens with Token.text.
          (See here for an example: https://spacy.io/usage/spacy-101#annotations-token)
    Note: SpaCy Tokens have a Token.is_stop field.
    
    Arguments:
        description (spacy.Doc): the description used to build the entity representation
        embeddings (Dict[str, np.array]): map of words (strings) to their embeddings (np.array)
        
    Returns:
        vector (np.array): The entity representation created by taking the mean of the words in
                           the description that are in the embeddings map and NOT stop words.
    '''
    
    embeddings_list = []
    
    for token in description:
        word = token.text.lower()
        if not token.is_stop and word in embeddings:
            embeddings_list.append(embeddings[word])
    
    if embeddings_list:
        vector = np.mean(embeddings_list, axis=0)
    else:
        vector = np.zeros((50,))  

    return vector
    
def get_top_k_similar(entity_embedding, choices, top_k):
    '''
    For this function, you are given the embedding for an entity,
    and a dictionary of choices for similar entities ({entity_name: entity_embedding}). 
    Return the top k (string) entities that are most similar to the provided entity.
    This function will be called many times and compared with many entities, so you
    shouldn't use the cosine_similarity function you defined earlier. Instead, you should
    parallelize the computation of cosine similarity using a numpy matrix multiplication
    (np.matmul).
    
    Arguments:
        entity_embedding (np.array): the vector embedding for the entity
        choices (Dict[str, np.array]): map from entity (str) to embedding (np.array)
                                          with the choices for the top k most similar entities
        top_k (int): the number of top similar entities to return

    Returns:
        similar_entities (List[str]): this should be of size top_k
    
    '''
    choice_names = list(choices.keys())
    choice_embeddings = np.array(list(choices.values()))

    
    norm_entity = entity_embedding / np.linalg.norm(entity_embedding)
    norm_choices = choice_embeddings / np.linalg.norm(choice_embeddings, axis=1, keepdims=True)

    similarities = np.matmul(norm_choices, norm_entity)
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    similar_entities = [choice_names[idx] for idx in top_k_indices]

    return similar_entities


# In[36]:


"""
DO NOT CHANGE
otherwise autograder may fail to evaluate
"""
def compute_entity_similarity(entity_representation_1, entity_representation_2):
    '''
    Computes whether two entities are similar. Since we are doing a binary
    classification task, we choose a threshold to convert from a 
    cosine similarity score to a boolean value. This function will not be
    called many times and compared with many entities, so it's OK to use
    your original, not parallelized, cosine_similarity function.
    
    Arguments:
        entity_representation_1, entity_representation_2 (np.array): entity representations
        
    Returns:
        similar (bool): True if the the representations are sufficiently similar.
    '''
    threshold = 0.97
    dist = cosine_similarity(entity_representation_1, entity_representation_2)
    return dist > threshold


# In[37]:


part5 = quizlet.Part5_Runner(extract_named_entities, compute_entity_representation,
                            compute_entity_similarity, get_top_k_similar)

# This part may take >10 seconds to complete
part5.build_representations()

# You should have found over 2,000 unique entity mentions


# In[38]:


# This is just overwriting the function in the runner class so you can run this 
# cell many times to test get_top_k_similar implementations without having to 
# re-build the representations
part5.get_top_k_similar = get_top_k_similar
part5_acc, part5_top = part5.evaluate(True)  # Pass in false to not print out top k similar entities
part5_score = 0
if part5_acc >= .88:
    part5_score += 50
if part5_top >= .14:
    part5_score += 50

score_c = int(part4_score * 0.5 + part5_score * 0.5)

(part5_acc, part5_top)
# You should be able to get above 88% accuracy on the binary entity similarity task
# You should be able to  get above 14% on the top 5 similar entities task.


# ## Follow up
# You may be surprised by the low accuracy on the top k task, but when you look at the output, you should see that the similar entities are reasonable. The dataset we are using wasn't designed to rank similar entities, rather only to do the binary  classification task. In this case our ground truth is quite noisy so the quantitative results may be misleading. 

# Once you're ready to submit, you can run the cell below to prepare and zip up your solution:
