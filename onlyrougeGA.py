import random
from preprocess import preprocess_raw_sent
from preprocess import sim_with_title
from preprocess import sim_with_doc
from preprocess import sim_2_sent
from preprocess import count_noun
from copy import copy
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import nltk
import os.path
import statistics as sta
from rouge import Rouge
import re
import time
import os
import glob
from shutil import copyfile
import pandas as pd
import math
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

     
def load_a_doc(filename):
    file = open(filename, encoding='utf-8')
    article_text = file.read()
    file.close()
    return article_text   


def load_docs(directory):
	docs = list()  
	for name in os.listdir(directory):
		filename = directory + '/' + name
		doc = load_a_doc(filename)
		docs.append((doc, name))
	return docs

def clean_text(text):
    cleaned = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", ",", "'", "(", ")")).strip()
    check_text = "".join((item for item in cleaned if not item.isdigit())).strip()
    if len(check_text.split(" ")) < 4:
        return 'None'
    return text

def evaluate_rouge(raw_sentences, abstract):
    rouge_scores = []
    for sent in raw_sentences:
        try:
            rouge = Rouge()
            scores = rouge.get_scores(sent, abstract, avg=True)
            rouge1f = scores["rouge-1"]["f"]
        except Exception:
            rouge1f = 0 
        rouge_scores.append((sent, rouge1f))
    return rouge_scores
  

def start_run(processID, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, sub_stories, save_path, order_params):
   
    for example in sub_stories:
        start_time = time.time()
        raw_sents = re.split("\n\n", example[0])[1].split(' . ')
        title = re.split("\n\n", example[0])[0] 
        abstract = re.split("\n\n", example[0])[2]

        #remove too short sentences
        df = pd.DataFrame(raw_sents, columns =['raw'])
        df['preprocess_raw'] = df['raw'].apply(lambda x: clean_text(x))
        newdf = df.loc[(df['preprocess_raw'] != 'None')]
        raw_sentences = newdf['preprocess_raw'].values.tolist()
        if len(raw_sentences) == 0:
            continue

        preprocessed_sentences = []
        for raw_sent in raw_sentences:
            preprocessed_sent = preprocess_raw_sent(raw_sent)
            preprocessed_sentences.append(preprocessed_sent)

        preprocessed_abs_sentences_list = []
        raw_abs_sent_list = abstract.split(' . ')
        for abs_sent in raw_abs_sent_list:
            preprocessed_abs_sent = preprocess_raw_sent(abs_sent)
            preprocessed_abs_sentences_list.append(preprocessed_abs_sent)    
        preprocessed_abs_sentences = (" ").join(preprocessed_abs_sentences_list)  

        if len(preprocessed_sentences) < 7 or len(preprocessed_abs_sentences_list) < 3:
            continue

        rougeforsentences = evaluate_rouge(raw_sentences,abstract)
        rank_rouge = sorted(rougeforsentences, key=lambda x: x[1], reverse=True)

        length_of_summary = int(0.2*len(raw_sentences))
             
        print("Done preprocessing!")
        
        print('time for processing', time.time() - start_time)


        file_name = os.path.join(save_path, example[1] )    
        f = open(file_name,'w', encoding='utf-8')
        for i in range(length_of_summary):
            f.write(rank_rouge[i][0] + ' ')
        f.close()

    
def multiprocess(num_process, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, stories, save_path):
    processes = []
    n = math.floor(len(stories)/5)
    set_of_docs = [stories[i:i + n] for i in range(0, len(stories), n)] 
    for index, sub_stories in enumerate(set_of_docs):
        p = multiprocessing.Process(target=start_run, args=(
            index, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE,sub_stories, save_path[index], 0))
        processes.append(p)
        p.start()      
    for p in processes:
        p.join()



def main():
    # Setting Variables
    POPU_SIZE = 30
    MAX_GEN = 200
    CROSS_RATE = 0.8
    MUTATE_RATE = 0.4

    directory = 'full_text_data'
    save_path=['hyp1', 'hyp2', 'hyp3', 'hyp4', 'hyp5']

    if not os.path.exists('hyp1'):
        os.makedirs('hyp1')
    if not os.path.exists('hyp2'):
        os.makedirs('hyp2')
    if not os.path.exists('hyp3'):
        os.makedirs('hyp3')
    if not os.path.exists('hyp4'):
        os.makedirs('hyp4')
    if not os.path.exists('hyp5'):
        os.makedirs('hyp5')


    print("Setting: ")
    print("POPULATION SIZE: {}".format(POPU_SIZE))
    print("MAX NUMBER OF GENERATIONS: {}".format(MAX_GEN))
    print("CROSSING RATE: {}".format(CROSS_RATE))
    print("MUTATION SIZE: {}".format(MUTATE_RATE))

    # list of documents
    stories = load_docs(directory)
    start_time = time.time()
    
    multiprocess(5, POPU_SIZE, MAX_GEN, CROSS_RATE,
                 MUTATE_RATE, stories, save_path)
    # start_run(1, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, stories, save_path[0], 0)

    print("--- %s mins ---" % ((time.time() - start_time)/(60.0*len(stories))))

if __name__ == '__main__':
    main()  
        
        
     
    


    
    
    
    
        
            
            
         
