from __future__ import print_function
import os
import re
import json
import time
import numpy as np
from datetime import datetime
import unicodedata
import concurrent.futures
import csv
import pandas as pd
import transformers as ppb
import torch
import torch.nn as nn
# # import BERT-base pretrained model
from transformers import DistilBertModel, BertTokenizerFast

def max_len_each(each):
    aid = each[0]
    article_dict = each[1]['article']
    comment_dict = each[1]['comment']
    length_dic = {}
    length_dic[aid] = {'title': len( article_dict['title'].split()) if len(article_dict['title']) > 0 else 0,
                       'content' : len( article_dict['content'].split()) if len(article_dict['content']) > 0 else 0,                       
                      }
    list_comment_len = []
    for cid, c_value in comment_dict.items():
        if c_value['time'] != 'N' and c_value['content']:
            list_comment_len.append( len( c_value['content'].split()) if len(c_value['content']) > 0 else 0 )
                
    length_dic[aid]['comment'] = list_comment_len
    return (aid,length_dic)

def calc_lengths(article_comment_dict):
    title_len = []
    content_len = []
    comment_len = []
    # Create a pool of processes. By default, one is created for each CPU in your machine.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for each,output in zip( article_comment_dict.items(), 
            executor.map( max_len_each, article_comment_dict.items())):
            
            aid = output[0]
            value = output[1][aid]
            title_len.append( value['title'])
            content_len.append( value['content'])
            comment_len.extend( value['comment'])
    
    return title_len, content_len, comment_len

def cal_optimum_lengths(article_comment_dict):
    title_len, content_len, comment_len = calc_lengths( article_comment_dict)
    # 90th quantile length
    MAX_LEN_COMMENT = int(np.quantile(comment_len,0.9))
    # 90th quantile length
    MAX_LEN_TITLE = int(np.quantile(title_len,0.9))
    return MAX_LEN_TITLE, MAX_LEN_COMMENT

def create_art_dict(each):
    aid = each[0]
    article_dict = each[1]['article']
    pure_article_dict ={}
    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    pure_article_dict[aid] = {'time': article_dict['time'],
                              'title': article_dict['title'],
                              't_bert': tokenizer.encode( text= article_dict['title'], #text_preprocessing(sent),  # Preprocess sentence
                                                                add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                                                                max_length=MAX_LENGTH_TITLE,                  # Max length to truncate/pad
                                                                padding = 'max_length', # Pad sentence to max length
                                                                truncation = True
                                                                ) if len(article_dict['title']) > 0 else [],
                              'a_bert': tokenizer.encode( text= article_dict['content'], #text_preprocessing(sent),  # Preprocess sentence
                                                                add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                                                                max_length=MAX_LENGTH_TITLE,                  # Max length to truncate/pad
                                                                padding = 'max_length', # Pad sentence to max length
                                                                #return_tensors='pt',           # Return PyTorch tensor
                                                                #return_attention_mask=True,      # Return attention mask
                                                                truncation = True
                                                                ) if len(article_dict['content']) > 0 else [],
                              'topic': article_dict['topic'], 
                              'outlet': article_dict['outlet'],
                              'category': article_dict['category']}
    return (aid, pure_article_dict[aid])


def bert_article_encode( article_comment_dict):
    pure_article_dict = {}
    
    #######PURE ARTICLE BERT creation##############
    # Create a pool of processes. By default, one is created for each CPU in your machine.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for each,output in zip( article_comment_dict.items(), 
            executor.map( create_art_dict, article_comment_dict.items())):            
            aid = output[0]
            value = output[1]
            pure_article_dict[aid] = value
    ########################################################
    return pure_article_dict

def create_comm_dict(each):
    aid = each[0]
    comment_dict = each[1]['comment']
     # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    pure_comment_dict = {}    
    for cid, c_value in comment_dict.items():
        if c_value['time'] != 'N' and c_value['content']:
            pure_comment_dict[cid] = {'time': c_value['time'], 
                                      'article_id': aid,
                                      'author': unicodedata.normalize('NFD', c_value['author']),
                                      'com_bert': tokenizer.encode( text= c_value['content'],  # Preprocess sentence
                                                                add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                                                                max_length=MAX_LENGTH_COMMENT,  # Max length to truncate/pad
                                                                padding = 'max_length', # Pad sentence to max length
                                                                truncation = True
                                                                ) if len(c_value['content']) > 0 else[],                                      
                                      'content': c_value['content'], 
                                      'pid': c_value['pid']}
#             print('pure_comment_dict: ',pure_comment_dict)
    return pure_comment_dict

def bert_comment_encode( article_comment_dict):
    pure_comment_dict = {}    
    ########PURE COMMENT BERT creation##############
    # Create a pool of processes. By default, one is created for each CPU in your machine.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for each,output in zip( article_comment_dict.items(), 
            executor.map( create_comm_dict, article_comment_dict.items())):

            pure_comment_dict.update(output)
    ########################################################
    return pure_comment_dict

def create_pure_art(article_comment_dict, output_folder):
    pure_article_dict = bert_article_encode( article_comment_dict )
    target_file = os.path.join(output_folder, 'shortbert_inputs/pure_article_bert.json')
    json.dump(pure_article_dict, open(target_file, 'w'))
#     print(pure_article_dict)

def create_pure_comment(article_comment_dict, output_folder):
    pure_comment_dict = bert_comment_encode( article_comment_dict )
    target_file = os.path.join(output_folder, 'shortbert_inputs/pure_comment_bert.json')
    json.dump(pure_comment_dict, open(target_file, 'w'))

def create_articleidx_micro(each):    
    aid = each[0]
    article_dict = each[1]
    article_idx_micro = {}
    article_idx_micro[aid] = { 'a_bert': article_dict['a_bert'],
                              't_bert' : article_dict['t_bert'],
                               'topic' : article_dict['topic']
                             }
    return article_idx_micro

def create_articleidx_bert( data_folder,output_folder, pure_article_bert_dict, vocab_topic_dict):
    article_idx_bert_dict = {}
    #create required files
    # Create a pool of processes. By default, one is created for each CPU in your machine.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for each,output in zip( pure_article_bert_dict.items(), 
            executor.map( create_articleidx_micro, pure_article_bert_dict.items())):
            for result in output.items():
                aid, value = result[0], result[1]
                article_idx_bert_dict[ aid] = value
                topic_key = article_idx_bert_dict[ aid]['topic']
                article_idx_bert_dict[ aid]['topic'] = vocab_topic_dict[ topic_key]
    return article_idx_bert_dict

def create_articleidx_final(data_folder,output_folder):
    #Read reference files
    pure_article_bert_file =  os.path.join(output_folder, 'shortbert_inputs/pure_article_bert.json')
    pure_article_bert_dict = json.load(open( pure_article_bert_file))
    
    vocab_topic_file = os.path.join( data_folder,'vocab.topic')
    vocab_topic_dict = {}    
    with open(vocab_topic_file) as vocab_topic:                                                                                          
        topic_reader = csv.reader( vocab_topic, delimiter='\t' )
        for topic in topic_reader:
            vocab_topic_dict[topic[0]] = topic[1]

    article_idx_bert_dict_final = create_articleidx_bert( data_folder, output_folder, pure_article_bert_dict,
                                                         vocab_topic_dict)    
    #save in dataframe and then to json to optimize size
    df_article_idx = pd.DataFrame.from_dict(article_idx_bert_dict_final,orient = 'index')
    df_article_idx.index.rename('aid',inplace=True)
    #dump into json files
    article_idx_bert_file =  os.path.join(output_folder, 'shortbert_inputs/article_idx_bert.json')
    df_article_idx.to_json(article_idx_bert_file, orient='columns')

def create_emocom_micro(each):
    cid = each[0]
    comment_dict = each[1]
    emocom_micro = {}
    emocom_micro[ cid ] = {'com_bert' : comment_dict['com_bert'],
                           'pid' : comment_dict['pid'],
                           'aid' : comment_dict['article_id']
                          }
    return emocom_micro

def create_comment_emotion( data_folder,output_folder, pure_comment_bert_dict, emotion_comment_olddict):
    emotion_comm_bert = {}
    
    # Create a pool of processes. By default, one is created for each CPU in your machine.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for each,output in zip( pure_comment_bert_dict.items(), 
            executor.map( create_emocom_micro, pure_comment_bert_dict.items())):            
            for result in output.items(): 
                cid,value = result[0], result[1]
                emotion_comm_bert[cid] = value 
                emotion_comm_bert[cid]['sentiment'] = emotion_comment_olddict[ cid]['sentiment'],
                emotion_comm_bert[cid]['emotion'] = emotion_comment_olddict[ cid]['emotion']          
    return emotion_comm_bert

def create_file_comment_emotion(data_folder,output_folder):
    #read reference files
    pure_comment_bert_file =  os.path.join( output_folder, 'shortbert_inputs/pure_comment_bert.json')
    pure_comment_bert_dict = json.load(open( pure_comment_bert_file))                               
    emotion_comment_oldfile = os.path.join( data_folder, 'sentiment_emotion_comment.json')
    emotion_comment_olddict = json.load( open( emotion_comment_oldfile))

    comment_emo_sent_bert_dict = create_comment_emotion( data_folder,output_folder, 
                                                        pure_comment_bert_dict, emotion_comment_olddict)
    #save in dataframe and then to json to optimize size
    df_comment_emo = pd.DataFrame.from_dict( comment_emo_sent_bert_dict,orient = 'index')
    df_comment_emo.index.rename('cid',inplace=True)
    #Extract sentiment dictionary to separate columns
    df_comment_emo['sentiment'] = df_comment_emo['sentiment'].apply(lambda x:x[0])
    df_comment_emo[[ 'vader', 'flair','blob_sentiment','blob_subjective' ]] = df_comment_emo['sentiment'].apply(pd.Series)
    df_comment_emo.drop(columns=['sentiment'], inplace=True)
    
    #dump into json files
    senti_emo_comment_bert_file =  os.path.join(output_folder, 'shortbert_inputs/senti_emo_comment_bert.json')
    df_comment_emo.to_json(senti_emo_comment_bert_file, orient='columns')

def main():
    #Changing filepaths
    print('current directory: ', os.getcwd() )
    ## NEW LAB PATH
    output_folder = '/home/kishore/kishore_data/outlets'
    data_folder = '/home/kishore/fan_backup/Old_code/news/outlets'
    # ##SABINE PATH 
    # output_folder = '/project/mukherjee/kishore/news_code/output'
    # data_folder = '/project/mukherjee/kishore/news_code/outlets'
    outlet = 'NewYorkTimes'
    output_folder = os.path.join( output_folder, outlet)
    data_folder = os.path.join( data_folder, outlet)
    os.chdir(output_folder)
    print('New directory: ', os.getcwd() )

    #Read input files
    article_comment_file = os.path.join(data_folder, 'article_comment.json')
    article_comment_dict = json.load(open(article_comment_file))    
    
    ##MAX LENGTH for articles and comments
    global MAX_LENGTH_TITLE, MAX_LENGTH_COMMENT
    MAX_LENGTH_TITLE, MAX_LENGTH_COMMENT = cal_optimum_lengths(article_comment_dict)

    ##Create intermediate files PURE_ARTICLE and PURE_COMMENT
    # create_pure_art(article_comment_dict, output_folder)
    # create_pure_comment(article_comment_dict, output_folder)

    ##Create article_idx_bert and senti_emo_comment_bert files
    # create_articleidx_final(data_folder,output_folder)
    # create_file_comment_emotion(data_folder,output_folder)

if __name__ == '__main__':
    main()