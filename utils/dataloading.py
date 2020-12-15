
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import logging
import os
import json

class FinalDataset(Dataset):
    def __init__(self, output_folder,MAX_LEN_TITLE, MAX_LEN_COMMENT):
        #read reference files
        self.authors = json.load(open(os.path.join(output_folder, 'frequent_author_record.json')))
        self.topic_size = len(open(os.path.join(output_folder, 'vocab.topic')).readlines())

        article_idx_bert_file =  os.path.join(output_folder, 'shortbert_inputs/article_idx_bert.json');
        self.df_articles_idx = pd.read_json(article_idx_bert_file, orient='columns') 

        comment_emosent_bert_file =  os.path.join(output_folder, 'shortbert_inputs/senti_emo_comment_bert.json')
        self.df_comments_emo = pd.read_json(comment_emosent_bert_file, orient='columns')

        file_examples_pkl = os.path.join(output_folder, 'shortbert_inputs/examples.pkl')
        self.df_examples = pd.read_pickle(file_examples_pkl)
        
        self.authors_ar = self.df_examples['author'].unique()
        #create train, val, test data
        # self.MAX_LEN_TITLE, self.MAX_LEN_COMMENT, self.device = MAX_LEN_TITLE, MAX_LEN_COMMENT,device        
        self.MAX_LEN_TITLE, self.MAX_LEN_COMMENT = MAX_LEN_TITLE, MAX_LEN_COMMENT 
        
    def __len__(self):
        return len( self.df_examples)

    def __getitem__(self, idx):
        sr_track = self.df_examples.iloc[idx]
        
        #extract authors
        author = int(sr_track['author'])
        #extract selected comments dataframe
        sel_cid = sr_track.iloc[1:].values
        sel_cid = sel_cid.astype(int) 
        df_comments_sel = self.df_comments_emo.loc[ sel_cid]
        #extract read track 
        sel_aid = df_comments_sel['aid'].values
        df_articles_sel = self.df_articles_idx.loc[ sel_aid]
        ar_art_bert_token = df_articles_sel['t_bert'].tolist()        
        read_track = self.extract_art_track(ar_art_bert_token)

        #extract write track
        ar_com_bert_token = df_comments_sel['com_bert'].tolist()
        #combine article bert and comment bert along with token type ids and attention masks
        write_track = self.extract_combotrack(ar_art_bert_token,ar_com_bert_token )
        # write_track = self.extract_track(ar_com_bert_token)
        #extract sentiment and emotion
        emotion_track = df_comments_sel['emotion'].tolist()

        ar_vader = df_comments_sel['vader'].tolist()
        ar_flair = df_comments_sel['flair'].tolist()
        ar_blob_sentiment = df_comments_sel['blob_sentiment'].tolist()
        ar_blob_subjective = df_comments_sel['blob_subjective'].tolist()
        sentiment_track = [ar_vader, ar_flair, ar_blob_sentiment, ar_blob_subjective]

        #create tensors
        # author = torch.tensor([author], device = self.device)
        # read_track = torch.tensor([read_track], device = self.device)
        # write_track = torch.tensor([write_track], device = self.device)
        # sentiment_track = torch.tensor([sentiment_track], device = self.device)
        # emotion_track = torch.tensor([emotion_track], device = self.device, dtype = torch.float)
        
        author = torch.tensor([author])
        read_track = torch.tensor([read_track])
        write_track = torch.tensor([write_track])
        sentiment_track = torch.tensor([sentiment_track])
        emotion_track = torch.tensor([emotion_track], dtype = torch.float)
        
        return author,read_track, write_track, sentiment_track,emotion_track
    
    def my_collate(self, batch):
        author,read_track,write_track,sentiment_track,emotion_track = zip(*batch)
        return torch.cat(author), torch.cat(read_track), torch.cat(write_track), torch.cat(sentiment_track), torch.cat(emotion_track)

    #split dataset indices
    def datasplit(self, test_train_split=0.9, val_train_split=0.1, shuffle=False ):
        df_dataset = self.df_examples.copy()

        #create test split indices
        df_dataset.reset_index(inplace = True)
        df_result = df_dataset[['author','index']].groupby('author').agg(['count','min','max'])
        df_result.columns = df_result.columns.get_level_values(1)
        df_result['test_split'] = np.floor(df_result['count']*test_train_split).astype(int)
        # print('df_result: \n',df_result)
        #function to create indices for train, val and test sets
        def fun_split_indices(df_in, val_train_split):
            indices = list(range( df_in['min'], df_in['max']+1))
            train_indices, test_indices = indices[: df_in['test_split']], indices[ df_in['test_split']:]
            train_size = len(train_indices)
            validation_split = int(np.floor((1 - val_train_split) * train_size))
            train_indices, val_indices = train_indices[ : validation_split], train_indices[validation_split:]
            df_in['train_indices'] = train_indices
            df_in['val_indices'] = val_indices
            df_in['test_indices'] = test_indices
            return df_in

        df_result = df_result.apply( fun_split_indices,args =(val_train_split,), axis = 1)

        modes = ['train_indices', 'val_indices', 'test_indices']
        self.dic_indices ={}
        # dic_indices ={}
        for each in modes:            
            ls_indices = df_result[each].tolist()
            # print('each: ', ls_indices)
            ls_indices.sort()
            tot_indices = []
            for sub_indices in ls_indices:
                tot_indices.extend( sub_indices) 

            self.dic_indices[each] = tot_indices

        self.train_sampler = SubsetRandomSampler( self.dic_indices['train_indices'])
        self.val_sampler = SubsetRandomSampler( self.dic_indices['val_indices'])
        self.test_sampler = SubsetRandomSampler( self.dic_indices['test_indices'])
        # return dic_indices

    def extract_art_track(self,ar_art_bert_token):
        input_id = []
        # token_type_id = []
        attention_mask = []
        read_track = []
        for each in ar_art_bert_token:
            input_length = len(each)
            each_attention_mask = [1]* input_length + [0 for _ in range(self.MAX_LEN_TITLE - input_length)]
            each = each + [0 for _ in range(self.MAX_LEN_TITLE - input_length)]
            input_id.append( each)
            attention_mask.append( each_attention_mask)
        read_track.append(input_id)
        read_track.append(attention_mask)
        # read_track.append(token_type_id)
        
        return read_track
    
    def extract_combotrack(self,ar_art_bert_token,ar_com_bert_token ):
        input_id = []        
        attention_mask = []
        token_type_id = []
        combo_track = []
        for article, comment in zip(ar_art_bert_token, ar_com_bert_token):
            len_art = len(article)
            len_com = len(comment)
            mod_article = article + [0 for _ in range(self.MAX_LEN_TITLE - len_art)]
            mod_comment = comment + [0 for _ in range(self.MAX_LEN_COMMENT - len_com)]
            combo_input = mod_article + mod_comment[1:]
            each_attention_mask = [1 if x!=0 else 0 for x in combo_input ]
            each_tokentype = [0]* self.MAX_LEN_TITLE + [1]* (self.MAX_LEN_COMMENT - 1)
            #append tokens
            input_id.append( combo_input)
            attention_mask.append( each_attention_mask)
            token_type_id.append( each_tokentype)

        combo_track.append(input_id)
        combo_track.append( attention_mask)
        combo_track.append( token_type_id)
        return combo_track

    # def combine_tokens(comment_tok, article_tok):
    #     result = {}
    #     result['input_ids'] = [comment_tok['input_ids'][0] + article_tok['input_ids'][0][1:]]
    #     # droppping the mask for [CLS] token of article
    #     result['token_type_ids'] = [comment_tok['token_type_ids'][0] + [1] * (len(article_tok['token_type_ids'][0]) - 1)] 
    #     result['attention_mask'] = [comment_tok['attention_mask'][0] + article_tok['attention_mask'][0][1:]]
    #     return result

    def extract_sentiment_track(self, input_sentiment):
        vader = []
        flair = []
        blob_sentiment = []
        blob_subjective = []
        sentiment_track = []
        {'vader': 2, 'flair': 1, 'blob_sentiment': 1, 'blob_subjective': 2}
        for each in input_sentiment:
            vader.append(each['vader'])
            flair.append(each['flair'])
            blob_sentiment.append(each['blob_sentiment'])
            blob_subjective.append(each['blob_subjective'])
            
        sentiment_track.append(vader)
        sentiment_track.append(flair)
        sentiment_track.append(blob_sentiment)
        sentiment_track.append(blob_subjective)
        return sentiment_track
    
    # def get_split(self, batch_size=32, num_workers=4):
    #     logging.debug('Initializing train-validation-test dataloaders')
    #     self.train_loader = self.get_train_loader(batch_size=batch_size, num_workers=num_workers)
    #     self.val_loader = self.get_validation_loader(batch_size=batch_size, num_workers=num_workers)
    #     self.test_loader = self.get_test_loader(batch_size=batch_size, num_workers=num_workers)
    #     return self.train_loader, self.val_loader, self.test_loader
    
    # def get_train_loader(self, batch_size=32, num_workers=4):
    #     logging.debug('Initializing train dataloader')
    #     self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.train_sampler, shuffle=False, num_workers=num_workers)
    #     return self.train_loader

    # def get_validation_loader(self, batch_size=32, num_workers=4):
    #     logging.debug('Initializing validation dataloader')
    #     self.val_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.val_sampler, shuffle=False, num_workers=num_workers)
    #     return self.val_loader

    # def get_test_loader(self, batch_size=32, num_workers=4):
    #     logging.debug('Initializing test dataloader')
    #     self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.test_sampler, shuffle=False, num_workers=num_workers)
    #     return self.test_loader


