import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

# from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import wandb

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
# from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
from transformers import DistilBertModel, BertTokenizerFast


class FingerprintModel(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        
        self.token_embedding = None
        self.train_loss = {'vader': [0], 'flair': [0], 'sent': [0], 'subj': [0], 'emotion': [0], 'author': [0], 'mean': [0]}
        self.dev_loss = {'vader': [0], 'flair': [0], 'sent': [0], 'subj': [0], 'emotion': [0], 'author': [0], 'mean': [0]}
       
        self.train_perf = {'vader': 0, 'flair': 0, 'sent': 0, 'subj': 0, 'emotion': 0, 'mean': 0}
        self.dev_perf   = {'vader': 0, 'flair': 0, 'sent': 0, 'subj': 0,'emotion': 0, 'mean': 0}
        self.test_perf  = {'vader': 0, 'flair': 0, 'sent': 0, 'subj': 0,'emotion': 0, 'mean': 0}
        ###############kishore_update#######################################
        ##BERT pretrained
        # import BERT-base pretrained model
        self.bert  = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # Freeze the BERT model
        if config['freeze_bert']:
            for param in self.bert.parameters():
                param.requires_grad = False
        #####################################################################
        self.token_encoder = getattr(nn, config['rnn_type'].upper())(
            input_size=config['token_dim'], hidden_size=config['hid_dim'] // 2,
            num_layers=1, dropout=config['dropout'],
            batch_first=True, bidirectional=True)

        author_final_dim = 0
        if config['build_author_emb']:
            self.author_embedding = nn.Embedding(config['author_size'], config['author_dim'])
            author_final_dim += config['author_dim']

        if config['build_author_track']:
            input_size = 2 * config['hid_dim']
            if config['build_sentiment_embedding']:
                input_size += 4 * config['sentiment_dim']
            if config['build_topic_predict'] and config['leverage_topic']:
                input_size += config['topic_size']
            if config['leverage_emotion']:
                input_size += 6

            self.timestamp_merge = nn.Sequential(
                nn.Linear(input_size, config['author_track_dim']),
                nn.ReLU(),
                nn.Linear(config['author_track_dim'], config['author_track_dim']),
                nn.ReLU())

            self.track_encoder = getattr(nn, config['rnn_type'].upper())(
                input_size=config['author_track_dim'], hidden_size=config['author_track_dim'],
                num_layers=config['rnn_layer'], dropout=config['dropout'],
                batch_first=True, bidirectional=False)
            author_final_dim += config['author_track_dim'] * config['rnn_layer']
            # self.author_merge = nn.Linear(config['hid_dim'] * 2, config['author_dim'])

        if config['build_author_predict']:
            self.author_predict = nn.Linear(author_final_dim, config['author_size'])

        if config['build_topic_predict']:
            self.topic_predict = nn.Linear(config['hid_dim'], config['topic_size'])

        in_dim = author_final_dim + config['hid_dim']
        self.dropout = nn.Dropout(config['dropout'])

        if config['sentiment_fingerprinting']:
            self.vader_predict = nn.Sequential(
                nn.Linear(in_dim, config['sentiment_dim']),
                nn.ReLU(),
                nn.Linear(config['sentiment_dim'], 3))
            self.flair_predict = nn.Sequential(
                nn.Linear(in_dim, config['sentiment_dim']),
                nn.ReLU(),
                nn.Linear(config['sentiment_dim'], 3))
            self.blob_sent = nn.Sequential(
                nn.Linear(in_dim, config['sentiment_dim']),
                nn.ReLU(),
                nn.Linear(config['sentiment_dim'], 3))
            self.blob_subj = nn.Sequential(
                nn.Linear(in_dim, config['sentiment_dim']),
                nn.ReLU(),
                nn.Linear(config['sentiment_dim'], 3))

        if config['emotion_fingerprinting']:
            self.emotion_predict = nn.Linear(in_dim, config['emotion_dim'])

        if config['build_sentiment_embedding']:
            self.vader_embed = nn.Embedding(3, config['sentiment_dim'])
            self.vader_embed.weight = self.vader_predict[2].weight
            self.flair_embed = nn.Embedding(3, config['sentiment_dim'])
            self.flair_embed.weight = self.flair_predict[2].weight
            self.sent_embed = nn.Embedding(3, config['sentiment_dim'])
            self.sent_embed.weight = self.blob_sent[2].weight
            self.subj_embed = nn.Embedding(3, config['sentiment_dim'])
            self.subj_embed.weight = self.blob_subj[2].weight

        self.config = config

    def forward(self,  author, read_track, write_track, article_pack, sentiments, emotion):
        result = {}
        
        len_track = self.config['previous_comment_cnt']+1
        batch_size = author.size()[0]
        seq_len = torch.tensor( [len_track]* batch_size, device=self.device)

        r_ht = self._encode_seq_seq_(read_track[0], read_track[1])
        
        author_embeds = []
        if self.config['build_author_track']:
            w_ht = self._encode_seq_seq_(write_track[0], write_track[1] )

            tracks = [r_ht, w_ht]
            if self.config['build_sentiment_embedding']:
                tracks.extend([self.vader_embed( sentiments[0]),
                               self.flair_embed(sentiments[1]),
                               self.sent_embed(sentiments[2]),
                               self.subj_embed(sentiments[3])])
            if self.config['build_topic_predict'] and self.config['leverage_topic']:
                predict_topic = F.softmax(self.topic_predict(r_ht).detach(), dim=-1)
                tracks.append(predict_topic)
            if self.config['leverage_emotion']:
                tracks.append( emotion)
            track_embeds = torch.cat(tracks, dim=-1)[:, :-1, :]
            track_embeds = F.relu(self.timestamp_merge(track_embeds))
            _, track_ht = self._rnn_encode_(self.track_encoder, track_embeds,
                                            seq_len - 1)
            author_embeds.append(track_ht)
            
        if self.config['build_author_emb']:
            
            author_embeds.append(self.author_embedding(author))

        if len(author_embeds) > 1:
            author_embeds = torch.cat(author_embeds, dim=-1)
        elif len(author_embeds) == 1:
            author_embeds = author_embeds[0]
        else:
            raise NotImplementedError()
        
        #Extracting last article to predict the sentiment
        final_idx = (seq_len - 1).view(-1, 1).expand(-1, r_ht.size(2))
        final_idx = final_idx.unsqueeze(1)
        final_rt = r_ht.gather(1, final_idx).squeeze(1)
        # final_rt = r_ht[:, -1, :]  # last time stamp to predict
        if self.config['sentiment_fingerprinting']:
            result['flair'] = self.flair_predict(torch.cat((author_embeds, final_rt), dim=-1))  # batch, seq, 3
            result['vader'] = self.vader_predict(torch.cat((author_embeds, final_rt), dim=-1))
            result['sent'] = self.blob_sent(torch.cat((author_embeds, final_rt), dim=-1))
            result['subj'] = self.blob_subj(torch.cat((author_embeds, final_rt), dim=-1))

        if self.config['emotion_fingerprinting']:
            result['emotion'] = self.emotion_predict(torch.cat((author_embeds, final_rt), dim=-1))
        
        return result

    def training_step(self, batch, batch_idx):        
        # training_step defined the train loop. It is independent of forward
        batch = self.batch_transform(batch)
        loss = self.get_loss( batch_idx, batch, self.train_loss)

        #train accuracy
        train_perf, _ = self.get_perf( batch_idx, batch, self.train_perf)

        # Log training loss
        self.log('train_loss', loss, on_step=True)#, on_epoch=True)
        # self.log('train_loss_vader', self.train_loss['vader'], on_step=True)#, on_epoch=True)
        # self.log('train_loss_flair', self.train_loss['flair'], on_step=True)#, on_epoch=True)
        # self.log('train_loss_sent', self.train_loss['sent'], on_step=True)#, on_epoch=True)
        # self.log('train_loss_subj', self.train_loss['subj'], on_step=True)#, on_epoch=True)
        # self.log('train_loss_emotion', self.train_loss['emotion'], on_step=True)#, on_epoch=True)

        # Log metrics
        self.log('train_acc', train_perf, on_step=True)#, on_epoch=True)
        # self.log('train_acc_vader', self.train_perf['vader'], on_step=True)#, on_epoch=True)
        # self.log('train_acc_flair', self.train_perf['flair'], on_step=True)#, on_epoch=True)
        # self.log('train_acc_sent', self.train_perf['sent'], on_step=True)#, on_epoch=True)
        # self.log('train_acc_subj', self.train_perf['subj'], on_step=True)#, on_epoch=True)
        # self.log('train_acc_emotion', self.train_perf['emotion'], on_step=True)#, on_epoch=True)

        # #logging metrics as dictionary
        # values = {'train_loss': loss,'train_loss_vader': self.train_loss['vader'], 
        # 'train_loss_flair': self.train_loss['flair'], 'train_loss_sent': self.train_loss['sent'], 
        # 'train_loss_subj': self.train_loss['subj'], 'train_loss_emotion': self.train_loss['emotion'],
        # 'train_acc':train_perf , 'train_acc_vader':self.train_perf['vader'], 
        # 'train_acc_flair':self.train_perf['flair'], 'train_acc_sent':self.train_perf['sent'], 
        # 'train_acc_subj':self.train_perf['subj'], 'train_acc_emotion': self.train_perf['emotion']
        # }
        # self.log_dict(values,on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        val_batch = self.batch_transform( batch)
        val_loss = self.get_loss( batch_idx, val_batch, self.dev_loss)
        val_perf, _ = self.get_perf( batch_idx, val_batch, self.dev_perf)
        
        # # Log devset loss
        self.log('val_loss', val_loss, on_step=True)#, on_epoch=True)#, sync_dist=True)
        # self.log('val_loss_vader', self.dev_loss['vader'], on_step=True)#, on_epoch=True)#, sync_dist=True)
        # self.log('val_loss_flair', self.dev_loss['flair'], on_step=True)#, on_epoch=True)#, sync_dist=True)
        # self.log('val_loss_sent', self.dev_loss['sent'], on_step=True)#, on_epoch=True)#, sync_dist=True)
        # self.log('val_loss_subj', self.dev_loss['subj'], on_step=True)#, on_epoch=True)#, sync_dist=True)
        # self.log('val_loss_emotion', self.dev_loss['emotion'], on_step=True)#, on_epoch=True)#, sync_dist=True)

        # # Log metrics
        self.log('val_acc', val_perf, on_step=True)#, on_epoch=True)#, sync_dist=True,)
        # self.log('val_acc_vader', self.dev_perf['vader'], on_step=True)#, on_epoch=True)#, sync_dist=True)
        # self.log('val_acc_flair', self.dev_perf['flair'], on_step=True)#, on_epoch=True)#, sync_dist=True)
        # self.log('val_acc_sent', self.dev_perf['sent'], on_step=True)#, on_epoch=True)#, sync_dist=True)
        # self.log('val_acc_subj', self.dev_perf['subj'], on_step=True)#, on_epoch=True)#, sync_dist=True)
        # self.log('val_acc_emotion', self.dev_perf['emotion'], on_step=True)#, on_epoch=True)#, sync_dist=True)

        #logging metrics as dictionary
        # values = {'val_loss': val_loss,'val_loss_vader': self.dev_loss['vader'], 
        # 'val_loss_flair': self.dev_loss['flair'], 'val_loss_sent': self.dev_loss['sent'], 
        # 'val_loss_subj': self.dev_loss['subj'], 'val_loss_emotion': self.dev_loss['emotion'],
        # 'val_acc':val_perf , 'val_acc_vader':self.dev_perf['vader'], 
        # 'val_acc_flair':self.dev_perf['flair'], 'val_acc_sent':self.dev_perf['sent'], 
        # 'val_acc_subj':self.dev_perf['subj'], 'val_acc_emotion': self.dev_perf['emotion']
        # }
        # self.log_dict(values, on_epoch=True)
        
        return val_loss

    def test_step(self, batch, batch_idx):
        test_batch = self.batch_transform( batch)
        test_acc, test_preds = self.get_perf( batch_idx, test_batch, self.test_perf)

        json.dump(test_acc, open(os.path.join( self.config['root_folder'], self.config['outlet'], 'test_perf.json'), 'w'))
        with open(os.path.join(self.config['root_folder'],
                                self.config['outlet'], 'test_pred.jsonl'), 'w') as file:
            for pred in test_preds:
                data = json.dumps(pred)
                file.write(data)

         # Calling self.log will surface up scalars
        self.log('test_acc', test_acc)#, sync_dist=True)
                
        return test_acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer

    #helper functions    
    def _encode_seq_seq_(self, seq_seq_tensor, token_mask):
        #Here token_mask is attention_mask in bert token
        batch_size, seq_len, token_size = seq_seq_tensor.size()
        token_encode_mtx = seq_seq_tensor.reshape(-1, token_size)
        attention_mtx = token_mask.reshape(-1, token_size)
        token_len = token_mask.reshape(-1, token_size).sum(-1)
    
        # attention_encode_mtx = track_attention.view(-1, token_size)
        # embeds, ht = self._rnn_encode_(self.token_encoder, self.token_embedding(token_encode_mtx),
        #                                token_len)
        # embeds = embeds.view(batch_size, seq_len, token_size, embeds.size(-1))
        # ht = ht.view(batch_size, seq_len, ht.size(-1))
        # return embeds, ht

        # Feed input to BERT
        bert_outputs = self.bert(input_ids= token_encode_mtx, attention_mask= attention_mtx)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = bert_outputs[0][:, 0, :]

        cls_format = last_hidden_state_cls.view(batch_size, seq_len, last_hidden_state_cls.size(-1))
        return cls_format

    def _rnn_encode_(self, rnn, x, length, order=None, track=None):
        if len(x.size()) == 3:
            batch_size, seq_len, token_num = x.size()
        elif len(x.size()) == 2:
            batch_size, token_num = x.size()
        else:
            raise NotImplementedError("Not support input dimensions {}".format(x.size()))

        if order is not None:
            x = x.index_select(0, order)
        x = self.dropout(x)
        x = pack(x, length, batch_first=True, enforce_sorted=False)
        
        outputs, h_t = rnn(x)
        outputs = unpack(outputs, batch_first=True)[0]
        if isinstance(h_t, tuple):
            h_t = h_t[0]
        if track is not None:
            outputs = outputs[track]
            h_t = h_t.index_select(1, track).transpose(0, 1).contiguous()
        else:
            h_t = h_t.transpose(0, 1).contiguous()
        return outputs, h_t.view(batch_size, -1)

    def get_loss(self, batch_idx, batch, loss_dict):
        loss = 0
        # if self.config['build_topic_predict']:
        #     articles, topics = self.data_io.topic_classification_input(batch_idx, examples)
        # else:
        #     articles, topics = None, None
        articles, topics = None, None
        #sending to tensors to cuda from device a to b's device
        # tensor_a = tensor_a.cuda(b.get_device()) if b.is_cuda else tensor_a
        

        author, r_tracks, w_tracks, sentiment, emotion = batch

        train_result = self(author, r_tracks, w_tracks, articles, sentiment, emotion)#, self.device)

        if self.config['build_author_predict']:
            author_loss = F.cross_entropy(train_result['author'],
                                          author) #torch.tensor(author, device=self.device))
                                          
            loss += 0.1 * author_loss
            loss_dict['author'].append(author_loss.detach().cpu().item()) #.item())
        # if self.config['build_topic_predict']:
        #     b, d = train_result['topic'].size()
        #     topic_loss = F.cross_entropy(train_result['topic'],
        #                                  torch.tensor(topics, device=self.device))
        #     loss += topic_loss
        #     loss_dict['topic'].append(topic_loss.item())
        if self.config['sentiment_fingerprinting']:
            vader_loss = F.cross_entropy(
                train_result['vader'],
                # torch.tensor(sentiment[0][0], device=self.device)[:, -1])
                sentiment[0][:, -1])
            loss += vader_loss
            loss_dict['vader'].append(vader_loss.detach().cpu().item())

            flair_loss = F.cross_entropy(
                train_result['flair'],
                # torch.tensor(sentiment[1][0], device=self.device)[:, -1])
                sentiment[1][:, -1])
            loss += flair_loss
            loss_dict['flair'].append(flair_loss.detach().cpu().item())

            blob_sent = F.cross_entropy(
                train_result['sent'],
                # torch.tensor(sentiment[2][0], device=self.device)[:, -1])
                sentiment[2][:, -1])
            loss += blob_sent
            loss_dict['sent'].append(blob_sent.detach().cpu().item())

            blob_subj = F.cross_entropy(
                train_result['subj'],
                # torch.tensor(sentiment[3][0], device=self.device)[:, -1])
                sentiment[3][:, -1])
            loss += blob_subj
            loss_dict['subj'].append(blob_subj.detach().cpu().item())
        if self.config['emotion_fingerprinting']:
            emotion_loss = F.binary_cross_entropy_with_logits(train_result['emotion'], emotion[:,-1])
                                                            #   torch.tensor(emotion[0], device=self.device,
                                                            #                dtype=torch.float)[:, -1])
            loss += emotion_loss
            loss_dict['emotion'].append(emotion_loss.detach().cpu().item())
        return loss

    def get_perf(self, batch_idx, batch, acc_dict):#data_iter, examples):
        pred_records = []
        tmp_cnt = {'vader': [0, 0], 'flair': [0, 0], 'sent': [0, 0], 'subj': [0, 0],  'emotion': [0, 0]}
        # acc = {'vader': 0, 'flair': 0, 'sent': 0, 'subj': 0,
        #        'emotion': 0, 'mean': 0}
        # self.model.eval()
        # for i, batch_idx in enumerate(data_iter):

        # batch = [each.cuda(self.device) for each in batch ]
        author, r_tracks, w_tracks, sentiment, emotion = batch
        articles, topics = None, None
        result = self(author, r_tracks, w_tracks, articles, sentiment, emotion)#, self.device, train=False)
        for j, a in enumerate(author):
            pred_record = {'vader': [0, 0], 'flair': [0, 0], 'sent': [0, 0], 'subj': [0, 0],
                            'emotion': [0, 0]}
            if self.config['sentiment_fingerprinting']:
                pred_record['vader'][0] = result['vader'][j].argmax().detach().cpu().item()
                pred_record['vader'][1] = sentiment[0,j,-1].detach().cpu().item() #sentiment[0][0][j][-1]
                tmp_cnt['vader'][int(pred_record['vader'][0] == pred_record['vader'][1])] += 1
                pred_record['flair'][0] = result['flair'][j].argmax().detach().cpu().item()
                pred_record['flair'][1] = sentiment[1,j,-1].detach().cpu().item() #sentiment[1][0][j][-1]
                tmp_cnt['flair'][int(pred_record['flair'][0] == pred_record['flair'][1])] += 1
                pred_record['sent'][0] = result['sent'][j].argmax().detach().cpu().item()
                pred_record['sent'][1] = sentiment[2,j,-1].detach().cpu().item() #sentiment[2][0][j][-1]
                tmp_cnt['sent'][int(pred_record['sent'][0] == pred_record['sent'][1])] += 1
                pred_record['subj'][0] = result['subj'][j].argmax().detach().cpu().item()
                pred_record['subj'][1] = sentiment[3 ,j,-1].detach().cpu().item() #sentiment[3][0][j][-1]
                tmp_cnt['subj'][int(pred_record['subj'][0] == pred_record['subj'][1])] += 1
            if self.config['emotion_fingerprinting']:
                pred_emo = [int(i > 0) for i in result['emotion'][j].detach().tolist()]
                pred_record['emotion'][0] = pred_emo
                pred_record['emotion'][1] = (emotion[j,-1]).detach().cpu().numpy().astype(int).tolist()
                tmp_cnt['emotion'][0] += sum([pred == gold for pred, gold in zip(*pred_record['emotion'])])
                tmp_cnt['emotion'][1] += sum([pred != gold for pred, gold in zip(*pred_record['emotion'])])
            pred_records.append(pred_record)

        # self.model.train()
        tmp_acc_sum = []
        if self.config['sentiment_fingerprinting']:
            for key in ['vader', 'flair', 'sent', 'subj']:
                acc_dict[key] = 1.0 * tmp_cnt[key][1] / (tmp_cnt[key][0] + tmp_cnt[key][1])
                tmp_acc_sum.append(acc_dict[key])
        if self.config['emotion_fingerprinting']:
            acc_dict['emotion'] = 1.0 * tmp_cnt['emotion'][0] / (tmp_cnt['emotion'][0] + tmp_cnt['emotion'][1])
            # acc['emotion'] = acc['emotion'].item()
            tmp_acc_sum.append(acc_dict['emotion'])
        acc_dict['mean'] = sum(tmp_acc_sum) / len(tmp_acc_sum)
        return acc_dict['mean'], pred_records

    def batch_transform(self, batch):
        batch[0] = torch.squeeze( batch[0]) #author
        batch[1] = torch.squeeze( batch[1]).transpose(0,1) #read_track
        batch[2] = torch.squeeze( batch[2]).transpose(0,1) #write_track
        batch[3] = torch.squeeze( batch[3]).transpose(0,1) #sentiments
        batch[4] = torch.squeeze( batch[4]) #emotion

        # batch = [each.cuda(self.device) for each in batch ]
        return batch  
