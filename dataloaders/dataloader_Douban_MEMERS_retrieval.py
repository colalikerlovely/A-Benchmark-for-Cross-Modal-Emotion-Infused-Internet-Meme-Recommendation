from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
import sys
import torch

sys.path.append('..')
from dataloaders.rawframes_util import RawFramesExtractor
from transformers import BertTokenizer, BertModel
from dataloaders.emotion_transform import EmotionTransform

class Douban_MEMERS_DataLoader(Dataset):
    """Douban_MEMERS dataset loader."""

    def __init__(
            self,
            csv_path,
            features_path,
            tokenizer,
            train_emotion,
            test_emotion,
            max_words=30,
            feature_framerate=1.0,
            max_frames=1,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            generate_text_backbone_clip_bert=0,
            original_text_backbone_clip_bert=0,
            num_class_names=12,
            generate_images=None,
            load_image=True
    ):

        self.data = json.load(open(csv_path, 'r',encoding='utf-8')) 
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        self.strategy = 1
        # print('Using uniform sampling without random offset for validation.')
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.rawFramesExtractor = RawFramesExtractor(
            num_segments=max_frames, size=image_resolution, random_shift=True, strategy=self.strategy)

        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.BERT_SPECIAL_TOKEN = {"CLS_TOKEN": "[CLS]", "SEP_TOKEN": "[SEP]",
                                   "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.load_image = load_image
        self.emotion_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.emotion_model = BertModel.from_pretrained('bert-base-uncased')
        self.test_emotion=test_emotion

    def __len__(self):
        return len(self.data)

    def _get_text(self, image_id, sentence):
        choice_image_ids = [image_id]
        n_caption = len(choice_image_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, image_id in enumerate(choice_image_ids):
            words = self.tokenizer.tokenize(sentence)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_image_ids

  
    def get_emotion_embedding(self,file_path, image_id):

        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        for item in json_data:
            if item["image_id"] == image_id:
                emotions = [list(person.values())[0] for person in item["emotion"]]
                break
        emotion_embeddings = []
        for emotion in emotions:
            encoded_dict = self.emotion_tokenizer.encode_plus(text=emotion,add_special_tokens=True)
            input_ids=encoded_dict['input_ids']
            max_length = 128
            input_ids = input_ids[:max_length]
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + ([self.emotion_tokenizer.pad_token_id] * padding_length)
            input_ids_tensor = torch.tensor([input_ids])

            with torch.no_grad():
                outputs = self.emotion_model(input_ids=input_ids_tensor)
            embedding = outputs[0][:, 0, :]
            emotion_embeddings.append(embedding)
        avg_embedding = torch.mean(torch.stack(emotion_embeddings), dim=0)

        return avg_embedding
    def _get_rawframes(self, choice_image_ids):
        image_mask = np.zeros((len(choice_image_ids), self.max_frames), dtype=np.long)
        max_image_length = [0] * len(choice_image_ids)

        image = np.zeros((len(choice_image_ids), self.max_frames, 1, 4,
                          self.rawFramesExtractor.size, self.rawFramesExtractor.size), dtype=np.float)

        for i, image_id in enumerate(choice_image_ids):
            emotion_file_path=self.test_emotion
            emotion_embedding = self.get_emotion_embedding(emotion_file_path, image_id) 
            image_path = os.path.join(self.features_path, "{}".format(image_id)) 

            raw_image_data = self.rawFramesExtractor.get_image_data(image_path)
            raw_image_data = raw_image_data['image']
            if len(raw_image_data.shape) > 3:
                raw_image_data_clip = raw_image_data
                raw_image_slice = self.rawFramesExtractor.process_raw_data(raw_image_data)
                if self.max_frames < raw_image_slice.shape[0]:
                    if self.slice_framepos == 0: 
                        image_slice = raw_image_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1: 
                        image_slice = raw_image_slice[-self.max_frames:, ...]
                    else: 
                        sample_index = np.linspace(0, raw_image_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        image_slice = raw_image_slice[sample_index, ...]
                else:
                    image_slice = raw_image_slice

                image_slice = self.rawFramesExtractor.process_frame_order(image_slice, frame_order=self.frame_order)

                emotion_transform = EmotionTransform(input_dim=768, output_dim=224*224)
                transformed_emotion = emotion_transform(emotion_embedding)
                emotion_channel = transformed_emotion.view(1, 1, 1, 224, 224)
                image_slice = torch.cat((image_slice, emotion_channel), dim=2)

                slice_len = image_slice.shape[0]
                max_image_length[i] = max_image_length[i] if max_image_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    image[i][:slice_len, ...] = image_slice

        for i, v_length in enumerate(max_image_length):
            image_mask[i][:v_length] = [1] * v_length

        return image, image_mask

    def _get_title(self, titles):
        if titles is None:
            titles = ' '
        if isinstance(titles, str):
            titles = [titles]

        n_text = len(titles)

        pairs_text = np.zeros((n_text, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((n_text, self.max_words), dtype=np.long)

        for idx in range(n_text):
            title = titles[idx]
            words = self.tokenizer.tokenize(title)

            # add begin
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            # add end
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words

            pairs_text[idx] = np.array(input_ids)
            pairs_mask[idx] = np.array(input_mask)

        return pairs_text, pairs_mask


    def __getitem__(self, idx):
        image_id = list(self.data.keys())[idx]
        sentence = self.data[image_id]['gt']
        title = self.data[image_id]['titles']

        pairs_text, pairs_mask, pairs_segment, choice_image_ids = self._get_text(image_id, sentence)

        if self.load_image == True:
            image, image_mask = self._get_rawframes(choice_image_ids)
        else:
            image = 1
            image_mask = 1

        title_text, title_mask = self._get_title(title)

        return pairs_text, pairs_mask, pairs_segment, image, image_mask, title_text, title_mask

class Douban_MEMERS_TrainDataLoader(Dataset):
    """Douban_MEMERS train dataset loader."""

    def __init__(
            self,
            csv_path,
            json_path,
            features_path,
            tokenizer,
            train_emotion,
            test_emotion,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            unfold_sentences=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            strategy=1,
            generate_text_backbone_clip_bert=0,
            original_text_backbone_clip_bert=1,
            num_class_names=12,
            generate_images=None,
            load_image=True,
            caption_aug = True
    ):

        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r',encoding='utf-8'))
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        self.strategy = strategy
        self.caption_aug = caption_aug
        self.train_emotion = train_emotion

        if self.caption_aug == True: 
            title_list = []
            for k, v in self.data['title'].items():
                if isinstance(v,list):
                    for i in range(len(v)):
                        title_list.append({'image_id': k, 'caption': v[i]})
                else:
                    title_list.append({'image_id': k, 'caption': v})
            self.data['sentences'].extend(title_list)

        if self.unfold_sentences:
            train_image_ids = list(self.csv['image_id'].values)
            self.sentences_dict = {}
            for itm in self.data['sentences']:
                if itm['image_id'] in train_image_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (
                    itm['image_id'], itm['caption'])
            self.sample_len = len(self.sentences_dict)

        self.rawFramesExtractor = RawFramesExtractor(
            num_segments=max_frames, size=image_resolution, random_shift=True, strategy=self.strategy)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.BERT_SPECIAL_TOKEN = {"CLS_TOKEN": "[CLS]", "SEP_TOKEN": "[SEP]",
                                   "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.load_image = load_image
        self.emotion_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.emotion_model = BertModel.from_pretrained('bert-base-uncased')

    def __len__(self):
        return self.sample_len


    def _get_text(self, image_id, caption=None):
        k = 1
        choice_image_ids = [image_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, image_id in enumerate(choice_image_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else: 
                words = self._get_single_text(image_id)

            # add begin
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            # add end
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask) 
            pairs_segment[i] = np.array(segment_ids) 
        return pairs_text, pairs_mask, pairs_segment, choice_image_ids

    def _get_single_text(self, image_id):

        rind = len(self.sentences['image_id']) // 2 
        caption = self.sentences[image_id][rind]
        words = self.tokenizer.tokenize(caption)

        return words

    def _get_title(self, image_id):
        title = self.data['title'][image_id]
        words = self.tokenizer.tokenize(title)

        pairs_text = np.zeros((1, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((1, self.max_words), dtype=np.long)

        # add begin
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        # add end
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words

        pairs_text[0] = np.array(input_ids)
        pairs_mask[0] = np.array(input_mask)

        return pairs_text, pairs_mask

    def _get_titles(self, image_id):

        titles = self.data['titles'][image_id]
        n_text = len(titles)
        pairs_text = np.zeros((n_text, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((n_text, self.max_words), dtype=np.long)
        for idx in range(n_text):
            title = titles[idx]
            words = self.tokenizer.tokenize(title)
            # add begin
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            # add end
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words

            pairs_text[idx] = np.array(input_ids)
            pairs_mask[idx] = np.array(input_mask)

        return pairs_text, pairs_mask

    
    def get_emotion_embedding(self,file_path, image_id):

        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        for item in json_data:
            if item["image_id"] == image_id:
                emotions = [list(person.values())[0] for person in item["emotion"]]
                break

        emotion_embeddings = []
        for emotion in emotions:
            encoded_dict = self.emotion_tokenizer.encode_plus(text=emotion,add_special_tokens=True)
            input_ids=encoded_dict['input_ids']
            max_length = 128
            input_ids = input_ids[:max_length]
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + ([self.emotion_tokenizer.pad_token_id] * padding_length)
            input_ids_tensor = torch.tensor([input_ids])

            with torch.no_grad():
                outputs = self.emotion_model(input_ids=input_ids_tensor)
            embedding = outputs[0][:, 0, :]
            emotion_embeddings.append(embedding)
        avg_embedding = torch.mean(torch.stack(emotion_embeddings), dim=0)

        return avg_embedding

    def _get_rawframes(self, choice_image_ids):
        image_mask = np.zeros((len(choice_image_ids), self.max_frames), dtype=np.long) 
        max_image_length = [0] * len(choice_image_ids)
        image = np.zeros((len(choice_image_ids), self.max_frames, 1, 4,
                          self.rawFramesExtractor.size, self.rawFramesExtractor.size), dtype=np.float)

        for i, image_id in enumerate(choice_image_ids):
            file_path=self.train_emotion
            emotion_embedding = self.get_emotion_embedding(file_path,image_id) 
            image_path = os.path.join(self.features_path, "{}".format(image_id))  
            raw_image_data = self.rawFramesExtractor.get_image_data(image_path)
            raw_image_data = raw_image_data['image']

            if len(raw_image_data.shape) > 3:
                raw_image_data_clip = raw_image_data
                raw_image_slice = self.rawFramesExtractor.process_raw_data(raw_image_data_clip) 
                image_slice = raw_image_slice
                self.frame_order=0 
                image_slice = self.rawFramesExtractor.process_frame_order(image_slice, frame_order=self.frame_order)

                emotion_transform = EmotionTransform(input_dim=768, output_dim=224*224)
                transformed_emotion = emotion_transform(emotion_embedding)
                emotion_channel = transformed_emotion.view(1, 1, 1, 224, 224)
                image_slice = torch.cat((image_slice, emotion_channel), dim=2)

                slice_len = image_slice.shape[0]
                max_image_length[i] = max_image_length[i] if max_image_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    image[i][:slice_len, ...] = image_slice.detach() 

        for i, i_length in enumerate(max_image_length):
            image_mask[i][:i_length] = [1] * i_length

        return image, image_mask

    def __getitem__(self, idx):
        if self.unfold_sentences:
            image_id, caption = self.sentences_dict[idx]
        else: 
            image_id, caption = self.csv['image_id'].values[idx], None
        pairs_text, pairs_mask, pairs_segment, choice_image_id_ids = self._get_text(image_id, caption) 

        if self.load_image == True:
            image, image_mask = self._get_rawframes(choice_image_id_ids)
        else:
            image_id = 1
            image_id_mask = 1
        title_text, title_mask = self._get_titles(image_id) 

        return pairs_text, pairs_mask, pairs_segment, image, image_mask, title_text, title_mask
