import streamlit as st
import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import from_pretrained_keras
import tensorflow as tf

import os
import tensorflow_datasets as tfds




def get_model():
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
    model = from_pretrained_keras("yunuskoyun/gazdasBERT")
    return tokenizer,model


tokenizer,model = get_model()

user_input = st.text_area('İhbar Tanımı Tahmin Etmek İçin Yorum Yazınız...')
button = st.button("Tahmin Et")

d = {
    
  4:'Gaz Yokluğu',
  8:'Mühür',
  0:'Basinç Problemi',
  3:'Gaz Kokusu',
  12:'Yangin',
  10:'Sayaç Problemleri',
  5:'Hasar',
  2:'Elektrik Problemleri',
  1:'Diğer Problemler',
  11:'Servis Kutusu Problemleri',
  7:'Kazi Çalişmasi',
  9:'Patlama',
  13:'Zehirlenme',
  6:'İntihar'
}


def prep_data(text):
  import tensorflow as tf
  
  def transformation(X):

    seq_len = 164

    Xids = []
    Xmask = []


    for sentence in X:

        tokens = tokenizer.encode_plus(sentence, max_length=seq_len, truncation=True,
                                        padding='max_length', add_special_tokens=True)



        Xids.append(tokens['input_ids'])
        Xmask.append(tokens['attention_mask'])

    return np.array(Xids), np.array(Xmask)


  Xids_obs, Xmask_obs = transformation(text)


  dataset_obs = tf.data.Dataset.from_tensor_slices((Xids_obs, Xmask_obs))

 
  def map_func(Tensor_Xids, Tensor_Xmask):
       return {'input_ids': Tensor_Xids, 'attention_mask': Tensor_Xmask}

  dataset_obs = dataset_obs.map(map_func)

 
  batch_size = 32 
  obs_ds = dataset_obs.batch(batch_size)

  return obs_ds



if user_input and button:
    probs = model.predict(prep_data([user_input]))
    pred = np.argmax(probs[0])
    st.write(f"%{round(max(probs[0])*100, 0)} Olasılıkla İhbar Tanımınız:  **{d[pred]}**")