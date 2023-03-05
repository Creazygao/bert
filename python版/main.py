import os
import tensorflow as tf
from datetime import datetime
import pandas as pd
import numpy as np
import re
from tensorflow.core.example.feature_pb2 import Features
from typing import List
from datetime import datetime
from typing import Counter
import zhconv
import copy
import tensorflow_addons as tfa

def get_vocb(file_path, vocb_path, min_freq):
  '''
  使用file_path文件生成vocb_path词表，min_freq为最小词频
  '''
    vocabulary2 = ""
    if os.path.exists(vocb_path):
        with open(vocb_path, 'r', encoding='utf-8') as f:
            vocabulary2 = eval(f.read())
    elif os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read();
        res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9]")
        data = res.sub('', data)
        words_freq = Counter(data).most_common()
        vocabulary2 = [words for (words, freq) in words_freq if freq > min_freq]
        vocabulary2 = ['CLS', 'SEP', 'MASK', 'PAD', 'UNK'] + vocabulary2
        with open(vocb_path, 'w', encoding='utf-8') as f:
            f.write(str(vocabulary2))
    vocb2id_dic = dict(zip(vocabulary2, list(range(len(vocabulary2)))))
    id2vocb_dic = dict(zip(list(range(len(vocabulary2))), vocabulary2))
    length = len(vocabulary2)
    return vocb2id_dic, id2vocb_dic, length


def _bytes_feature(value):
  '''
  字符串特征格式
  '''
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    '''
  浮点数特征格式
  '''
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  '''
  整数型特征格式
  '''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def get_example_data(f1, f2, f3, f4, f5, f6):
  '''
  返回tf.train.example作为后续惰性加载数据
  '''
    example = tf.train.Example(features=tf.train.Features(feature={
        "label_y": _float_feature([f1]),
        "raw_token": _float_feature(
            f2),
        "segment_token": _float_feature(
            f3),
        "mask_token": _float_feature(
            f4),
        "mask_pos": _float_feature(f5),
        "pad_token": _float_feature(f6),
    }))
    return example


def get_record_data(sentences1, sentences2, word_to_id, seq_len, mask_rate):
  '''
 生成tfrecord格式数据
  '''
    s_len = 0
    sep_token = word_to_id["SEP"]
    cls_token = word_to_id["CLS"]

    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9]")
    sentence1 = res.sub('', sentences1)
    sentence2 = res.sub('', sentences2)
    if len(sentence1) > seq_len - 2:
        sentence1 = sentence1[:seq_len - 2]
    if len(sentence2) > seq_len - 2:
        sentence2 = sentence2[:seq_len - 2]
    # 生成句子的token
    token1 = []
    token2 = []
    exam = []
    for v in sentence1:
        if v in word_to_id:
            token1.append(word_to_id[v])
        else:
            token1.append(word_to_id['UNK'])
    for v in sentence2:
        if v in word_to_id:
            token2.append(word_to_id[v])
        else:
            token2.append(word_to_id['UNK'])
    sentence1_token = [cls_token] + token1 + [sep_token] + token2 + [sep_token]
    sentence2_token = [cls_token] + token2 + [sep_token] + token1 + [sep_token]
    pad1_token=[]
    #生成token、padding
    if len(sentence1_token) < seq_len:
        pad1_token = [0] * len(sentence1_token) + [1] * (seq_len - len(sentence1_token))
        s_len = len(sentence1_token) - 2
        sentence1_token += [word_to_id['PAD']] * (seq_len - len(sentence1_token))
        sentence2_token += [word_to_id['PAD']] * (seq_len - len(sentence2_token))
       

    elif len(sentence1_token) > seq_len:
        s_len = seq_len - 2
        sentence1_token = sentence1_token[0:seq_len - 1]
        sentence1_token.extend([word_to_id["SEP"]])
        sentence2_token = sentence2_token[0:seq_len - 1]
        sentence2_token.extend([word_to_id["SEP"]])
        pad1_token = [0] * seq_len

    pad2_token = pad1_token
    #生成segment padding
    segment1_token = [0] * (len(sentence1) + 2) + [1] * (seq_len - len(sentence1) - 2)
    segment2_token = [0] * (len(sentence2) + 2) + [1] * (seq_len - len(sentence2) - 2)
    #生成mask，多次重复生成，模仿动态mask
    for i in range(s_len // 50):
        mask_num = np.ceil(s_len * mask_rate).astype(int)
        position1 = np.random.choice(a=np.arange(1, s_len + 1), size=mask_num, replace=False)
        position2 = np.random.choice(a=np.arange(1, s_len + 1), size=mask_num, replace=False)
        mask1_token = copy.deepcopy(sentence1_token)
        mask2_token = copy.deepcopy(sentence2_token)
        mask1_pos = [0] * seq_len
        mask2_pos = [0] * seq_len
        for p in position1:
            mask1_token[p] = word_to_id["MASK"]
            mask1_pos[p] = 1
        for p in position2:
            mask2_token[p] = word_to_id["MASK"]
            mask2_pos[p] = 1
        example1 = get_example_data(1, sentence1_token, segment1_token, mask1_token, mask1_pos, pad1_token)
        example2 = get_example_data(0, sentence2_token, segment2_token, mask2_token, mask2_pos, pad2_token)
        exam.append(example1)
        exam.append(example2)
    return exam




def clean_data(file_path, target_file, word_to_id, seq_len, mask_rate):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        writer = tf.io.TFRecordWriter(target_file)
        for line in data:
            new_line = zhconv.convert(line, 'zh-cn')
            sentences = new_line.strip('\n').strip('。').split('。')
            if len(sentences) < 2:
                continue
            for i in range(len(sentences) - 1):
                tempdata = get_record_data(sentences[i], sentences[i + 1], word_to_id, seq_len, mask_rate)
                for val in tempdata:
                    writer.write(val.SerializeToString())
        f.close()
        writer.close()
    else:
        print("no such file or dir")
        return

class Embedding_layer(tf.keras.layers.Layer):
    def __init__(self, vocb_size, embedding_size=512, max_seq_len=128, segment_size=2, dropout_prob=0.0, **kwargs):
        super(Embedding_layer, self).__init__(**kwargs)
        self.vocb_size = vocb_size
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len
        self.segment_size = segment_size
        self.dropout_prob = dropout_prob

    def build(self, input_shape):
        self.token_embedding = tf.keras.layers.Embedding(input_dim=self.vocb_size,
                                                         output_dim=self.embedding_size,
                                                         embeddings_initializer=tf.keras.initializers.TruncatedNormal(),
                                                         dtype=tf.float32, name="layers1")
        self.segment_embedding = tf.keras.layers.Embedding(input_dim=self.segment_size, output_dim=self.embedding_size,
                                                           embeddings_initializer=tf.keras.initializers.TruncatedNormal(),
                                                           dtype=tf.float32, name="layers2")
        #使用固定位置编码，暂时未实现其他位置编码
        self.positional_embedding = self.add_weight(shape=(self.max_seq_len, self.embedding_size),
                                                    initializer=tf.keras.initializers.TruncatedNormal(),
                                                    dtype=tf.float32, name="layers4")
        self.output_layer_norm = tf.keras.layers.LayerNormalization()
        self.output_dropout = tf.keras.layers.Dropout(self.dropout_prob)
        super(Embedding_layer, self).build(input_shape)

def attention(q, k, v, mask):
  '''
  通过使用-1e9，使mask位置置零
  '''
    matmul_qk = tf.matmul(q, k, transpose_b=True) 
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits + (tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32) * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) 
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  '''
  严格意义上，并不清除分为多头的意义何在，cnn分为多头是多了参数量，
  但是在该处，参数量并没有增加，只是在维度上的运算
  '''
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, mask):
        batch_size = tf.shape(x)[0]
        query = self.wq(x)  
        key = self.wk(x)  
        value = self.wv(x)  
        query = self.split_heads(query, batch_size)  
        key = self.split_heads(key, batch_size) 
        value = self.split_heads(value, batch_size)  
        scaled_attention, attention_weights = attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output


class Transformer(tf.keras.layers.Layer):
  '''
  多头注意力+layernorm+残差+dense+dense
  '''
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(Transformer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model) 
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, mask, training=None):
        attn_output = self.mha(x, mask)  
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output) 

        ffn_output = self.ffn(out1) 
        ffn_output = self.dropout2(ffn_output, training=training)
        out = self.layernorm2(out1 + ffn_output) 

        return out
class Bert(tf.keras.Model):
  '''
  embedding+多个transformer，最后输出3项结果用于不同任务
  '''
    def __init__(self, vocab_size, embedding_size, max_seq_len, segment_size, num_transformer_layers,
                 num_attention_heads, intermediate_size, **kwargs):
        super(Bert, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len
        self.segment_size = segment_size
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.embedding = Embedding_layer(vocb_size=self.vocab_size, embedding_size=self.embedding_size,
                                         max_seq_len=self.max_seq_len,
                                         segment_size=self.segment_size, )
        self.transformer1 = Transformer(d_model=self.embedding_size, num_heads=self.num_attention_heads,
                                               dff=self.intermediate_size)
        self.transformer2 = Transformer(d_model=self.embedding_size, num_heads=self.num_attention_heads,
                                               dff=self.intermediate_size)
        self.transformer3 = Transformer(d_model=self.embedding_size, num_heads=self.num_attention_heads,
                                               dff=self.intermediate_size)
        self.transformer4 = Transformer(d_model=self.embedding_size, num_heads=self.num_attention_heads,
                                               dff=self.intermediate_size)
        self.transformer5 = Transformer(d_model=self.embedding_size, num_heads=self.num_attention_heads,
                                               dff=self.intermediate_size)
        self.transformer6 = Transformer(d_model=self.embedding_size, num_heads=self.num_attention_heads,
                                               dff=self.intermediate_size)
        self.transformer7 = Transformer(d_model=self.embedding_size, num_heads=self.num_attention_heads,
                                               dff=self.intermediate_size)
        self.transformer8 = Transformer(d_model=self.embedding_size, num_heads=self.num_attention_heads,
                                               dff=self.intermediate_size)
        self.nsp_predictor = tf.keras.layers.Dense(2)

    def call(self, inputs, training=None):
        batch_x, batch_mask, batch_segment = inputs
        x = self.embedding((batch_x, batch_segment))
        x = self.transformer1(x, mask=batch_mask, training=training)
        x = self.transformer2(x, mask=batch_mask, training=training)
        x = self.transformer3(x, mask=batch_mask, training=training)
        x = self.transformer4(x, mask=batch_mask, training=training)
        x = self.transformer5(x, mask=batch_mask, training=training)
        x = self.transformer6(x, mask=batch_mask, training=training)
        x = self.transformer7(x, mask=batch_mask, training=training)
        x = self.transformer8(x, mask=batch_mask, training=training)
        first_token_tensor = x[:, 0, :]
        is_next_predict = self.nsp_predictor(first_token_tensor)
        word_mask_predict = tf.matmul(x, self.embedding.token_embedding.embeddings, transpose_b=True)
        sequence_output = x

        return is_next_predict, word_mask_predict, sequence_output
class BERT_Loss(tf.keras.layers.Layer):
'''
loss=mask_loss+next_sentences_loss
'''
    def __init__(self):
        super(BERT_Loss, self).__init__()

    def call(self, inputs):
        (mlm_predict, batch_mlm_mask, origin_x, nsp_predict, batch_y) = inputs
        x_pred = tf.nn.softmax(mlm_predict, axis=-1)
        mlm_loss = tf.keras.losses.sparse_categorical_crossentropy(origin_x, x_pred)
        mlm_loss = tf.math.reduce_sum(mlm_loss * batch_mlm_mask, axis=-1) / (
                    tf.math.reduce_sum(batch_mlm_mask, axis=-1) + 1)
        y_pred = tf.nn.softmax(nsp_predict, axis=-1)
        nsp_loss = tf.keras.losses.sparse_categorical_crossentropy(batch_y, y_pred)

        return nsp_loss, mlm_loss



def calculate_pretrain_task_accuracy(nsp_predict, mlm_predict, batch_mlm_mask, origin_x, batch_y):
  '''
  使用loss和acc作为每个batch的指标
  '''
    y_predict = tf.math.argmax(nsp_predict, axis=-1)
    nsp_accuracy = tf.keras.metrics.Accuracy()
    nsp_accuracy.update_state(y_predict, batch_y)
    nsp_accuracy = nsp_accuracy.result().numpy()

    batch_mlm_mask = tf.cast(batch_mlm_mask, dtype=tf.int32)
    index = tf.where(batch_mlm_mask == 1)
    x_predict = tf.math.argmax(mlm_predict, axis=-1)
    x_predict = tf.gather_nd(x_predict, index)
    x_real = tf.gather_nd(origin_x, index)
    mlm_accuracy = tf.keras.metrics.Accuracy()
    mlm_accuracy.update_state(x_predict, x_real)
    mlm_accuracy = mlm_accuracy.result().numpy()

    return nsp_accuracy, mlm_accuracy
#加载词表
w_i, i_w, vocb_len = get_vocb("/content/drive/MyDrive/wiki_data/wiki_00", "/content/drive/MyDrive/wiki_data/bert_vocb.txt", 1)
#清洗数据，生成tfrecord
clean_data("/content/drive/MyDrive/wiki_data/wiki_00", "/content/drive/MyDrive/wiki_data/bert.tfrecords", w_i, 128, 0.10)
#构建数据解码器
def decode_fn(record_bytes):
    feature_map = {
                   "label_y": tf.io.FixedLenFeature([1], dtype=tf.float32),
                   "raw_token": tf.io.FixedLenFeature([128], dtype=tf.float32),
                   "segment_token": tf.io.FixedLenFeature([128], dtype=tf.float32),
                   "mask_token": tf.io.FixedLenFeature([128], dtype=tf.float32),
                   "mask_pos": tf.io.FixedLenFeature([128], dtype=tf.float32),
                   "pad_token": tf.io.FixedLenFeature([128], dtype=tf.float32)
                   }
    tempdata = tf.io.parse_single_example(record_bytes, features=feature_map)
    return tempdata
#加载数据
data_set = tf.data.TFRecordDataset(["/content/drive/MyDrive/wiki_data/bert.tfrecords"]).shuffle(buffer_size=1000,reshuffle_each_iteration=True)
data_use = data_set.map(decode_fn)
data_use = data_use.batch(96)
#定义bert超参数
vocab_size = vocb_len
print(vocab_size)
embedding_size = 512
max_seq_len = 512
segment_size = 2
num_transformer_layers = 12
num_attention_heads = 12
intermediate_size = 2048
#初始化bert
model = Bert(vocab_size, embedding_size, max_seq_len,
             segment_size, num_transformer_layers, num_attention_heads, intermediate_size, )
optimizer = tfa.optimizers.LAMB(learning_rate=5e-4)
loss_fn = BERT_Loss()
#管理checkpoint
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint("/content/drive/MyDrive/wiki_data/model"))
manager = tf.train.CheckpointManager(checkpoint, directory="/content/drive/MyDrive/wiki_data/model", max_to_keep=5)
#自定义训练过程
EPOCH = 100
for epoch in range(EPOCH):
    it = iter(data_use)
    for step in range(100000):    
        mydata = next(it, None)
        if mydata == None:
            break
        batch_x = mydata["mask_token"]
        batch_mlm_mask =mydata["mask_pos"]
        origin_x = mydata["raw_token"]
        batch_segment = mydata["segment_token"]
        batch_padding_mask = mydata["pad_token"]
        batch_y = mydata["label_y"]
        with tf.GradientTape() as t:
            nsp_predict, mlm_predict, sequence_output = model((batch_x, batch_padding_mask, batch_segment),training=True)
            nsp_loss, mlm_loss = loss_fn((mlm_predict, batch_mlm_mask, origin_x, nsp_predict, batch_y))
            nsp_loss = tf.reduce_mean(nsp_loss)
            mlm_loss = tf.reduce_mean(mlm_loss)
            loss = nsp_loss + mlm_loss
        gradients = t.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        nsp_acc, mlm_acc = calculate_pretrain_task_accuracy(nsp_predict, mlm_predict, batch_mlm_mask, origin_x, batch_y)
        #每50步输出一次训练信息
        if step % 50 == 0:
            print(
                'Epoch {}, step {}, loss {:.4f}, mask_loss {:.4f}, mask_acc {:.4f}, next_sentence_loss {:.4f}, next_sentence_acc {:.4f}'.format(
                    epoch, step, loss.numpy(),
                    mlm_loss.numpy(),
                    mlm_acc,
                    nsp_loss.numpy(), nsp_acc
                ))
    #每个epoch保存一次模型
    path = manager.save(checkpoint_number=epoch)
