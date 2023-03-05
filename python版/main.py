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
