# coding=utf-8
import tensorflow as tf
"""
    中文事件 卷积神经网络模型
    john.zhang 2016-11-26
"""

class ace_cnn_model():
    """
    中文事件的发现 由以后候选词和一个触发词是否构成事件
    """
    def __init__(self, sentence_length=30, num_labels=10, vocab_size=2048,word_embedding_size=100, pos_embedding_size = 10,filter_sizes= [3, 4, 5], filter_num=100, batch_size = 2):
        """
        :param sentence_length: 输入句子的长度
        :param num_labels: 类别数目
        :param vocab_size: 训练集中词的数目
        :param word_embedding_size: 词嵌入维数
        :param pos_embedding_size: 位置嵌入维数
        :param trigger_vec: 触发词向量
        :param candidate_vec: 候选词向量
        :param filter_sizes: 滤波器尺寸
        :param filter_num: 滤波器数目
        """
        # 输入占位符
        # [batch_size, sentence_length] 整个句子
        input_x = tf.placeholder(tf.int32, shape=[batch_size, sentence_length], name="input_x")
        self.input_x = input_x
        # [batch_size, num_labels]
        input_y = tf.placeholder(tf.float32, shape=[batch_size, num_labels], name="input_y")
        self.input_y = input_y
        # 触发词
        input_t = tf.placeholder(tf.int32, shape=[batch_size, sentence_length], name="input_t")
        self.input_t = input_t
        # 候选词
        input_c = tf.placeholder(tf.int32, shape=[batch_size, sentence_length], name="input_c")
        self.input_c = input_c
        # 触发词位置向量
        input_t_pos = tf.placeholder(tf.int32, shape=[batch_size, sentence_length], name="input_t_pos")
        self.input_t_pos = input_t_pos
        # 候选词位置向量
        input_c_pos = tf.placeholder(tf.int32, shape=[batch_size, sentence_length], name="input_c_pos")
        self.input_c_pos = input_c_pos
        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.dropout_keep_prob = dropout_keep_prob
        # 正则化代价
        # l2_loss = tf.constant(0.0)

        # 生成word_embedding 这部分操作相对比较简单 建议用cpu做
        with tf.device('/gpu:0'), tf.name_scope("word_embedding_layer"):
            # 词表 [vocab_size, embedding_size]
            W = tf.Variable(tf.random_normal(shape=[vocab_size, word_embedding_size],mean=0.0, stddev=0.5), name="word_table")
            # 依据词id查找词向量 [batch_size, sequence_length, embedding_size]
            # 句子特征
            input_word_vec = tf.nn.embedding_lookup(W, input_x)
            # 候选词及触发词
            input_c_vec = tf.nn.embedding_lookup(W, input_c)
            input_t_vec = tf.nn.embedding_lookup(W, input_t)
            # 根据 候选词位置 触发词位置 选取位置向量
            # 在(-sentence_length+1,sentence_length-1)之间一共2*(sentence_length-1)+1个数
            # look_up 变成合适的正整数
            input_t_pos_t = input_t_pos + (sentence_length-1)
            Tri_pos = tf.Variable(tf.random_normal(shape=[2*(sentence_length-1)+1, pos_embedding_size],mean=0.0, stddev=0.5), name="tri_pos_table")
            input_t_pos_vec = tf.nn.embedding_lookup(Tri_pos, input_t_pos_t)
            # look_up 变成合适的正整数
            input_c_pos_c = input_c_pos + (sentence_length-1)
            Can_pos = tf.Variable(tf.random_normal(shape=[2*(sentence_length-1)+1, pos_embedding_size],mean=0.0, stddev=0.5), name="candidate_pos_table")
            input_c_pos_vec = tf.nn.embedding_lookup(Can_pos, input_c_pos_c)
            # 对句子向量 触发词 候选词 触发词位置 候选词位置 合并 形成一整个句子
            intput_sentence_vec = tf.concat(2, [input_word_vec, input_c_vec, input_t_vec, input_t_pos_vec, input_c_pos_vec])
            # CNN支持4d输入 因此增加一维向量 用于表示输入通道数目
            intput_sentence_vec_expanded = tf.expand_dims(intput_sentence_vec, -1)
        #print intput_sentence_vec_expanded
        # 多种滤波器的卷积加池化结果
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.device('/gpu:0') , tf.name_scope('conv-maxpool-%s'%filter_size):
                filter_shape = [filter_size, 3*word_embedding_size+2*pos_embedding_size, 1, filter_num]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="b")
                # 卷积运算
                conv = tf.nn.conv2d(
                    intput_sentence_vec_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                #print h
                # 最大化池化 暂时不用动态池化
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sentence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        # 使用到的所有滤波器数目(输出的通道数目)
        num_filters_total = filter_num * len(filter_sizes)
        # print pooled_outputs
        # 多通道的数据合并
        h_pool = tf.concat(3, pooled_outputs)
        # print h_pool
        # 展开送入下一层分类器
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        # print h_pool_flat
        with tf.device('/gpu:0'), tf.name_scope('dropout'):
            h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
        # print h_drop
        # 分类器
        with tf.device('/gpu:0'), tf.name_scope('softmax'):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_labels], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_labels]), name="b")
            scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
            predicts = tf.arg_max(scores, dimension=1, name="predicts")
            self.scores = scores
            self.predicts = predicts
        # print scores
        # print input_y
        # 模型的代价函数 交叉熵代价函数
        with tf.device('/gpu:0'), tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(scores, input_y)
            loss = tf.reduce_mean(entropy)
            self.loss = loss
        # 准确度 用于每一次训练时调用
        with tf.device('/gpu:0'), tf.name_scope("accuracy"):
            correct = tf.equal(predicts, tf.argmax(input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")
            self.accuracy = accuracy
        # print accuracy
"""
测试程序 john.zhang 2016-11-26 已验证 程序可以运行
"""

# 数据集配置参数
from DataSets import datasets
import time
import datetime
import os
# 文件信息
file = 'datas_ace.txt'
# 生成文件的存储位置
store_path = "ace_data_2016_12_02"
# batch_size的大小
data_batch_size = 20
# 句子的最大长度
max_sequence_length = 20
# 选取的上下文窗口的大小
windows = 3
# 数据集
datas = datasets(file=file, store_path=store_path, batch_size=data_batch_size, max_sequence_length=max_sequence_length,
                 windows=windows)

# 神经网络模型的一些参数
# 模型的最大长度
sentence_length = max_sequence_length
num_labels = datas.labels_size  # 分类类别数目
vocab_size = datas.words_size  # 训练集中词的数目
word_embedding_size = 100  # 词嵌入维数
pos_embedding_size = 10  # 位置嵌入维数
filter_sizes = [3, 4, 5]  # 滤波器大小
filter_num = 100  # 滤波器大小
batch_size = None  # tensorflow中支持为定义长度的标记符合
lr = 1e-3  # 学习率
num_epochs = 20
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        # 模型文件
        model = ace_cnn_model(sentence_length=sentence_length, num_labels=num_labels, vocab_size=vocab_size,
                              word_embedding_size=word_embedding_size,
                              pos_embedding_size=pos_embedding_size, filter_sizes=filter_sizes, filter_num=filter_num,
                              batch_size=batch_size)
        # 模型优化算法
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "ace_cnn_model01", timestamp))
        print("Writing to {}\n".format(out_dir))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)  # 最大支持存储100个模型
        # 初始化　变量
        sess.run(tf.initialize_all_variables())


        def train_step(input_x, input_y, input_t, input_c, input_t_pos, input_c_pos, dropout_keep_prob):
            feed_dict = {
                model.input_x: input_x,
                model.input_y: input_y,
                model.input_t:input_t,
                model.input_c:input_c,
                model.input_t_pos: input_t_pos,
                model.input_c_pos: input_c_pos,
                model.dropout_keep_prob: dropout_keep_prob
            }
            _, loss, accuracy = sess.run(
                [train_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: , loss {:g}, acc {:g}".format(time_str, loss, accuracy))


        # 测试阶段不需要计算梯度 也不需要进行权值更新 仅仅需要计算acc的值
        def eval_step(input_x, input_y, input_t, input_c, input_t_pos, input_c_pos, dropout_keep_prob):
            feed_dict = {
                model.input_x: input_x,
                model.input_y: input_y,
                model.input_t:input_t,
                model.input_c:input_c,
                model.input_t_pos: input_t_pos,
                model.input_c_pos: input_c_pos,
                model.dropout_keep_prob: dropout_keep_prob
            }
            accuracy, predicts = sess.run([model.accuracy, model.predicts], feed_dict)
            print ("eval accuracy:{}".format(accuracy))
            return predicts


        for i in range(num_epochs):
            for j in range(datas.instances_size // data_batch_size):
                x, t, c, y, pos_c, pos_t, _, _, _, _ = datas.next_cnn_data()
                train_step(input_x=x, input_y=y, input_t=t, input_c=c, input_c_pos=pos_c, input_t_pos=pos_t,
                           dropout_keep_prob=0.8)
        # john.zhang 2016-12-04 将训练集当作测试集
        print "----------------------------华丽的分割线-----------------------------------------"
        x, t, c, y, pos_c, pos_t, _, _, _, _ = datas.eval_cnn_data()
        predicts = eval_step(input_x=x, input_y=y, input_t=t, input_c=c, input_c_pos=pos_c, input_t_pos=pos_t,
                  dropout_keep_prob=1.0)
        # 输出测试结果
        for i in range(len(x)):
            print "输入数据：{}".format(", ".join(map(lambda h: datas.all_words[h], x[i])))
            print "触发词：{}".format(", ".join(map(lambda h: datas.all_words[h], t[i])))
            print "候选词：{}".format(", ".join(map(lambda h: datas.all_words[h], c[i])))
            print "预测事件类别:{}".format(predicts[i])
            print "----------------------------华丽的分割线-----------------------------------------"