# coding=utf-8
import tensorflow as tf

"""
    中文事件提取 卷积神经网络模型
    john.zhang 2016-11-28
"""


class ace_cnn_model():
    """
    中文事件的发现 由以后候选词和一个触发词是否构成事件 这里是另外一种比较简单的模型
    Relation Classification via Convolutional Deep Neural Network
    这篇文章句子特征用的是每一个词的上下文(包含当前词)作为整体的输入
    """

    def __init__(self, sentence_length=30, num_labels=10, windows=3, vocab_size=2048, word_embedding_size=100,
                 pos_embedding_size=10, filter_sizes=[3, 4, 5], filter_num=50, batch_size=2):
        """
        :param sentence_length: 输入句子的长度
        :param num_labels: 类别数目
        :param windows: 窗口的大小
        :param vocab_size: 训练集中词的数目
        :param word_embedding_size: 词嵌入维数
        :param pos_embedding_size: 位置嵌入维数
        :param trigger_vec: 触发词向量
        :param candidate_vec: 候选词向量
        :param filter_sizes: 滤波器尺寸
        :param filter_num: 滤波器数目
        """
        # 输入标签占位符
        # [batch_size, num_labels]
        input_y = tf.placeholder(tf.float32, shape=[batch_size, num_labels], name="input_y")
        self.input_y = input_y
        # 触发词位置向量
        input_t_pos = tf.placeholder(tf.int32, shape=[batch_size, sentence_length], name="input_t_pos")
        self.input_t_pos = input_t_pos
        # 候选词位置向量
        input_c_pos = tf.placeholder(tf.int32, shape=[batch_size, sentence_length], name="input_c_pos")
        self.input_c_pos = input_c_pos

        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.dropout_keep_prob = dropout_keep_prob

        # 句子 上下文特征向量 [batch_size, sentence_length, windows]
        sentence_features = tf.placeholder(tf.int32, shape=[batch_size, sentence_length, windows],
                                           name="sentences_features")
        self.sentence_features = sentence_features
        # 候选词的上下文组成的特征向量
        input_c_context = tf.placeholder(tf.int32, shape=[batch_size, windows], name="input_c_context")
        self.input_c_context = input_c_context
        # 　触发词的上下文组成的特征向量
        input_t_context = tf.placeholder(tf.int32, shape=[batch_size, windows], name="input_t_context")
        self.input_t_context = input_t_context
        # 生成word_embedding 这部分操作相对比较简单 建议用cpu做
        with tf.device('/gpu:0'), tf.name_scope("word_embedding_layer"):
            # 词表 [vocab_size, embedding_size]
            W = tf.Variable(tf.random_normal(shape=[vocab_size, word_embedding_size], mean=0.0, stddev=0.5),
                            name="word_table")
            # 句子特征向量 [batch_size, sentence_length, windows, word_embedding]
            sentences_features_vec = tf.nn.embedding_lookup(W, sentence_features)
            #             print sentence_features
            #             print sentences_features_vec
            #             根据 候选词位置 触发词位置 选取位置向量
            #             在(-sentence_length+1,sentence_length-1)之间一共2*(sentence_length-1)+1个数
            #             look_up 变成合适的正整数
            input_t_pos_t = input_t_pos + (sentence_length - 1)
            Tri_pos = tf.Variable(
                tf.random_normal(shape=[2 * (sentence_length - 1) + 1, pos_embedding_size], mean=0.0, stddev=0.5),
                name="tri_pos_table")
            input_t_pos_vec = tf.nn.embedding_lookup(Tri_pos, input_t_pos_t)
            #             look_up 变成合适的正整数
            input_c_pos_c = input_c_pos + (sentence_length - 1)
            Can_pos = tf.Variable(
                tf.random_normal(shape=[2 * (sentence_length - 1) + 1, pos_embedding_size], mean=0.0, stddev=0.5),
                name="candidate_pos_table")
            input_c_pos_vec = tf.nn.embedding_lookup(Can_pos, input_c_pos_c)
            #             print input_t_pos_vec
            #             print input_c_pos_vec
            # 对句子特征(每一个词由当前词以及其上下文组成)，进行reshape处理，目的是为了与其它特征进行整合
            # [batch_size, sentence_length, windows, word_embedding] -> [batch_size, sentence_length, windows*word_embedding]
            sentences_features_vec_flat = tf.reshape(sentences_features_vec,
                                                     [-1, sentence_length, windows * word_embedding_size])
            # 将距离特征和句子的词特征构成一个整理的特征 作为卷积神经网络的输入
            # [batch_size, sentence_length, word_embedding_size+2*pos_size]
            input_sentence_vec = tf.concat(2, [sentences_features_vec_flat, input_t_pos_vec, input_c_pos_vec])
            # CNN支持4d输入 因此增加一维向量 用于表示输入通道数目
            intput_sentence_vec_expanded = tf.expand_dims(input_sentence_vec, -1)
            # 词汇特征 lexical leval features
            # 候选词极其上下文 [batch_size, windows, word_embedding_size]
            input_c_context_vec = tf.nn.embedding_lookup(W, input_c_context)
            # 触发词及其上下文 [batch_size, windows, word_embedding_size]
            input_t_context_vec = tf.nn.embedding_lookup(W, input_t_context)
            # print input_sentence_vec
            # print input_c_context_vec
            # print input_t_context_vec
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.device('/gpu:0'), tf.name_scope('conv-maxpool-%s' % filter_size):
                # 这里的句子特征考虑的上当前词 以及上下文
                filter_shape = [filter_size, 3 * word_embedding_size + 2 * pos_embedding_size, 1, filter_num]
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
        # 多通道的数据合并
        h_pool = tf.concat(3, pooled_outputs)
        #         print sentences_features_vec_flat
        #         print h_pool
        # 展开送入下一层分类器
        # [batch_size, num_filters_total]
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        # print h_pool_flat
        # 候选词极其上下文 触发词极其上下文 [batch_size, windows, word_embedding_size] => [batch_size, windows*word_embedding_size]
        # [batch_size, windows * word_embedding_size]
        input_c_context_vec_flat = tf.reshape(input_c_context_vec, [-1, windows * word_embedding_size])
        # print input_c_context_vec_flat
        # [batch_size, windows * word_embedding_size]
        input_t_context_vec_flat = tf.reshape(input_t_context_vec, [-1, windows * word_embedding_size])
        # print input_t_context_vec_flat
        # print h_pool_flat
        # 将lexical leval features和sentence leval features组合
        all_input_fatures = tf.concat(1, [input_c_context_vec_flat, input_t_context_vec_flat, h_pool_flat])
        # 总体的分类器 经过一层dropout 然后再送入softmax
        with tf.device('/gpu:0'), tf.name_scope('dropout'):
            all_fatures = tf.nn.dropout(all_input_fatures, dropout_keep_prob)
        #
        # print all_fatures
        # 分类器
        with tf.device('/gpu:0'), tf.name_scope('softmax'):
            # num_filters_total是卷积之后的结果 sentence level features
            # 2*windows*word_embedding_size是lexical leval features 具体的请看论文当中的详细介绍
            W = tf.Variable(tf.truncated_normal([num_filters_total+2*windows*word_embedding_size, num_labels], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_labels]), name="b")
            scores = tf.nn.xw_plus_b(all_fatures, W, b, name="scores")
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
"""
测试程序 john.zhang 2016-11-28 已验证 程序可以运行
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
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "ace_cnn_model_02", timestamp))
        print("Writing to {}\n".format(out_dir))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)  # 最大支持存储100个模型
        # 初始化　变量
        sess.run(tf.initialize_all_variables())
        def train_step(input_x, input_y, input_t, input_c, input_t_pos, input_c_pos, dropout_keep_prob
                       ,sentence_features,input_t_context,input_c_context):
            feed_dict = {
                # model.input_x: input_x,
                model.input_y: input_y,
                # model.input_t:input_t,
                # model.input_c:input_c,
                model.input_t_pos: input_t_pos,
                model.input_c_pos: input_c_pos,
                model.dropout_keep_prob: dropout_keep_prob,
                model.sentence_features : sentence_features,
                model.input_t_context:input_t_context,
                model.input_c_context:input_c_context
            }
            _, loss, accuracy = sess.run(
                [train_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: , loss {:g}, acc {:g}".format(time_str, loss, accuracy))


        # 测试阶段不需要计算梯度 也不需要进行权值更新 仅仅需要计算acc的值
        def eval_step(input_x, input_y, input_t, input_c, input_t_pos, input_c_pos, dropout_keep_prob
                      , sentence_features, input_t_context, input_c_context):
            feed_dict = {
                # model.input_x: input_x,
                model.input_y: input_y,
                # model.input_t:input_t,
                # model.input_c:input_c,
                model.input_t_pos: input_t_pos,
                model.input_c_pos: input_c_pos,
                model.dropout_keep_prob: dropout_keep_prob,
                model.sentence_features: sentence_features,
                model.input_t_context: input_t_context,
                model.input_c_context: input_c_context
            }
            accuracy, predicts = sess.run([model.accuracy, model.predicts], feed_dict)
            print ("eval accuracy:{}".format(accuracy))
            return predicts

# //sentences_fatures, c_context, t_context, pos_tag
        for i in range(num_epochs):
            for j in range(datas.instances_size // data_batch_size):
                x, t, c, y, pos_c, pos_t, sentences_f, c_context, t_context, _ = datas.next_cnn_data()
                train_step(input_x=x, input_y=y, input_t=t, input_c=c, input_c_pos=pos_c, input_t_pos=pos_t,
                           dropout_keep_prob=0.8, sentence_features=sentences_f, input_t_context= t_context,
                           input_c_context = c_context)
        # john.zhang 2016-12-04 将训练集当作测试集
        print "----------------------------华丽的分割线-----------------------------------------"
        x, t, c, y, pos_c, pos_t, sentences_f, c_context, t_context, _ = datas.eval_cnn_data()
        predicts = eval_step(input_x=x, input_y=y, input_t=t, input_c=c, input_c_pos=pos_c, input_t_pos=pos_t,
                  dropout_keep_prob=1.0, sentence_features=sentences_f, input_t_context= t_context,
                  input_c_context = c_context)

        # 输出测试结果
        for i in range(len(x)):
            print "输入数据：{}".format(", ".join(map(lambda h: datas.all_words[h], x[i])))
            print "触发词：{}".format(", ".join(map(lambda h: datas.all_words[h], t[i])))
            print "候选词：{}".format(", ".join(map(lambda h: datas.all_words[h], c[i])))
            print "预测事件类别:{}".format(predicts[i])
            print "----------------------------华丽的分割线-----------------------------------------"