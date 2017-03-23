# coding=utf-8
import tensorflow as tf

"""
    中文事件提取 卷积神经网络模型 多任务分类
    john.zhang 2016-12-15
"""

# coding=utf-8
import tensorflow as tf

"""
    中文事件提取 卷积神经网络模型
    john.zhang 2016-12-15
"""


class ace_cnn_model():
    """
    中文事件的发现 由以后候选词和一个触发词是否构成事件 这里是另外一种比较简单的模型
    Relation Classification via Convolutional Deep Neural Network
    这篇文章句子特征用的是每一个词的上下文(包含当前词)作为整体的输入
    """

    def __init__(self, sentence_length=30, event_num_labels=10, role_num_labels=6, windows=3, vocab_size=2048,
                 word_embedding_size=100,
                 pos_embedding_size=10, filter_sizes=[3, 4, 5], filter_num=50, batch_size=2):
        """
        :param sentence_length: 输入句子的长度
        :param event_num_labels: 事件类别数目
        :param role_num_labels: 角色类别数目
        :param windows: 窗口的大小
        :param vocab_size: 训练集中词的数目
        :param word_embedding_size: 词嵌入维数
        :param pos_embedding_size: 位置嵌入维数
        :param trigger_vec: 触发词向量
        :param candidate_vec: 候选词向量
        :param filter_sizes: 滤波器尺寸
        :param filter_num: 滤波器数目
        """
        input_x = tf.placeholder(tf.int32, shape=[batch_size, sentence_length], name="input_x")
        self.input_x = input_x
        # 输入事件标签占位符
        # [batch_size, event_num_labels]
        input_event_y = tf.placeholder(tf.float32, shape=[batch_size, event_num_labels], name="input_event_y")
        self.input_event_y = input_event_y
        # 输入角色标签占位符
        # [batch_size, role_num_labels]
        input_role_y = tf.placeholder(tf.float32, shape=[batch_size, role_num_labels], name="input_role_y")
        self.input_role_y = input_role_y

        # 触发词位置向量
        input_t_pos = tf.placeholder(tf.int32, shape=[batch_size, sentence_length], name="input_t_pos")
        self.input_t_pos = input_t_pos
        # 候选词位置向量
        input_c_pos = tf.placeholder(tf.int32, shape=[batch_size, sentence_length], name="input_c_pos")
        self.input_c_pos = input_c_pos

        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.dropout_keep_prob = dropout_keep_prob

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
            # 句子特征向量 [batch_size, sentence_length, word_embedding]
            input_word_vec = tf.nn.embedding_lookup(W, input_x)
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
            # 将距离特征和句子的词特征构成一个整理的特征 作为卷积神经网络的输入
            # [batch_size, sentence_length, word_embedding_size+2*pos_size]
            input_sentence_vec = tf.concat(2, [input_word_vec, input_t_pos_vec, input_c_pos_vec])
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
                # 这里的句子特征考虑的上当前词
                filter_shape = [filter_size, word_embedding_size + 2 * pos_embedding_size, 1, filter_num]
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
        # 原本的论文将lexical leval features和sentence leval features组合
        # 这里采用multi-task的思路 sentence leval features主要应用于事件的发现
        # lexical leval features主要应用于角色的分类
        input_sentence_features = h_pool_flat
        input_lexical_features = tf.concat(1, [input_c_context_vec_flat, input_t_context_vec_flat])
        # 总体的分类器 经过一层dropout 然后再送入softmax
        with tf.device('/gpu:0'), tf.name_scope('dropout'):
            input_sentence_features_dropout = tf.nn.dropout(input_sentence_features, dropout_keep_prob)
            input_lexical_features_dropout = tf.nn.dropout(input_lexical_features, dropout_keep_prob)
        # 分类器
        with tf.device('/gpu:0'), tf.name_scope('softmax'):
            # num_filters_total是卷积之后的结果 sentence level features
            # 事件类别分类
            W1 = tf.Variable(tf.truncated_normal([num_filters_total, event_num_labels], stddev=0.1), name="W1")
            b1 = tf.Variable(tf.constant(0.1, shape=[event_num_labels]), name="b1")
            xw1 = tf.nn.xw_plus_b(input_sentence_features_dropout, W1, b1)
            scores_event = tf.nn.softmax(xw1, name="scores_event")
            # 2*windows*word_embedding_size是lexical leval features 具体的请看论文当中的详细介绍
            predicts_event = tf.arg_max(scores_event, dimension=1, name="predicts_event")
            self.scores_event = scores_event
            self.predicts_event = predicts_event
            # 角色类别分类
            W2 = tf.Variable(tf.truncated_normal(
                [num_filters_total + 2 * windows * word_embedding_size + event_num_labels, role_num_labels],
                stddev=0.1), name="W2")
            b2 = tf.Variable(tf.constant(0.1, shape=[role_num_labels]), name="b2")
            # 将句子特征 候选词和词性以及触发词和词性特征合并用于角色分类
            # input_sentence_features_dropout  sentence features
            # input_lexical_features_dropout  lexical_features
            # scores_event  事件类型分类的输出得分结果
            all_input_fatures = tf.concat(1, [input_lexical_features_dropout, input_sentence_features, scores_event])
            xw2 = tf.nn.xw_plus_b(all_input_fatures, W2, b2)
            scores_role = tf.nn.softmax(xw2, name="scores_role")
            predicts_role = tf.arg_max(scores_role, dimension=1, name="predicts_role")
            self.scores_role = scores_role
            self.predicts_role = predicts_role
        # 模型的代价函数 交叉熵代价函数
        with tf.device('/gpu:0'), tf.name_scope('loss'):
            # 事件的交叉熵代价函数
            entropy_event = tf.nn.softmax_cross_entropy_with_logits(scores_event, input_event_y)
            loss_event = tf.reduce_mean(entropy_event)
            self.loss_event = loss_event
            # 角色的交叉熵代价函数
            entropy_role = tf.nn.softmax_cross_entropy_with_logits(scores_role, input_role_y)
            loss_role = tf.reduce_mean(entropy_role)
            self.loss_role = loss_role
        # 准确度 用于每一次训练时调用
        with tf.device('/gpu:0'), tf.name_scope("accuracy"):
            # 事件的准确度
            correct_event = tf.equal(predicts_event, tf.argmax(input_event_y, 1))
            accuracy_event = tf.reduce_mean(tf.cast(correct_event, "float"), name="accuracy_event")
            self.accuracy_event = accuracy_event
            # 角色的准确度
            correct_role = tf.equal(predicts_role, tf.argmax(input_role_y, 1))
            accuracy_role = tf.reduce_mean(tf.cast(correct_role, "float"), name="accuracy_role")
            self.accuracy_role = accuracy_role

"""
测试程序
"""
"""
测试程序 john.zhang 2016-11-28 已验证 程序可以运行
"""
# 数据集配置参数
from DataSets_multi_task import datasets
import time
import datetime
import os

# 文件信息
file = 'datas_ace.txt'
# 生成文件的存储位置
store_path = "ace_data_2016_12_19"
# batch_size的大小
data_batch_size = 20
# 句子的最大长度
max_sequence_length = 25
# 选取的上下文窗口的大小
windows = 3
# 数据集
datas = datasets(file=file, store_path=store_path, batch_size=data_batch_size, max_sequence_length=max_sequence_length,
                 windows=windows)
# 神经网络模型的一些参数
# 模型的最大长度
sentence_length = max_sequence_length
event_num_labels = datas.labels_event_size  # 事件类别数目
role_num_labels = datas.labels_role_size  # 角色类别数目
vocab_size = datas.words_size  # 训练集中词的数目
word_embedding_size = 100  # 词嵌入维数
pos_embedding_size = 10  # 位置嵌入维数
filter_sizes = [3, 4, 5]  # 滤波器大小
filter_num = 100  # 滤波器大小
batch_size = None  # tensorflow中支持为定义长度的标记符合
lr = 1e-3  # 学习率
num_epochs = 1000
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        # 模型文件
        model = ace_cnn_model(sentence_length=sentence_length,
                              event_num_labels=event_num_labels,
                              role_num_labels=role_num_labels,
                              vocab_size=vocab_size,
                              word_embedding_size=word_embedding_size,
                              pos_embedding_size=pos_embedding_size,
                              filter_sizes=filter_sizes,
                              filter_num=filter_num,
                              batch_size=batch_size)
        # 模型优化算法  两种方式：
        # 方式一 采用两种优化器分别进行
        # 方式二 将两种误差函数合并一起进行
        optimizer_event = tf.train.AdamOptimizer(1e-3)
        grads_and_vars_event = optimizer_event.compute_gradients(model.loss_event)  # 事件优化器
        train_op_event = optimizer_event.apply_gradients(grads_and_vars_event)

        optimizer_role = tf.train.AdamOptimizer(1e-3)
        grads_and_vars_role = optimizer_role.compute_gradients(model.loss_role)  # 角色优化器
        train_op_role = optimizer_role.apply_gradients(grads_and_vars_role)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "ace_cnn_model_multi_task", timestamp))
        print("Writing to {}\n".format(out_dir))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)  # 最大支持存储100个模型
        # 初始化　变量
        sess.run(tf.initialize_all_variables())


        def train_step(input_x, input_event_y, input_role_y, input_t, input_c, input_t_pos, input_c_pos,
                       dropout_keep_prob
                       , sentence_features, input_t_context, input_c_context, train_op, loss, accuracy, epoch, stype):
            feed_dict = {
                model.input_x: input_x,  # 句子级别特征
                model.input_event_y: input_event_y,  # 事件类别
                model.input_role_y: input_role_y,  # 角色类别
                # model.input_t:input_t, # 触发词
                # model.input_c:input_c, # 候选词
                model.input_t_pos: input_t_pos,  # 触发词位置向量
                model.input_c_pos: input_c_pos,  # 候选词位置向量
                model.dropout_keep_prob: dropout_keep_prob,  # drop_out 概率
                #                 model.sentence_features : sentence_features, # 句子级别的特征 这里每一个词都有其上下文组成
                model.input_t_context: input_t_context,  # 触发词以及其上下文
                model.input_c_context: input_c_context  # 候选词以及上下文
            }
            _, loss, accuracy = sess.run(
                [train_op, loss, accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("stype:{}, {}: , loss {:g}, acc {:g}".format(stype, epoch, loss, accuracy))


        # 测试阶段不需要计算梯度 也不需要进行权值更新 仅仅需要计算acc的值
        def eval_step(input_x, input_event_y, input_role_y, input_t, input_c, input_t_pos, input_c_pos,
                      dropout_keep_prob
                      , sentence_features, input_t_context, input_c_context, accuracy, predicts, stype):
            feed_dict = {
                model.input_x: input_x,  # 句子级别特征
                model.input_event_y: input_event_y,  # 事件类别
                model.input_role_y: input_role_y,  # 角色类别
                # model.input_t:input_t,  # 触发词
                # model.input_c:input_c,  # 候选词
                model.input_t_pos: input_t_pos,  # 触发词位置向量
                model.input_c_pos: input_c_pos,  # 候选词位置向量
                model.dropout_keep_prob: dropout_keep_prob,  # drop_out 概率
                #                 model.sentence_features: sentence_features,  # 句子级别的特征 这里每一个词都有其上下文组成
                model.input_t_context: input_t_context,  # 触发词以及其上下文
                model.input_c_context: input_c_context  # 候选词以及上下文
            }
            accuracy, predicts = sess.run([accuracy, predicts], feed_dict)
            print ("{} eval accuracy:{}".format(stype, accuracy))
            return predicts


        for i in range(num_epochs):
            for j in range(datas.instances_size // data_batch_size):
                x, t, c, y_e, y_r, pos_c, pos_t, sentences_f, c_context, t_context, _ = datas.next_cnn_data()
            # print ", ".join(map(lambda t:datas.all_words[t]  , x[0]))
            #     事件类型预测
            train_step(input_x=x, input_event_y=y_e, input_role_y=y_r,
                       input_t=t, input_c=c, input_c_pos=pos_c, input_t_pos=pos_t,
                       dropout_keep_prob=0.8, sentence_features=sentences_f,
                       input_t_context=t_context, input_c_context=c_context,
                       train_op=train_op_event, loss=model.loss_event, accuracy=model.accuracy_event,
                       epoch=i, stype="event"
                       )
            #     角色类型预测
            train_step(input_x=x, input_event_y=y_e, input_role_y=y_r,
                       input_t=t, input_c=c, input_c_pos=pos_c, input_t_pos=pos_t,
                       dropout_keep_prob=0.8, sentence_features=sentences_f,
                       input_t_context=t_context, input_c_context=c_context,
                       train_op=train_op_role, loss=model.loss_role, accuracy=model.accuracy_role,
                       epoch=i, stype="role"
                       )
            # if i / 99 == 0:
            #     path = saver.save(sess, checkpoint_prefix, global_step=i)
        # john.zhang 2016-12-16 最后50个instance作为测试集 用于测试数据
        print "----------------------------华丽的分割线-----------------------------------------"
        x, t, c, y_e, y_r, pos_c, pos_t, sentences_f, c_context, t_context, _ = datas.eval_cnn_data()
        # 事件类型预测
        predicts_event = eval_step(input_x=x, input_event_y=y_e, input_role_y=y_r, input_t=t, input_c=c,
                                   input_c_pos=pos_c, input_t_pos=pos_t,
                                   dropout_keep_prob=1.0, sentence_features=sentences_f, input_t_context=t_context,
                                   input_c_context=c_context, accuracy=model.accuracy_event,
                                   predicts=model.predicts_event, stype="event")
        # 角色类型预测
        predicts_role = eval_step(input_x=x, input_event_y=y_e, input_role_y=y_r, input_t=t, input_c=c,
                                  input_c_pos=pos_c, input_t_pos=pos_t,
                                  dropout_keep_prob=1.0, sentence_features=sentences_f, input_t_context=t_context,
                                  input_c_context=c_context, accuracy=model.accuracy_role, predicts=model.predicts_role,
                                  stype="role")
        convert_event = {0: "非事件", 1: "股票增长类事件", 2: "股票降低类事件", 3: "股票异动类事件", 4: "股票交易类事件",
                         5: "业绩上升类事件", 6: "业绩下滑类事件", 7: "产品涨价类事件", 8: "产品跌价类事件"
                         }  # 事件类型准换
        convert_role = {0: "其它角色", 1: "施事角色", 2: "受事角色", 3: "时间角色", 4: "地点角色", 5: "数字角色"}  # 角色类型转换
        # 输出测试结果
        for i in range(len(x)):
            print "输入词id:{}".format(x[i])
            print "候选词位置:{}".format(pos_c[i])
            print "触发词位置:{}".format(pos_t[i])
            print "候选词上下文:{}".format(c_context[i])
            print "触发词上下文:{}".format(t_context[i])
            print "输入数据：{}".format(", ".join(map(lambda h: datas.all_words[h], x[i])))
            print "触发词：{}".format(", ".join(map(lambda h: datas.all_words[h], t[i])))
            print "候选词：{}".format(", ".join(map(lambda h: datas.all_words[h], c[i])))
            print "预测事件类别:{}".format(convert_event[predicts_event[i]])
            print "预测角色类别:{}".format(convert_role[predicts_role[i]])
            #print "----------------------------华丽的分割线-----------------------------------------"