import torch
import torch.nn as nn
import torch.optim as optim
# 可以结合MOMO笔记来看，对前向的理解不够深，可以改造的
torch.manual_seed(1)

# some helper functions
def argmax(vec):
    # return the argmax as a python int
    # 第1维度上最大值的下标
    # input: tensor([[2,3,4]])
    # output: 2
    _, idx = torch.max(vec,1)
    return idx.item()

def prepare_sequence(seq,to_ix):
    # 文本序列转化为index的序列形式
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def log_sum_exp(vec):
    #compute log sum exp in a numerically stable way for the forward algorithm
    # 用数值稳定的方法计算正演算法的对数和exp
    # input: tensor([[2,3,4]])
    # max_score_broadcast: tensor([[4,4,4]])
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1,-1).expand(1,vec.size()[1])
    return max_score+torch.log(torch.sum(torch.exp(vec-max_score_broadcast)))

START_TAG = "<s>"
END_TAG = "<e>"

# create model
class BiLSTM_CRF(nn.Module):
    def __init__(self,vocab_size, tag2ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF,self).__init__()
        self.embedding_dim = embedding_dim  # 词向量维度
        self.hidden_dim = hidden_dim  # 记忆维度
        self.tag2ix = tag2ix
        self.tagset_size = len(tag2ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)  # 词向量矩阵
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, num_layers=1, bidirectional=True)

        # maps output of lstm to tog space
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # matrix of transition parameters
        # entry i, j is the score of transitioning to i from j
        # tag间的转移矩阵，是CRF层的参数
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # these two statements enforce the constraint that we never transfer to the start tag
        # and we never transfer from the stop tag
        # 不会转移到起点，终点也不会再次转移？？？
        self.transitions.data[tag2ix[START_TAG], :] = -10000  # 表示转移到此状态
        self.transitions.data[:, tag2ix[END_TAG]] = -10000

        self.hidden = self.init_hidden()  # hidden 初始化

    def init_hidden(self):
        return (torch.randn(2, 1,self.hidden_dim//2),
                torch.randn(2, 1,self.hidden_dim//2))

    def _forward_alg(self, feats):  # 总分，归一化项，分母
        # to compute partition function
        # 求归一化项的值，应用动态归化算法
        init_alphas = torch.full((1,self.tagset_size), -10000.)# tensor([[-10000.,-10000.,-10000.,-10000.,-10000.]])
        # START_TAG has all of the score
        init_alphas[0][self.tag2ix[START_TAG]] = 0#tensor([[-10000.,-10000.,-10000.,0,-10000.]])

        forward_var = init_alphas

        for feat in feats:  # 每一帧数据，实际对应的是5个发射分数
            #feat指Bi-LSTM模型每一步的输出(抽取的特征)，大小为tagset_size
            alphas_t = []
            # 针对每一维计算发射、转移分数
            for next_tag in range(self.tagset_size):
                # 如tensor([3]) -> tensor([[3,3,3,3,3]])
                # 1*5 发射分数
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # 1*5 转移概率
                trans_score = self.transitions[next_tag].view(1, -1)  # 转移到此状态
                # 上一步的路径和+转移分数+发射分数  # 对应Sij那个公式
                next_tag_var = forward_var + trans_score + emit_score
                # log_sum_exp求和
                alphas_t.append(log_sum_exp(next_tag_var).view(1))  # 对应previous中的第i项

            # 此时alpha_t有5个分数，分别表示转移到某状态的得分
            forward_var = torch.cat(alphas_t).view(1, -1)  # 将列表转化为tensor
        terminal_var = forward_var+self.transitions[self.tag2ix[END_TAG]]
        alpha = log_sum_exp(terminal_var)
        #归一项的值
        return alpha

    def _get_lstm_features(self,sentence):
        # 特征抽取
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence),1,-1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self,feats,tags):
        # gives the score of a provides tag sequence
        # 求某一路径的值
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag2ix[START_TAG]], dtype=torch.long), tags])  # 加上起始标记
        for i , feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag2ix[END_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        # 当参数确定的时候，求解最佳路径
        backpointers = []

        init_vars = torch.full((1,self.tagset_size),-10000.)# tensor([[-10000.,-10000.,-10000.,-10000.,-10000.]])
        init_vars[0][self.tag2ix[START_TAG]] = 0#tensor([[-10000.,-10000.,-10000.,0,-10000.]])

        forward_var = init_vars
        for feat in feats:
            bptrs_t = [] # holds the back pointers for this step
            viterbivars_t = [] # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag2ix[END_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag2ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        # 由lstm层计算得的每一时刻属于某一tag的值
        # 例如句子长度为11，标签类数为5，则返回11*5的矩阵，表示11步，每步对用的发射分数
        feats = self._get_lstm_features(sentence)  # 句子长度，标签类数
        # 归一项的值
        forward_score = self._forward_alg(feats)
        # 正确路径的值
        gold_score = self._score_sentence(feats, tags)
        # 正确路径的分值 - 归一项的值，希望两者最后的分差为0
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


#　主函数
if __name__ == "__main__":
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 4
    # Make up some training data
    # 11 + 7
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]
    word2ix = {}  # 单词:id
    for sentence, tags in training_data:  #　句子　标记
        for word in sentence:
            if word not in word2ix:
                # 妙啊，逐个的添加到字典中，index会渐渐加一
                word2ix[word] = len(word2ix)

    # tag:id
    tag2ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, END_TAG: 4}
    id2tag = {ix:tag for (tag, ix) in tag2ix.items()}
    model = BiLSTM_CRF(len(word2ix), tag2ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    # Check predictions before training
    # 输出训练前的预测序列
    print("label", training_data[0][1])
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word2ix)
        precheck_tags = torch.tensor([tag2ix[t] for t in training_data[0][1]], dtype=torch.long)
        score, tags = model(precheck_sent)
        tags = [id2tag[ix] for ix in tags]
        print(score, tags)
    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            # 构建单词序列、标记序列
            sentence_in = prepare_sequence(sentence, word2ix)
            targets = torch.tensor([tag2ix[t] for t in tags], dtype=torch.long)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()
    # Check predictions after training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word2ix)
        score, tags = model(precheck_sent)
        tags = [id2tag[ix] for ix in tags]
        print(score, tags)

# 输出结果
# label           ['B', 'I', 'I', 'I', 'O', 'O', 'O', 'B', 'I', 'O', 'O']
# tensor(2.6907)  ['I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I']
# tensor(20.4906) ['B', 'I', 'I', 'I', 'O', 'O', 'O', 'B', 'I', 'O', 'O']

