from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import itertools

# data_path = "D:/match/moj_sim/origindata/"
data_path = "origindata/"
TRAIN_PATH = data_path + 'train.csv'
TEST_PATH = data_path + 'test.csv'
QUESTION_PATH = data_path + 'question.csv'
CHAR_PATH = data_path + 'char_embed.txt'
WORD_PATH = data_path + 'word_embed.txt'


def get_ids(qids):
    ids = []
    for t_ in qids:
        ids.append(int(t_[1:]))
    return np.asarray(ids)


def get_texts(file_path, question_path):
    qes = pd.read_csv(question_path)
    file = pd.read_csv(file_path)
    q1id, q2id = file['q1'], file['q2']
    id1s, id2s = get_ids(q1id), get_ids(q2id)
    all_words = qes['words']
    texts = []
    for t_ in zip(id1s, id2s):
        texts.append(all_words[t_[0]] + ' ' + all_words[t_[1]])
    return texts


def get_embed(path):
    data = pd.read_table(path, header=None)
    print(data.head())

    return


def make_submission(predict_prob):
    with open(data_path + 'submission.csv', 'w') as file:
        file.write(str('y_pre') + '\n')
        for line in predict_prob:
            file.write(str(line) + '\n')
    file.close()


print('Load files...')
questions = pd.read_csv(QUESTION_PATH)
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
corpus = questions['words']

char = pd.read_table(CHAR_PATH, header=None, sep=' ', index_col=0)
word = pd.read_table(WORD_PATH, header=None, sep=' ', index_col=0)


def analy_sen_lenth():
    # 分析 questions
    # 句子词数
    questions['wtokens'] = questions.words.apply(lambda x: len(x.split(' ')))
    questions.wtokens.value_counts()
    sum(questions.wtokens <= 10)  # 95% 词数小于10
    questions['ctokens'] = questions.chars.apply(lambda x: len(x.split(' ')))
    questions.ctokens.value_counts()


# 相似扩展
train0 = train.ix[train.label == 0]
train1 = train.ix[train.label == 1]

train_revert = train.copy(deep=True)
train_revert.columns = ['label', 'q2', 'q1']

train_symmetry = train.append(train_revert, ignore_index=True)
ts0 = train_symmetry.ix[train_symmetry.label == 0]
ts1 = train_symmetry.ix[train_symmetry.label == 1]
ts1.q1.value_counts().describe([0.01, 0.1, 0.2, 0.5, 0.9, 0.99])
sum(ts1.q1.value_counts() < 2)

enlarge_simi = pd.DataFrame()

gp_q1 = ts1.groupby("q1")
for name, gp in gp_q1:
    if len(gp) > 1:
        iters = itertools.combinations(gp.q2.tolist(), 2)
        for q1, q2 in iters:
            enlarge_simi = enlarge_simi.append(pd.DataFrame([{"label": 1, "q1": q1, "q2": q2}]), ignore_index=True)
        # print(enlarge_simi)
        # break

print(len(ts1))
print(len(enlarge_simi))
print(enlarge_simi.duplicated().sum())

enlarged_simi = ts1.append(enlarge_simi, ignore_index=True)
print(enlarged_simi.duplicated().sum())

