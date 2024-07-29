- [论文概述](#论文概述)
  - [什么是多标签学习](#什么是多标签学习)
  - [论文贡献](#论文贡献)
- [算法流程\`\`\`](#算法流程)
- [导入必要的库](#导入必要的库)
- [定义函数](#定义函数)
- [主程序](#主程序)
- [定义LIFTClassifier类，继承自BaseEstimator和ClassifierMixin](#定义liftclassifier类继承自baseestimator和classifiermixin)
- [定义MLkNN类](#定义mlknn类)
- [定义RankSVM类，继承自BaseEstimator和ClassifierMixin](#定义ranksvm类继承自baseestimator和classifiermixin)
- [定义MultiLabelDecisionTree类](#定义multilabeldecisiontree类)
- [定义MLP神经网络类，继承自nn.Module](#定义mlp神经网络类继承自nnmodule)
- [定义BPMLL类，继承自BaseEstimator和ClassifierMixin](#定义bpmll类继承自baseestimator和classifiermixin)
- [定义RandomKLabelsetsClassifier类，继承自BaseEstimator和ClassifierMixin](#定义randomklabelsetsclassifier类继承自baseestimator和classifiermixin)
- [未来可能的改进方向](#未来可能的改进方向)
- [环境配置](#环境配置)

# 论文概述
![Description](https://cdn-store.aspiringcode.com/1720349070671_截屏20240707184418.png)
帕金森病是一种使人虚弱的慢性神经系统疾病。传统中医（TCM）是一种诊断帕金森病的新方法，而用于诊断帕金森病的中医数据集是一个多标签数据集。考虑到帕金森病数据集中的症状（标签）之间总是存在相关性，可以通过利用标签相关性来促进多标签学习过程。目前的多标签分类方法主要尝试从标签对或标签链中挖掘相关性。该文章提出了一种简单且高效的多标签分类框架，称为潜在狄利克雷分布多标签（LDAML），该框架旨在通过使用类别标签的主题模型来学习全局相关性。简而言之，研究人员试图通过主题模型在标签集上获得抽象的“主题”，从而能够挖掘标签之间的全局相关性。大量实验清楚地验证了所提出的方法是一个通用且有效的框架，能够提高大多数多标签算法的性能。基于该框架，研究人员在中医帕金森病数据集上取得了令人满意的实验结果，这可以为该领域的发展提供参考和帮助。

## 什么是多标签学习
多标签学习（Multi-Label Learning）是一种机器学习方法，用于处理具有多个标签的数据样本。与传统的单标签学习不同，每个数据点在多标签学习中可以同时属于一个或多个类别，而不仅仅是一个确定的标签。其目标是经过算法训练后输出一个分类模型，即学习一组从特征空间到标记空间的实值函数映射。假设使用$X=R^d$表示一个d维的输入空间，$Y=\{y_1,y_2,y_3,...,y_q\}$表示可能输出的q个类别，多标签任务即在训练集合$D=\{(x_1, Y_1),(x_2, Y_2),...,(x_m, Y_m)\}$上学习一个X到Y的函数，该函数可以衡量x和y的相关性，对于未见过的实例x预测其对应的标签y。
## 论文贡献
1. 提出了一种通用且高效的多标签分类框架——Latent Dirichlet Allocation Multi-Label (LDAML)。该框架通过利用标签间的关联性进行多标签分类。
2. 该框架可以应用于大多数当前的多标签分类方法，使其性能得到提升。通过使用LDAML框架，可以显著提升简单方法（如Binary Relevance, BR）的性能，甚至超过某些最新的方法，同时保持较低的时间成本。
3. 提出的改进LDAML在某些特殊数据集（如帕金森数据集）上取得了最佳性能。特别是在帕金森数据集上，改进的LDAML框架实现了最优性能，达到了本文的最终目标。该方法能够在未来为医生提供指导和帮助。
# 算法流程
## 挖掘“主题“——提取标签相关性
与通过查找标签子集或标签链来利用相关性的传统方法不同，LDAML通过发现标签的抽象“主题”来利用相关性。假设为d维特征向量的输入空间，表示q类标号的输出空间。给定多标签训练集，其中为d维特征向量，为对应的标签集。我们可以将每个实例看作一个文档，每个标签看作文档中的一个单词。直观地说，一定有一些抽象的“主题”，期望特定的标签或多或少地出现在实例中，特别是在包含大量相关标签的多标签数据集中。LDAML算法的主要流程分为两步：（1）从训练集中挖掘标签主题；（2）计算主题的离散分布。

**从训练集中挖掘标签主题:** 首先，我们将LDA引入到训练集d中，每个实例xi表示文档，每个标签表示第i个实例中的第j个标签。然后利用LDA模型生成过程计算实例-主题 θ 的概率分布矩阵，其中 表示第i个实例注入第j主题的概率。
**主题的离散分布:** 计算实例-主题分布矩阵后，得到每个实例属于每个主题的概率值。为了确定实例确切属于哪个主题，我们需要用离散值0/1来代替概率值。在这里我们使用的离散化方法如下所示：
![Description](https://cdn-store.aspiringcode.com/1720347277228_截屏20240707181415.png)

## 训练$M_T$模型——拟合{特征集, 主题集合}
在这里我们的训练集数据与测试集数据分布相似，因此我们可以假设测试数据集的主题概率分布与训练数据集相同。首先我们对训练集提取出具有标记相关性的k个主题(利用算法1)，然后我们使用多标签分类模型$M_T$对训练集的特征-主题进行拟合，然后利用训练好的MT模型对未知标记集合的测试集特征数据生成含有标记相关性的k个主题（这里需要注意的是，$M_T$可以随便选取一个有效的多标签分类模型，文章的重点是利用标签相关性来提高各种多标签学习模型的效率）。

## 用标记相关性扩增数据集
我们将这k个主题加入训练集，从而构建出新的训练集——{训练特征集，训练集标签主题}。我们将这k个主题加入数据集，从而构建出新的训练集——$D' = \{(x_i', Y_i) \mid 1 \leq i \leq N, x_i' = x_i \oplus Y_T \}$
新的测试集——$t' = \{(\hat{x_i'}, \hat{Y_i}) \mid 1 \leq i \leq N, \hat{x_i'} = x_i \oplus \hat{Y_T} \}$。

## 再次训练拟合$M$模型——对真实帕金森病例进行筛查
最后，可以再次使用一种多标签学习模型M对扩增后的训练集D'进行拟合，进一步建立输入数据和输出空间的数据联系。然后对扩增后的测试集t'进行多标签分类，获得输入样本是否患有病症以及其他情况的预测结果。上述过程的整体框架流程图如算法2所示。
![Description](https://cdn-store.aspiringcode.com/1720348725009_截屏20240707183834.png)

# 实验结果
文章在四份数据集上用多种多标签学习分类模型分别加上LDAML算法与其原始模型的分类效果进行对比，实验结果如图所示：
![Description](https://cdn-store.aspiringcode.com/1720409320830_截屏20240708112658.png)
![Description](https://cdn-store.aspiringcode.com/1720409340844_截屏20240708112716.png)
![Description](https://cdn-store.aspiringcode.com/1720409347516_截屏20240708112724.png)
以上实验结果表明，LDAML能够在性能和时间成本之间取得良好的平衡。目前的大多数方法都可以应用于LDAML。我们可以采用目前最先进的方法作为LDAML在原始基础上取得突破的基本方法（base model）。另一方面，唯一额外的时间代价是计算主题概率分布矩阵的小词空间。因此，LDAML的时间成本接近于其基础方法的时间成本。通过采用BR或CC等较弱的方法作为基本方法，可以在较低的时间成本下提高接近实际状态的性能。这些结果表明，LDAML是一个通用的框架，可以为具有标签相关性的多标签问题提供鲁棒且更优的解决方案。
# 核心代码复现
由于改论文代码目前尚未开源，因此在本文中我将给出由本人根据论文算法流程一比一复制的复现代码，代码源文件我将放在附件中，其核心逻辑如下：
## main.py文件
``` 
#########################伪代码###########################
# 导入必要的库
Import libraries

# 定义函数
Function discretize(theta):
    # 初始化二进制矩阵 YT
    Initialize YT as a zero matrix with the same shape as theta
    For each row i in theta:
        Find the maximum value in row i
        For each column j in row i:
            If the difference between the max value and theta[i][j] is less than 1/K:
                Set YT[i][j] to 1
            Else:
                Set YT[i][j] to 0
    Return YT

Function convert_to_one_hot(data):
    # 获取唯一值和类别数
    Find unique values in data
    Initialize one_hot_encoded as a zero matrix
    For each value in data:
        Find the index of the value in unique values
        Set the corresponding position in one_hot_encoded to 1
    Return one_hot_encoded

Function lda(labels, n):
    # 进行潜在狄利克雷分配（LDA）
    Initialize LDA model with n components
    Fit and transform labels using LDA model
    Discretize the transformed data
    Return the discretized data

Function metric_cal(test, pred):
    # 计算并打印评估指标
    Calculate accuracy, precision, recall, F1 score, and AUC
    Print the calculated metrics

# 主程序
If __name__ == "__main__":
    # 加载数据
    Load data from Excel file
    # 定义标签列和特征
    Define label_cols and features
    Convert features and labels to NumPy arrays
    # 设置主题数
    Set n to 6
    # 对标签进行LDA
    Call lda function to get Y_T
    # 将特征与离散化的标签组合
    Concatenate features and Y_T to get XYT
    # 划分训练集和测试集
    Split XYT and labels into X_train, X_test, y_train, y_test
    # 初始化多标签分类器
    Initialize MT_classifier as RankSVM
    # 从训练集和测试集中提取主题
    Extract yt_train and yt_test from X_train and X_test
    Remove last n columns from X_train and X_test
    # 训练多标签分类器
    Fit MT_classifier using X_train and yt_train
    # 预测测试集的主题
    Predict yt_proba and yt_pred using MT_classifier on X_test
    Convert yt_pred to integer
    # 使用预测的主题扩展训练集和测试集
    Concatenate X_train with yt_train to get X_train_aug
    Concatenate X_test with yt_pred to get X_test_aug
    # 初始化并训练二进制相关性分类器
    Initialize base_classifier as MLPClassifier
    Initialize clf as BinaryRelevance with base_classifier
    Fit clf using X_train_aug and y_train
    # 预测测试集的标签
    Predict y_pred and y_score using clf on X_test_aug
    # 计算评估指标
    Calculate hamming loss, ranking loss, coverage error, and average precision
    Print calculated metrics
    # 对每个标签计算并打印评估指标
    For each label i:
        Extract test and pred for label i
        Call metric_cal function to calculate and print metrics
        Print separator
    Print final separator
```
在主文件main.py中我复现了LDAML算法的整个流程，并实现了从输入数据到输出评价指标的全过程，在这里默认采用的多标签学习分类起$M_T$和$M$是RankSVM和二元回归+深度学习。
## multi_label_learn.py文件
```
# 定义LIFTClassifier类，继承自BaseEstimator和ClassifierMixin
class LIFTClassifier(BaseEstimator, ClassifierMixin):
    # 初始化函数，接受一个基本分类器作为参数
    def __init__(self, base_classifier=DecisionTreeClassifier()):
        设置base_classifier为传入的参数
        初始化classifiers字典

    # 训练模型函数
    def fit(self, X, y):
        获取标签数量
        遍历每个标签
            对每个标签训练一个分类器
            将训练好的分类器存入classifiers字典
        返回self

    # 预测函数
    def predict(self, X):
        获取标签数量
        初始化预测结果矩阵
        遍历每个标签
            使用对应的分类器进行预测
            将预测结果存入预测结果矩阵
        返回预测结果矩阵

    # 预测概率函数
    def predict_proba(self, X):
        获取标签数量
        初始化概率预测结果矩阵
        遍历每个标签
            使用对应的分类器进行概率预测
            将预测概率结果存入概率预测结果矩阵
        返回概率预测结果矩阵

# 定义MLkNN类
class MLkNN:
    # 初始化函数，接受一个k值作为参数
    def __init__(self, k=3):
        设置k值
        初始化k近邻模型

    # 训练模型函数
    def fit(self, X, y):
        保存训练数据X和y
        使用X训练k近邻模型

    # 预测函数
    def predict(self, X):
        获取样本数量
        初始化预测结果矩阵

        遍历每个样本
            获取样本的k+1个最近邻
            排除样本自身
            计算邻居标签的和
            根据标签和判断最终预测结果
        返回预测结果矩阵

    # 预测概率函数
    def predict_proba(self, X):
        获取样本数量
        初始化概率预测结果矩阵

        遍历每个样本
            获取样本的k+1个最近邻
            排除样本自身
            计算每个标签的概率
        返回概率预测结果矩阵

# 定义RankSVM类，继承自BaseEstimator和ClassifierMixin
class RankSVM(BaseEstimator, ClassifierMixin):
    # 初始化函数，接受参数C, kernel, gamma
    def __init__(self, C=1.0, kernel='rbf', gamma='scale'):
        设置C, kernel, gamma值
        初始化模型列表
        初始化多标签二值化器

    # 训练模型函数
    def fit(self, X, y):
        使用多标签二值化器转换y
        获取标签数量

        遍历每个标签
            将当前标签转换为二值格式
            使用SVM训练二值化后的标签
            将训练好的SVM模型加入模型列表

    # 预测函数
    def predict(self, X):
        初始化预测结果矩阵

        遍历每个SVM模型
            使用模型进行预测
            将预测结果存入预测结果矩阵
        返回预测结果矩阵

    # 预测概率函数
    def predict_proba(self, X):
        初始化概率预测结果矩阵

        遍历每个SVM模型
            使用模型进行概率预测
            将预测概率结果存入概率预测结果矩阵
        返回概率预测结果矩阵

# 定义MultiLabelDecisionTree类
class MultiLabelDecisionTree:
    # 初始化函数，接受参数max_depth, random_state
    def __init__(self, max_depth=None, random_state=None):
        设置max_depth, random_state值
        初始化标签幂集转换器
        初始化决策树分类器

    # 训练模型函数
    def fit(self, X, y):
        使用标签幂集转换器转换y
        使用转换后的y训练决策树分类器

    # 预测概率函数
    def predict_proba(self, X):
        使用决策树分类器进行概率预测
        将预测概率结果转换为原始标签格式
        返回概率预测结果

    # 预测函数
    def predict(self, X):
        使用决策树分类器进行预测
        将预测结果转换为原始标签格式
        返回预测结果

# 定义MLP神经网络类，继承自nn.Module
class MLP(nn.Module):
    # 初始化函数，接受输入大小、隐藏层大小和输出大小作为参数
    def __init__(self, input_size, hidden_size, output_size):
        调用父类的初始化函数
        初始化全连接层1
        初始化ReLU激活函数
        初始化全连接层2
        初始化Sigmoid激活函数

    # 前向传播函数
    def forward(self, x):
        通过全连接层1
        通过ReLU激活函数
        通过全连接层2
        通过Sigmoid激活函数
        返回输出

# 定义BPMLL类，继承自BaseEstimator和ClassifierMixin
class BPMLL(BaseEstimator, ClassifierMixin):
    # 初始化函数，接受参数input_size, hidden_size, output_size, epochs, lr
    def __init__(self, input_size, hidden_size, output_size, epochs=10, lr=0.0001):
        设置输入大小、隐藏层大小、输出大小、训练轮数、学习率
        初始化MLP模型
        初始化优化器
        初始化损失函数

    # 训练模型函数
    def fit(self, X_train, X_val, y_train, y_val):
        将训练数据和验证数据转换为张量
        创建训练数据集和数据加载器

        遍历每个训练轮次
            设置模型为训练模式
            遍历训练数据加载器
                清零梯度
                前向传播
                计算损失
                反向传播
                更新参数

            设置模型为评估模式
            计算验证损失并打印

    # 预测概率函数
    def predict_proba(self, X):
        设置模型为评估模式
        禁用梯度计算
        进行前向传播
        返回预测概率结果

    # 预测函数
    def predict(self, X, threshold=0.5):
        获取预测概率结果
        根据阈值判断最终预测结果
        返回预测结果

# 定义RandomKLabelsetsClassifier类，继承自BaseEstimator和ClassifierMixin
class RandomKLabelsetsClassifier(BaseEstimator, ClassifierMixin):
    # 初始化函数，接受参数base_classifier, labelset_size, model_count
    def __init__(self, base_classifier=None, labelset_size=3, model_count=10):
        设置基本分类器、标签集大小、模型数量
        初始化RakelD模型

    # 训练模型函数
    def fit(self, X, y):
        使用RakelD模型训练数据
        返回self

    # 预测函数
    def predict(self, X):
        使用RakelD模型进行预测
        返回预测结果

    # 预测概率函数
    def predict_proba(self, X):
        使用RakelD模型进行概率预测
        返回概率预测结果

```
同时我在文件multi_label_learning.py中定义了多种不同的多标签学习分类模型，大家可以自行调用相应的函数来进行实验以验证LDAML算法的有效性，使用方法我会在本文对应的视频中进行讲解。
# 使用方法
## 导入本地数据集
调用LDAML算法的方法放在main.py文件中，首先我们需要将文件路径修改成自己所要使用的数据集路径。这里我使用的文件路径为'./测试数据.xlsx'，供大家一键运行熟悉项目。然后大家需要将自己的标签列名称提取变量label_cols中，用于对数据集划分特征集合与标签集合。
![Description](https://cdn-store.aspiringcode.com/1720521567122_截屏20240709183846.png)
## 构建多标签学习分类模型
构建想要的多标签学习分类算法，这里我给大家复现了多种经典的多标签分类器，如LIFT、MlkNN和RankSVM等，并帮大家配置好了参数，大家可以将想要使用的算法对应行的注释删掉即可（$M_T$和$M$都是一样）。
![Description](https://cdn-store.aspiringcode.com/1720521750979_截屏20240709184026.png)
## 运行模型输出测试指标
设置好这些外在参数后，我们就可以运行代码，主文件将自动调用第三方库和multi_label_learn.py文件中的函数来进行训练和测试。下面是我选取的几种测试指标，分别会输出模型对整体的多标签分类性能指标（Hamming loss、Ranking loss、Coverage error和Average precision）和对单一标签的分类指标（Accuracy、Precision、Recall、F1 Score和AUC）。
![Description](https://cdn-store.aspiringcode.com/1720521976103_截屏20240709184249.png)

# 测试结果
下面是在测试数据集上模型的表现：
![Description](https://cdn-store.aspiringcode.com/1720522173842_截屏20240709184755.png)
以上是模型多标签学习分类的性能，Hamming Loss为0.051228070175438595，Ranking Loss为0.016737120579225842，Coverage Error为2.3263157894736843，Average Precision为0.7500066243540565
![Description](https://cdn-store.aspiringcode.com/1720522505735_截屏20240709185439.png)
以上是对模型在单一标签下的分类性能测试结果，测试数据集中有十个标签，因此这里会输出十个标签下模型分类的Accuracy、Precision、Recall、F1 Score和AUC，也就是说这样的数据会有十组
![Description](https://cdn-store.aspiringcode.com/1720523066940_截屏20240709190359.png)
我这里把数据列成表这样大家可以更直观的看到，我换用了不同的多标签学习算法结合LDAML，并比较了它们在Accuracy、AUC和F1-score上的表现。在上面的情况上来看，使用BPMLL在整体对单一标签进行分类时效果相比其他算法更好，但也会在某些标签下弱于其他模型。```

```

# 未来可能的改进方向
这一部分是笔者通过思考感觉可以在目前LDAML的基础上进行改进的方面，也就是我想给大家介绍的LSA算法。

潜在语义分析（Latent Semantic Analysis，LSA）是一种用于分析大规模文本数据的统计方法，旨在发现文本中的潜在语义结构并提取其语义信息。LSA假设文本中存在一些潜在的语义结构，即使在词语表达方式不同的情况下，这些结构也会保持一定的稳定性。其基本思想是将文本数据表示为一个矩阵，其中行代表文档，列代表词语，而矩阵中的元素则可以是词频、TF-IDF权重等。接下来，通过奇异值分解（Singular Value Decomposition，SVD）将这个矩阵分解为三个矩阵的乘积： 其中，A是原始文本矩阵，U是文档-概念矩阵，Σ是奇异值矩阵，是词语-概念矩阵的转置。LSA通过保留最重要的奇异值及其对应的左右奇异向量，将文本数据的维度降低到一个更小的空间，从而发现潜在的语义结构，并提取出文本数据的语义信息。

LSA在面对大规模文本数据时，能够有效地提取出其中的潜在语义信息。并且，LSA能发现文本数据中的主题结构并提取出其中的主题信息。受此启发，我们使用LSA对膝骨关节炎标记集合中的十个标记进行相关性计算并提取主题，从而获得标记集合中的高阶信息。相比之下，LSA比LDA更加灵活和简单。LDA对于大规模数据的处理速度较慢，因为它需要对每个词项和主题进行迭代推断，对主题分布和词项分布的先验参数进行设定，而LSA只需进行奇异值分解，不需要对先验参数进行设置，因此更容易实现和调试。LSA在语义上也更为易懂。LDA通过抽样方法从文档中抽取主题，它的主题在语义上可能难以解释，LSA通过奇异值分解从标签数据中提取主题，可以更直观地解释这些主题的含义，更好地反映标签之间的语义关系。

接下来是不是有可能将LSA融入到目前的框架中，或者直接基于LSA开发一种标记相关性提取的算法都是可以尝试的方向，可以留给大家一起去学习探索！
# 环境配置
- python3.8或以上版本
- 须事先安装第三方库torch、numpy、sklearn、pandas、skmultilearn
- 可修改变量——主题数n、所用的本地数据集、多标签分类器$M_T$和$M$
![Description](https://cdn-store.aspiringcode.com/1720409752103_截屏20240708113520.png)
![Description](https://cdn-store.aspiringcode.com/1720409758953_截屏20240708113529.png)
![Description](https://cdn-store.aspiringcode.com/1720409765248_截屏20240708113538.png)
