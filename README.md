# COMP SCI 7417
Applied Natural Language Processing
2024 semester1 
Assignment 2: Mini-project (report and code)
individual task for level 2- fine tuning bert
report for individual：
系统架构
设计的问答系统架构如下流程图所示：

![appendix](https://github.com/user-attachments/assets/73524c4f-61b5-4f3c-8eea-2002219bc791)


系统的架构主要由输入/输出模块、文本匹配模块、文本预处理模块和NLP模型(BERT)模块构成。对于输入/输出模块，主要由系统提示用户输入问题、参考的文章ID、用户是否退出系统和系统输出回答文本。文本预处理模块的功能为对系统输入的文本进口清晰，包括过滤掉无意义的字符、过滤掉停用词、提取词干等，这是文本数据投喂到NLP模型之前需要的文本清洗工作。文本匹配模块用于匹配与问题最相关的文章或许句子。当用户没有指定参考文章或者参考文章非常长，超出了模型输入的token长度限制时，会处触发该模块。如果用户没有指定参考文章，则文本匹配模块会遍历数据库中所有文档(文章)，计算美篇文章与查询问题的文本相似度，选择相似度最高的文章作为参考文章。文本之间的相似度计算步骤如下：
step 1: 预处理（过滤掉标点符号、过滤停用词、提取词干）文章文本和问题文本
step 2: 使用TF-IDF转换问题文本和文章文本为向量数据
step 3: 计算文档向量d与问题向量q的cosine相似度，计算公式如下：

cosine相似度计算公式参考教科书等4页公式(14.6)

如果用户指定的文章或者文本匹配模块所匹配到的最佳参考文章过长，超出了模型的token限制，那么使用spacy提取出文章中的每个句子，计算每个句子与问题之间的相似度，并选择匹配度最高的n个句子作为新的参考文章。在我们的项目中，n的值被设定为3。
NLP模型模块是QA系统的核心，它接受一个问题和一篇参考文章，并从参考文章中找到答案所谓的位置，并提取出答案文本。NLP模型模块的根据问题和上下文信息生成自然语言的答案。在下章节中，我们将会详细介绍该模块的NLP模型选择和介绍。


4.模型选择和训练：
我们选择BERT(Bidirectional Encoder Representations from Transformers)作为问答系统中的NLP模型。它是由Google在2018年推出的预训练语言模型，它在多项自然语言处理任务上展现了强大的性能。我们首先从Hugging Face加载bert-large-uncased-whole-word-masking-finetuned-squad预训练模型。预训练模型名称中的"Large"意味着该模型具有更多的参数，通常能够捕获更复杂的语言特征，虽然这也会增加计算资源的需求。"Uncased"表示模型在预处理时忽略大小写，这简化了文本处理过程，同时对于大多数问答场景来说，大小写差异并不影响语义理解。”Finetuned on SQuAD”意味着该模型在SQuAD（Stanford Question Answering Dataset）上进行了微调。SQuAD是自然语言处理领域中一个广泛使用的阅读理解数据集，它包含大量的问题-篇章对，要求模型在给定的文本中找到正确答案。经过SQuAD数据集微调的BERT模型特别擅长于从文本中抽取答案。但该模型可能在本项目数据(新闻数据)中的性能并不好，因此我们还会根据新闻数据中的问题-篇章-答案数据进行微调。为了验证我们微调过的BERT更适合新闻数据的问答系统，我们将bert-large-uncased-whole-word-masking-finetuned-squad作为baseline模型。
我们从news数据中随机抽取了41篇文章，对于每一篇文章设计了问题，然后根据文章内容提取出合适的答案。这样我们可以得到41对问题-文章-答案数据，这些数据作为训练集，用于微调BERT。模型以Cross-Entropy Loss作为损失函数。我们选择AdamW作为优化器，学习率为0.00001,训练迭代20次后，损失函数以及收敛，停止训练后作为最终预测模型。我们使用F1-score和exact match(EM)作为模型评估指标。这是因为我们大部分个案的答案都非常简单，只有几个单词。An example of question for article number 17574 would be: “Who is the vice chairman of Samsung”, answer: “Jay Y. Lee”.
对于微调过的BERT，我们使用SQuAD 2.0作为测试集，对比模型在不同数据集之间的性能。模型的训练包括如下步骤：
从文章中找到找到答案所在的起始索引和结束索引
Tokenize passages and queries。我们select the BERT-base pretrained model “bert-base-uncased” for the tokenization.
Convert the start-end positions to tokens start-end positions.
加载预训练模型：bert-large-uncased-whole-word-masking-finetuned-squad
参数设置与优化器选择：学习率为0.00001，训练迭代次数为20，优化器为AdamW。
在训练集上进行模型微调。



5.用户与系统的交互：
进入系统后，系统会提示用户输入问题，问题应该是自然语言中的一个简单句子，指的是用户给出的文章中的一句话中的一个事实。比如，问题的示例是“谁是三星副董事长”。然后系统会继续提示用户提供参考文章的ID，如果用户不指定文章ID，那么系统提示用户输入“no”，文档匹配模块被激活，搜索与问题最相关的文章作为参考文章。系统输出答案文本，答案为参考文章的片段。特别地，有时候模型从参考文章中搜索不到答案，则会说出”There is no answer”。 当完成一个问询-回答后，系统循环提示用户询问下一个问题，直到用户输入“退出”命令。
