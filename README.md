# BITNLP2022
## 1. 介绍
我们实现了[机器翻译领域](https://www.datafountain.cn/competitions/543)的比赛，任务描述如下：
本次本次任务旨在面向低资源神经机器翻译的领域适应方法研究，提供口语领域的中英平行句对、专利领域的中英平行句对和英文单语数据以及医药领域英文单语数据作为训练样本，参赛队伍需要基于提供的训练样本进行中到英机器翻译模型的构建与训练，并基于口语、专利、医药三个领域测试集分别提供翻译结果。
## 2. 运行环境安装
### 2.1 准备虚拟环境 ###
1. 准备conda环境并进行激活.
	```shell
	conda create -n NLP python=3.7
	conda active NLP
	```
2. 在[官网](https://pytorch.org/)安装对应版本的pytorch
![image](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ec360791671f4a4ab322eb4e71cc9e62~tplv-k3u1fbpfcp-zoom-1.image)
可以在终端直接使用官方提供的code
   **Note:  安装前请确保电脑上是否有显卡且显卡算力和pytorch版本匹配**
3.  安装FedKNOW所需要的包
	```shell
	git clone https://github.com/LINC-BIT/FedKNOW.git
	pip install -r requirements.txt
	```
### 2.2 运行代码
1. 将数据样本转换为csv文件
	```shell
	python data/integrate_all_data.py
	```
2. 单语样本转换
	```shell
	python run_translation_no_trainer_myown_zh2en.py
	```
3. 生成预测样本
	```shell
	python run_translation_no_trainer_myown_en2zh.py
	```
你可以使用submit/submit_example/txt直接进行提交
## 3. 结果
我们在比赛中达到了第3名，得分为35.377