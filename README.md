# ECO-paddlepaddle
# **将数据集进行解压**

# In[ ]:


get_ipython().system('unzip -q /home/aistudio/data/data48916/UCF-101.zip -d data')


# **抽帧**

# In[ ]:


get_ipython().system('python /home/aistudio/work/jpg_convert.py')


# **把数据集按照训练集：验证集：测试集=8：1：1进行切分并配置好数据迭代器**




get_ipython().system('python  /home/aistudio/work/jpg_to_pkl.py')




get_ipython().system('python  /home/aistudio/work/data_gener.py')


# **训练：因为原论文是采用预训练模型，而我是重头开始训练，所以采取分步训练，先设学习率为0.001训练30个批次，然后再训练30个批次，感觉差不多了，进行微调，学习率设为0.0001，进行10个批次微调，发现准确率差了点，然后再进行5个批次的微调。**



get_ipython().system('python /home/aistudio/work/train.py --use_gpu  True --epoch 60 --pretrain True')


# **对模型进行评价，由于对视频进行32等份均分，并进行随机抽取，所以测试结果每次都会有一定波动。在自己项目中测试的准确率为93.6，项目地址为：**
# https://aistudio.baidu.com/aistudio/projectdetail/705066




get_ipython().system("python /home/aistudio/work/eval.py --weights '/home/aistudio/work/checkpoints_models/ECO_model4' --use_gpu True ")
