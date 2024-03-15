## This is a pytorch implementation of the paper *[Zhen Zhang*, Quming Shen, Yanyu Wang. Electromyographic hand gesture recognition using convolutional neural network with multi-attention. Biomedical Signal Processing and Control, 91, 105935, 2024]*，请随便学习使用，如有发表文章请引用该文章

##ninapro的版权属于原作者，本代码只是把它转成txt格式方便使用，如有发表如何引用请参考ninapro网站


#### Environment
- Pytorch 
- Python 

#### files Structure

```
--dataset--data_process.py : 前处理文件
                 |--multi_attention.py:模型文件
                 |--system_para.py:系统参数
                 |--semg_mam.py: 主文件，请直接执行

--data_x： Ninapro数据文件夹（为了方便大家使用，本代码只是把ninapro db5的肌电数据从mat中提取转成txt文件，如有侵权，请告知）



