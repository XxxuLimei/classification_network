## 0407：  
ResNet网络结构：  
![](https://github.com/XxxuLimei/classification_network/blob/main/ResNet/1_6hF97Upuqg_LdsqWY6n_wg.png)  
ResNet网络参数：  
![](https://github.com/XxxuLimei/classification_network/blob/main/ResNet/resnet.png)  
- 训练日志：  
以下是加预训练的情况。  
```
(base) xilm@xilm-MS-7D17:~/python_file/img_classification$ /home/xilm/anaconda3/bin/python /home/xilm/python_file/img_classification/train.py
cuda:0
 train loss: 100%[**************************************************->]1.343
[epoch 1] train loss: 0.503 test_accuracy: 0.912
 train loss: 100%[**************************************************->]0.348
[epoch 2] train loss: 0.339 test_accuracy: 0.926
 train loss: 100%[**************************************************->]0.218
[epoch 3] train loss: 0.296 test_accuracy: 0.956
 train loss: 100%[**************************************************->]0.512
[epoch 4] train loss: 0.242 test_accuracy: 0.964
 train loss: 100%[**************************************************->]0.422
[epoch 5] train loss: 0.245 test_accuracy: 0.956
 train loss: 100%[**************************************************->]0.860
[epoch 6] train loss: 0.218 test_accuracy: 0.956
 train loss: 100%[**************************************************->]0.101
[epoch 7] train loss: 0.214 test_accuracy: 0.929
 train loss: 100%[**************************************************->]0.061
[epoch 8] train loss: 0.217 test_accuracy: 0.926
 train loss: 100%[**************************************************->]0.833
[epoch 9] train loss: 0.193 test_accuracy: 0.956
 train loss: 100%[**************************************************->]0.348
[epoch 10] train loss: 0.209 test_accuracy: 0.962
 train loss: 100%[**************************************************->]0.075
[epoch 11] train loss: 0.177 test_accuracy: 0.937
 train loss: 100%[**************************************************->]0.180
[epoch 12] train loss: 0.169 test_accuracy: 0.951
 train loss: 100%[**************************************************->]0.496
[epoch 13] train loss: 0.176 test_accuracy: 0.945
 train loss: 100%[**************************************************->]0.199
[epoch 14] train loss: 0.159 test_accuracy: 0.951
 train loss: 100%[**************************************************->]0.043
[epoch 15] train loss: 0.188 test_accuracy: 0.959
 train loss: 100%[**************************************************->]0.155
[epoch 16] train loss: 0.149 test_accuracy: 0.937
 train loss: 100%[**************************************************->]0.235
[epoch 17] train loss: 0.143 test_accuracy: 0.945
 train loss: 100%[**************************************************->]0.105
[epoch 18] train loss: 0.139 test_accuracy: 0.948
 train loss: 100%[**************************************************->]0.121
[epoch 19] train loss: 0.148 test_accuracy: 0.940
 train loss: 100%[**************************************************->]0.072
[epoch 20] train loss: 0.157 test_accuracy: 0.956
 train loss: 100%[**************************************************->]0.115
[epoch 21] train loss: 0.159 test_accuracy: 0.934
 train loss: 100%[**************************************************->]0.457
[epoch 22] train loss: 0.139 test_accuracy: 0.956
 train loss: 100%[**************************************************->]0.101
[epoch 23] train loss: 0.120 test_accuracy: 0.962
 train loss: 100%[**************************************************->]0.142
[epoch 24] train loss: 0.134 test_accuracy: 0.962
 train loss: 100%[**************************************************->]0.283
[epoch 25] train loss: 0.136 test_accuracy: 0.962
 train loss: 100%[**************************************************->]0.034
[epoch 26] train loss: 0.137 test_accuracy: 0.951
 train loss: 100%[**************************************************->]0.317
[epoch 27] train loss: 0.120 test_accuracy: 0.948
 train loss: 100%[**************************************************->]0.049
[epoch 28] train loss: 0.113 test_accuracy: 0.967
 train loss: 100%[**************************************************->]0.083
[epoch 29] train loss: 0.136 test_accuracy: 0.973
 train loss: 100%[**************************************************->]0.042
[epoch 30] train loss: 0.112 test_accuracy: 0.948
Finished Training!
```  
测试日志：  
加入预训练的情况。  
```
(base) xilm@xilm-MS-7D17:~/python_file/img_classification$ /home/xilm/anaconda3/bin/python /home/xilm/python_file/img_classification/predict.py
tulips 0.9998856782913208
```  
![](https://github.com/XxxuLimei/classification_network/blob/main/ResNet/fly.jpg)
