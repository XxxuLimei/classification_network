## 0405：  
- LeNet的网络结构：  
![LeNet](https://github.com/XxxuLimei/classification_network/blob/main/LeNet/WeChat%20Image_20230405140000.png)  
根据该网络参数，自己编写model.py文件；  
- 很好笑，预测的时候，使用飞机图片，预测出来是cat  
```
(base) xilm@xilm-MS-7D17:~/python_file/img_classification$ /home/xilm/anaconda3/bin/python /home/xilm/python_file/img_classification/train.py
[1,   500] train_loss: 1.752  test_accuracy: 0.442
[1,  1000] train_loss: 1.439  test_accuracy: 0.502
[2,   500] train_loss: 1.229  test_accuracy: 0.579
[2,  1000] train_loss: 1.158  test_accuracy: 0.595
[3,   500] train_loss: 1.032  test_accuracy: 0.614
[3,  1000] train_loss: 1.013  test_accuracy: 0.632
[4,   500] train_loss: 0.912  test_accuracy: 0.651
[4,  1000] train_loss: 0.907  test_accuracy: 0.652
[5,   500] train_loss: 0.826  test_accuracy: 0.660
[5,  1000] train_loss: 0.827  test_accuracy: 0.676
[6,   500] train_loss: 0.753  test_accuracy: 0.676
[6,  1000] train_loss: 0.763  test_accuracy: 0.686
[7,   500] train_loss: 0.692  test_accuracy: 0.674
[7,  1000] train_loss: 0.702  test_accuracy: 0.681
[8,   500] train_loss: 0.639  test_accuracy: 0.681
[8,  1000] train_loss: 0.674  test_accuracy: 0.676
[9,   500] train_loss: 0.597  test_accuracy: 0.678
[9,  1000] train_loss: 0.627  test_accuracy: 0.693
[10,   500] train_loss: 0.561  test_accuracy: 0.689
[10,  1000] train_loss: 0.584  test_accuracy: 0.688
Finished Training!
```  
```
(base) xilm@xilm-MS-7D17:~/python_file/img_classification$ /home/xilm/anaconda3/bin/python /home/xilm/python_file/img_classification/predict.py
cat
```  
