## 0406：  
- VGG网络结构：  
![](https://github.com/XxxuLimei/classification_network/blob/main/VGG/block.png)
- 网络参数：  
![](https://github.com/XxxuLimei/classification_network/blob/main/VGG/WeChat%20Image_20230405171608.png)  
- 训练过程如下：  
```
(base) xilm@xilm-MS-7D17:~/python_file/img_classification$ /home/xilm/anaconda3/bin/python /home/xilm/python_file/img_classification/train.py
cuda:0
 train loss: 100%[**************************************************->]1.188
46.39243607899698
[epoch 1] train_loss: 1.445 test_accuracy: 0.404
 train loss: 100%[**************************************************->]1.245
50.117353347013704
[epoch 2] train_loss: 1.327 test_accuracy: 0.407
 train loss: 100%[**************************************************->]1.228
50.33390638900164
[epoch 3] train_loss: 1.326 test_accuracy: 0.407
 train loss: 100%[**************************************************->]0.938
50.41113602599944
[epoch 4] train_loss: 1.302 test_accuracy: 0.426
 train loss: 100%[**************************************************->]1.192
50.55425647800439
[epoch 5] train_loss: 1.287 test_accuracy: 0.431
 train loss: 100%[**************************************************->]1.357
50.858662833998096
[epoch 6] train_loss: 1.302 test_accuracy: 0.451
 train loss: 100%[**************************************************->]1.082
50.36560545999964
[epoch 7] train_loss: 1.259 test_accuracy: 0.453
 train loss: 100%[**************************************************->]1.082
50.1896048149938
[epoch 8] train_loss: 1.276 test_accuracy: 0.478
 train loss: 100%[**************************************************->]0.740
50.37754138298624
[epoch 9] train_loss: 1.218 test_accuracy: 0.514
 train loss: 100%[**************************************************->]1.355
50.36729804400238
[epoch 10] train_loss: 1.181 test_accuracy: 0.459
 train loss: 100%[**************************************************->]0.820
50.59076723999169
[epoch 11] train_loss: 1.147 test_accuracy: 0.552
 train loss: 100%[**************************************************->]0.990
50.32842817400524
[epoch 12] train_loss: 1.082 test_accuracy: 0.555
 train loss: 100%[**************************************************->]0.529
50.34816424999735
[epoch 13] train_loss: 1.054 test_accuracy: 0.607
 train loss: 100%[**************************************************->]0.833
50.269509076999384
[epoch 14] train_loss: 0.998 test_accuracy: 0.602
 train loss: 100%[**************************************************->]0.549
50.47509729999001
[epoch 15] train_loss: 0.956 test_accuracy: 0.654
 train loss: 100%[**************************************************->]0.523
50.16135985999426
[epoch 16] train_loss: 0.996 test_accuracy: 0.635
 train loss: 100%[**************************************************->]0.715
50.14356689300621
[epoch 17] train_loss: 0.926 test_accuracy: 0.635
 train loss: 100%[**************************************************->]0.973
50.16222157800803
[epoch 18] train_loss: 0.887 test_accuracy: 0.681
 train loss: 100%[**************************************************->]0.751
50.09126782399835
[epoch 19] train_loss: 0.868 test_accuracy: 0.629
 train loss: 100%[**************************************************->]0.992
50.113285907005775
[epoch 20] train_loss: 0.845 test_accuracy: 0.668
 train loss: 100%[**************************************************->]0.836
50.12942884900258
[epoch 21] train_loss: 0.836 test_accuracy: 0.706
 train loss: 100%[**************************************************->]0.784
50.20124581600248
[epoch 22] train_loss: 0.835 test_accuracy: 0.681
 train loss: 100%[**************************************************->]0.669
50.16383170899644
[epoch 23] train_loss: 0.806 test_accuracy: 0.690
 train loss: 100%[**************************************************->]0.785
50.15066898499208
[epoch 24] train_loss: 0.795 test_accuracy: 0.670
 train loss: 100%[**************************************************->]1.293
50.123613735006074
[epoch 25] train_loss: 0.777 test_accuracy: 0.703
 train loss: 100%[**************************************************->]0.589
50.18850477799424
[epoch 26] train_loss: 0.732 test_accuracy: 0.712
 train loss: 100%[**************************************************->]0.643
50.15636262198677
[epoch 27] train_loss: 0.706 test_accuracy: 0.745
 train loss: 100%[**************************************************->]1.128
50.130740585998865
[epoch 28] train_loss: 0.731 test_accuracy: 0.714
 train loss: 100%[**************************************************->]0.908
50.16293578200566
[epoch 29] train_loss: 0.709 test_accuracy: 0.706
 train loss: 100%[**************************************************->]0.689
50.17795212499914
[epoch 30] train_loss: 0.708 test_accuracy: 0.720
Finished Training!
```  
- 测试结果很无语，预测成蒲公英了。。。大网络果然需要预训练啊  
![](https://github.com/XxxuLimei/classification_network/blob/main/VGG/fly.jpg)  
```
(base) xilm@xilm-MS-7D17:~/python_file/img_classification$ /home/xilm/anaconda3/bin/python /home/xilm/python_file/img_classification/predict.py
dandelion 0.3050488829612732
```  
