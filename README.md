1. tensorflow 的version换成1.14.0程序可以正常运行，之前的不能运行的版本是多少，也没有注意看到，之前会报Process finished with exit code -1073741819 (0xC0000005)在sess.run()这里停住报错。
2. 当训练完数据集之后，开始进行valudation，如果报ValueError: Variable layer1-conv1/weight already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:
这个错误的话，那么就重启IDE，然后重新执行valudation程序
