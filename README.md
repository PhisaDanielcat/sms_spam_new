使用顺序:

0. pip install -r requirements.txt
1. extract_dataset.py 下载sms_spam_dataset.zip数据集，并解压和划分训练集，验证集和测试集到datasets文件夹
2. TransformerClassifier_model.py ，模型文件结构打印，可以调整
3. train.py 调用TransformerClassifier_model.py中的模型文件并在sms_spam_dataset下训练，打印出训练结果并保存模型到models文件夹
4. load_and_evaluation.py 载入参数并打印出在sms_spam_dataset验证集下的精度
5. params_monitor.py 打印模型各层结构和总的参数量
6. trace_params.py 给入一个固定输入，追踪模型所有层的输入输出结果，并将其和参数一并打印到params下的txt文件中
7. reconstruct.py 使用numpy重构模型的整个数据流，与torch的结果完全一致，便于FPGA实现网络