# CNNChineseClassifyer
This is an enhanced version of cnn-text-classification-tf-chinese. Original version can be found [here](https://github.com/indiejoseph/cnn-text-classification-tf-chinese) and [here](https://github.com/dennybritz/cnn-text-classification-tf)

## Requirements
- Python 2.7 or 3.6 (at least I tried them successfully)
- Tensorflow > 0.12 (this is the version I used when I released this version)
- Numpy
- Jieba
- Pandas

# Main features
- keep all features of original version.
- added text segmentation for Chinese with [jieba](https://github.com/fxsjy/jieba)
- changed the way to vocabulary to better evaluate Chinese text.
- added special evaluation file from [cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf) but it works for Chinese.
- it has been tested with health care data with good feedback on the accurancy of classification.

# To do features
- to extend one feature output into multiple feature output.
- to extend multiple machines running without changing codes.
- to support mixed English and Chinese text classifications.


# Training
Print parameters:
```
python train.py --help

optional arguments:
  -h, --help            show this help message and exit
  --unrolled_lstm [UNROLLED_LSTM]
                        use a statically unrolled LSTM instead of dynamic_rnn
  --nounrolled_lstm
  --dev_sample_percentage DEV_SAMPLE_PERCENTAGE
                        Percentage of the training data to use for validation
  --positive_data_file POSITIVE_DATA_FILE
                        Data source for the positive data.
  --negative_data_file NEGATIVE_DATA_FILE
                        Data source for the negative data.
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularization lambda (default: 0.0)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 200)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --num_checkpoints NUM_CHECKPOINTS
                        Number of checkpoints to store (default: 5)
  --allow_soft_placement [ALLOW_SOFT_PLACEMENT]
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement [LOG_DEVICE_PLACEMENT]
                        Log placement of ops on devices
  --nolog_device_placement
```

Train:
```
python train.py
```

# Evaluating
print parameters

```
python eval.py --help
optional arguments:
  -h, --help            show this help message and exit
  --unrolled_lstm [UNROLLED_LSTM]
                        use a statically unrolled LSTM instead of dynamic_rnn
  --nounrolled_lstm
  --test_data_file TEST_DATA_FILE
                        Data source for the positive data.
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --checkpoint_dir CHECKPOINT_DIR
                        Checkpoint directory from training run
  --allow_soft_placement [ALLOW_SOFT_PLACEMENT]
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement [LOG_DEVICE_PLACEMENT]
                        Log placement of ops on devices
  --nolog_device_placement
```

Predict 
```
python eval.py --test_data_file="your test data to be predicted" --checkpoint_dir="./runs/1459637919/checkpoints/"
```

Replace the checkpoint dir with the output from the training and test_data_file is the place you could use to 


# References
- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
