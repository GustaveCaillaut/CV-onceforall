python ofa_cifar10_options.py train_acc_predictor ^
  --dataset ./runs/ofa_cifar10/acc_dataset_1000.jsonl ^
  --save-dir ./runs/ofa_cifar10 ^
  --predictor-epochs 200 ^
  --predictor-batch-size 64 ^
  --predictor-lr 0.001