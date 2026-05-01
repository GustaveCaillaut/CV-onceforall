python ofa_cifar10_options.py eval_subnet ^
  --data-dir ./data ^
  --checkpoint ./runs/ofa_cifar10/checkpoints/supernet_last.pt ^
  --spec-json ./runs/ofa_cifar10/search/best_spec.json ^
  --bn-calib-batches 20 ^
  --batch-size 512 ^
  --num-workers 0 ^
  --eval-preset accurate