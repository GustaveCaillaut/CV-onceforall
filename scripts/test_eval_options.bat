python ofa_cifar10_options.py eval_subnet ^
  --data-dir ./data ^
  --checkpoint ./runs/ofa_cifar10/checkpoints/supernet_best.pt ^
  --spec-json largest_spec.json ^
  --bn-calib-batches 20 ^
  --test-batches 20 ^
  --batch-size 512 ^
  --num-workers 0 ^
  --eval-preset no_opti