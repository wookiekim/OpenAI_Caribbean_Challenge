python train_validation.py \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --momentum 0.5 \
  --weight_decay 0.01 \
  --epoch 200 \
  --model_name resnet_valtest.pt \
  --pretrained_name resnet18 \
  --train_all False \
  --upsampling True \
  --retrain True \
  --patience 15

python test.py \
  --model_name resnet_valtest.pt \
  --pretrained_name resnet18
