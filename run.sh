# Training
python train.py \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --momentum 0.5 \
  --weight_decay 0.01 \
  --epoch 150 \
  --model_name temp.pt \
  --pretrained_name resnet18 \
  --train_all False \
  --upsampling True \
  --retrain True \
  --patience 20 \
  --note with_class_weights_maxOverMin

# Testing
#    if retrain==True, and you want to test on the retrained model
#    append "retrained_" in front of model_name
python test.py \
  --model_name temp.pt \
  --pretrained_name resnet18

python test.py \
  --model_name retrained_temp.pt \
  --pretrained_name resnet18
