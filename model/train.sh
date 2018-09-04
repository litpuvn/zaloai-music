# Train deep model

cd /model/

python move.py
rm /model/train/*npy
rm -rf /model/train/fea_*

rm /model/val/*npy
rm -rf /model/val/fea_*

python preprocessing.py --folder "train" --type "train"
python preprocessing.py --folder "val" --type "val"

python train.py --n_model "fea_S" --num_workers 5 --batch_size 128  --depth 18 --lr 0.0001 --train_from 1 --weight_decay 1e-4 --l1 0 --l2 0 -fu 2
python train.py --n_model "fea_S" --num_workers 5 --batch_size 128  --depth 34 --lr 0.0001 --train_from 1 --weight_decay 1e-4 --l1 0 --l2 0 -fu 2
python train.py --n_model "fea_S" --num_workers 5 --batch_size 128  --depth 18 --lr 0.0005 --train_from 1 --weight_decay 1e-4  --l1 0 --l2 0 -fu 0

rm -rf /model/train/fea_*
python preprocessing.py --folder "train" --type "train" --upsample 3

python train.py --n_model "fea_S" --num_workers 5 --batch_size 256  --depth 18 --lr 0.0001 --train_from 1 --weight_decay 1e-4 --l1 0 --l2 0 -fu 2 --downsample 30


cd /model/src

# infer is to create features for next xgb model
python inference.py --model_path "0828_2354" --n_model "fea_S" --inf_folder "train" --depth 34 --ver 1
python inference.py --model_path "0828_2354" --n_model "fea_S" --inf_folder "val" --depth 34 --ver 1


python inference.py --model_path "0827_2244" --n_model "fea_S" --inf_folder "train" --depth 18 --ver 2
python inference.py --model_path "0827_2244" --n_model "fea_S" --inf_folder "val" --depth 18 --ver 2

python inference.py --model_path "0827_2302" --n_model "fea_S" --inf_folder "train" --depth 18 --ver 3
python inference.py --model_path "0827_2302" --n_model "fea_S" --inf_folder "val" --depth 18 --ver 3


python inference.py --model_path "0831_1731" --n_model "fea_S" --inf_folder "train" --depth 18 --ver 7
python inference.py --model_path "0831_1731" --n_model "fea_S" --inf_folder "val" --depth 18 --ver 7




cd /model/
# train boosting modelk

python xgb_train.py
