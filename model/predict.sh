cd /model/

rm -rf test

python preprocessing.py --folder "/data/" --type "test"

cd /model/src

python inference.py --model_path "0828_2354" --n_model "fea_S" --inf_folder "test" --depth 34 --ver 1
python inference.py --model_path "0827_2244" --n_model "fea_S" --inf_folder "test" --depth 18 --ver 2
python inference.py --model_path "0827_2302" --n_model "fea_S" --inf_folder "test" --depth 18 --ver 3
python inference.py --model_path "0831_1731" --n_model "fea_S" --inf_folder "test" --depth 18 --ver 7

cd /model
python xgb_predict.py

rm "test.csv"
rm "top_fea_S*"
rm "features_raw*"
rm "features_fea*"
rm "submission_fea_*"
rm -rf test
