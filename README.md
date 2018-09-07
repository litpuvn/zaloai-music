Môi trường:
ffmpeg, python 2.7

packages của python: pandas, sklearn, torch, torchvision, librosa, PIL, xgboost, GPy, GPyOpt

Docker:

version_1: đã submit

https://drive.google.com/file/d/1s1cdEj-KZOGotgh8qKlrduwxTnePxUso/view
md5: d35b3ca8481df143ce96c8692340df83

version_2: muốn submit thử 
https://drive.google.com/file/d/1zgZyRWfn-VvbSSEFiZU4xZfnb_xt_JR9/view?usp=sharing

md5: 5d4c427516736696e529517c9afdc277


Tuy nhiên trong version 1 mình đã submit nhầm version và đã liên hệ với BTC để sửa lại file về preprocessing.py
Do mạng chậm quá nên mình không up lại được docker của version_1
các bạn có thể tự sửa lại docker bằng cách sau:
sửa lại dòng 75 của file: preprocessing.py
Từ:
    im = Image.fromarray(np.uint8(cm.binary(normalize(y,-80,0))*255))
Thành:
    im = Image.fromarray(np.uint8(cm.seismic(normalize(y,-80,0))*255))

Download dữ liệu
+ wget https://dl.challenge.zalo.ai/music/train.csv
+ wget https://dl.challenge.zalo.ai/music/test.csv
+ wget https://dl.challenge.zalo.ai/music/genres.csv
+ wget https://dl.challenge.zalo.ai/music/sample_submission.csv
+ wget https://dl.challenge.zalo.ai/music/train.zip
+ wget https://dl.challenge.zalo.ai/music/test.zip
+ wget https://dl.challenge.zalo.ai/music/private.zip
+ unzip -a train.zip
+ unzip -a test.zip
+ unzip -a private.zip

Load docker
+ sudo nvidia-docker load -i vimentor_zaloai_music
+ sudo nvidia-docker iamges
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
<none>              <none>              7a36a4586588        3 days ago          4.84GB
nvidia/cuda         latest              04a9ce0dec6d        4 weeks ago         1.96GB
+ docker tag 7a36a4586588 vimentor_zaloai_music:latest
Test:
+ sudo docker run -v  <nơi dữ liệu>:/data -v <nơi nhận kết quả>:/result <tên hoặc ID của file ảnh đã import vào docker> /bin/bash /model/predict.sh
Ví dụ:
sudo nvidia-docker run -v /home/cuong/Zalo/private:/data -v /home/cuong/Zalo/private:/result vimentor_zaloai_music /bin/bash /model/predict.sh

Train:
Ý tưởng của bài toán cũng giống các bạn khác đã chia sẻ trên blog:
+ chia file nhạc thành nhiều đoạn nhỏ cà convert chúng thành dạng phổ (melspectrogram) và chỉ dựa vào mỗi đoạn nhạc đó sẽ đoán ra được bài nhạc thuộc lớp nào.
Bằng cách này có thể giảm được inbalance trong tập dữ liệu.

+ Sử dụng pretrain model resnet18, resnet34. Mình cũng đã viết các models khác nhau trong code nhưng do thiêu thời gian và sức mạnh tính toán nên dùng vậy thôi.
 Các bạn có thể chọn model khác để train tùy vào time cũng như GPUs mà bạn có.
+ Sau khi có được khoảng 4 models từ các hyper-parameters khác nhau thì mình kết hợp chúng với nhau qua cách sau:
   + Ở bước training cho mạng deep mình đã upsample và mỗi mp3 có rất nhiều file nhỏ khác nhau,
   tuy nhiên khi kết hợp các model ở bước này mình chỉ chọn 8 mảnh đầu tiên để inferencing.
   + Mình sử dụng đầu ra của lớp cuối (softmax) của mỗi mode làm features 
   + Tiến hành inferencing tập validation và tập training thành các file có tên là “features_raw_fea_S_%s_1.csv” đó là các features cho ensamble model của mình. 
   + Mình có 4 model nên: Số lượng features cho ensamble model sẽ là 10 * 8 * 5 (number of classes * subsample number * number of models)
   + Tiến hành train và tune ensamble models: randomforest -> GradientBoosting -> XGBoost. Cuối cùng chọn XGboost.
    Khi tune XGboost mình chọn BayesianOptimization và chỉ dùng tập validation để tune cho nhanh tiết kiệm thời gian và phù hợp với năng lực tính toán của máy mình có.
	Khi đã có được bộ hyper-parameters tốt nhất cho ensamble model thì mình đã train lại ensamble model với cả tập train và validation.
+ Kết thúc quá trình mình có 4 deep models và một xgboost model để kết hợp chúng lại với nhau.   

Cụ thể như sau:

Bước 1: Chia tập dữ liệu
Để train dữ liệu mình đã chia tập train chứa các file mp3 thành 2 tập: cho validation chiếm 15% và cho training chiếm 85%
+ python move.py 

Bước 2: preprocessing

Mình tiến hành upsample như sau: {1:12,2:10,3:10,4:7,5:5,6:5,7:10,8:10,9:10,10:12,-1:3},
 ở đây key là loại nhạc (chưa phân loại là -1) và value chính là số mẫu sẽ cắt ra. 
Ví dụ: Loại 1 thì cắt thành 12 bản.
Tất cả các trường hợp mình đều cắt ít nhất 5 mẫu có quy luật, phần còn lại là ngẫu nhiên nhưng đều đảm bảo độ dài mỗi mẫu là như nhau.

Để giảm thời gian preprocessing trong quá trình train và tune, mình đã chỉ làm preprocessing một lần để không nhất thiết mỗi lần tune lại phải chạy prprocessing lại.
+ python preprocessing.py --folder "train" --type "train"
+ python preprocessing.py --folder "val" --type "val"

Bước 3: train 4 models

3 models đầu tiên:
+ python train.py --n_model "fea_S" --num_workers 5 --batch_size 128  --depth 18 --lr 0.0001 --train_from 1 --weight_decay 1e-4 --l1 0 --l2 0 -fu 2
+ python train.py --n_model "fea_S" --num_workers 5 --batch_size 128  --depth 34 --lr 0.0001 --train_from 1 --weight_decay 1e-4 --l1 0 --l2 0 -fu 2
+ python train.py --n_model "fea_S" --num_workers 5 --batch_size 128  --depth 18 --lr 0.0005 --train_from 1 --weight_decay 1e-4  --l1 0 --l2 0 -fu 0

1 model cuối: Mục đích tăng thêm dữ liệu nên mình đã resample lại lần nữa.
+ python preprocessing.py --folder "train" --type "train" --upsample 3
+ python train.py --n_model "fea_S" --num_workers 5 --batch_size 256  --depth 18 --lr 0.0001 --train_from 1 --weight_decay 1e-4 --l1 0 --l2 0 -fu 2 --downsample 30

Note: mỗi model mình đều train qua đêm và stop chúng vào ngày hôm sau để làm việc khác nên số lượng epochs mình cũng không nhớ rõ lắm chắc cỡ tầm 150-200 epochs.
( mình quên không note lại)

Bước 4: Inference tập train và validation
+ python inference.py --model_path <model1> --n_model "fea_S" --inf_folder "train" --depth 34 --ver 1
+ python inference.py --model_path <model1> --n_model "fea_S" --inf_folder "val" --depth 34 --ver 1
+ python inference.py --model_path <model2> --n_model "fea_S" --inf_folder "train" --depth 18 --ver 2
+ python inference.py --model_path <model2> --n_model "fea_S" --inf_folder "val" --depth 18 --ver 2
+ python inference.py --model_path <model3> --n_model "fea_S" --inf_folder "train" --depth 18 --ver 3
+ python inference.py --model_path <model3> --n_model "fea_S" --inf_folder "val" --depth 18 --ver 3
+ python inference.py --model_path <model4> --n_model "fea_S" --inf_folder "train" --depth 18 --ver 7
+ python inference.py --model_path <model4> --n_model "fea_S" --inf_folder "val" --depth 18 --ver 7

Bước 5: Tìm Hyper-parameters và train XGBoost model 
+ python xgb_train.py

<well done>

Còn nhiều điểm mình muốn cải thiện trong các phương pháp của mình.
+ Mình muốn cải thiện kết quả của 4 mạng resnet bằng model khác nếu tốc độ tính toán nhanh hơn.
+ Có thể áp dụng BayesianOptimization để tuning các hyper-parameters cho mạng deep và điều này cũng đòi hỏi tốc độ tính toán nhiều hơn.
+ Mình chỉ dùng melspectrogram, có thể sử dụng loại khác như MFCC.
 Hoặc thay vì bóc tách phổ (melspectrogram) chúng ta có thể bóc tách chromagram và sử dụng thêm kĩ thuật lọc nhiễu theo beat 
 tuy nhiên cách này mình thấy preprocessing hơi lâu quá nên mình stop nó luôn.
+ Mình còn đang suy nghĩ liệu LSTM có hữu dụng không đôi với chromagram sau khi lọc nhiễu qua beat.

Trong challenge lần này mình chủ yếu tập trung vào tune ensamble model vì nó nhanh và không yêu cầu tính toán nhiều.

Khi submit lên public test đôi khi đạt kết quả cao trên validation nhưng mà khi submit lên public test lại cho kết quả không được tốt.
nên mình có tạo ra 2 version docker ứng với bộ tham số khác nhau và mình cũng chia sẻ luôn, khi các bạn chạy lại phần test các bạn cũng sẽ nhận thấy điều đó.











