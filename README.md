Vietnamese:

Đính kèm gồm có:
Docker:
 
https://drive.google.com/file/d/1zgZyRWfn-VvbSSEFiZU4xZfnb_xt_JR9/view?usp=sharing

md5: 5d4c427516736696e529517c9afdc277


Chứa toàn bộ source & model:

Gồm có:
+ Tiền xử lý dữ liệu gồm có move.py, preprocessing.py
+ train.sh để preprocessing và train lại toàn bộ model.
+ predict.sh để test
+ model gồm có:
1.	Deep được lưu tại model/src/checkpoint
2.	Kết hợp model: xgb_search.sav
Các bước train lại model:
1.	Copy dữ liệu gồm train.csv & train/*mp3 vào docker
2.	Chạy sh train.sh: Hoặc chạy từng bước trong file train.sh

Khi train thì cần chú ý: 
+ Train nhiều model deep 
+ Train ensamble model: xgb_train.py để có model cuối cùng là xgb_model.sav

Các bước test:
Làm theo hướng dẫn bên Zalo chỉ việc chạy dòng lệnh:

sudo docker run -v  <nơi dữ liệu>:/data -v <nơi nhận kết quả>:/result vimentor_zaloai_music /bin/bash /model/predict.sh
Ví dụ: sudo docker run -v /home/cuong/Zalo/submit:/data -v /home/cuong/Zalo:/result vimentor_zaloai_music:ver1 /bin/bash /model/predict.sh

Thuật toán:

Bước 1. Di chuyển train/*mp3 thành 2 tập train/*mp3 và val/*mp3 thành 2 folders riêng biệt. Mục đích việc này để tiến hành cross-validation cho tiện ở ensanble method.

Bước 2: Xử lý dữ liệu đầu vào từ file mp3 để biến thành các file *npy chứa nội dung melspectrogram. Mỗi file mp3 được cắt ngẫu nhiên thành n file npy tùy thuộc vào up or down sampling. Tham khảo code để biết thêm chi tiết các cắt

Bước 3. Train dữ liệu này sử dụng transfer learning với mạng resnet18 và 34 tùy thuộc vào các hyper-parameters. Do ko có nhiều thời gian nên tôi chỉ train 4 model.

Bước 4. Ta đã có 5 deep models.
Dùng 5 model đó tiến hành inferencing tập validation và tập train thành các file có tên là “features_raw_fea_S_%s_1.csv” đó là các features cho ensamble model. 
Một file mp3 được cắt thành 8 files: 5 files đầu liên tiếp nhau và 3 files tiếp theo có điểm bắt đầu cắt là ngẫu nhiên.  Kết quả inferencing cho mỗi file là softmax của deep model. 
	Số lượng features sẽ là 10 * 8 * 5 (classes * cutting number * number of model)

Bước 5. Sau quá trình train & test thì tôi chọn Xgboost
+ để tuning hyper-parameters của Xgboost model sử dụng BayesianOptimization
Chỉ chạy trên tập validation. (có thể đạt max acc trên tập validation là 80+%)
+ để có được model cuối cùng thì sử dụng parameters phía trên train lại xgboost model với input là toàn bộ train và validation data.

Bước 6 Test  ( submit test chỉ đạt được 79.67+%) 


English

will update ...

