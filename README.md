# Nhận diện chữ viết tay 
# 1. Clone code về /home(mở terminal)

 `git clone https://github.com/TUDONG05/OCR.git`

# 2. Chạy download_data.py để tải dữ liệu IAM

 `python download_data.py`

# 3. Train model bằng cách chạy train.py
 ( Trước đó nhớ sửa lại các đường dẫn trong file config.py, predict.py nếu chạy bị lỗi) 

# 4. Đánh giá
 Chạy predict.py để đánh giá ngẫu nhiên và tính WER , CER cả bộ 

 `python predict.py` 
 
# 5. Chạy website
`streamlit run app.py`