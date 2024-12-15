import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# 모델을 미리 학습시키고 저장한 후 불러오는 방식입니다.
model = tf.keras.models.load_model('your_trained_model.h5')

# 맵 이름을 예측하는 함수
def predict_map(image):
    # 이미지를 모델에 맞게 전처리
    image = image.resize((224, 224))  # 모델에 맞는 크기로 조정
    image = np.array(image) / 255.0  # 정규화
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    prediction = model.predict(image)  # 예측
    map_index = np.argmax(prediction)  # 예측된 맵의 인덱스
    
    # 인덱스에 해당하는 맵 이름 반환
    map_names = ['Bind', 'Haven', 'Icebox', 'Ascent', 'Split']  # 예시
    return map_names[map_index]

# 웹페이지 설정
st.title("발로란트 맵 인식기")
st.write("발로란트의 맵 이미지를 업로드하세요.")

# 이미지 업로드
uploaded_image = st.file_uploader("맵 이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='업로드된 이미지', use_column_width=True)

    # 예측 버튼
    if st.button('맵 인식하기'):
        map_name = predict_map(image)
        st.write(f"이 맵은 {map_name} 입니다!")
