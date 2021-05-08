import streamlit as st
import requests
import torch  
import time
from PIL import Image
from base64 import decodebytes

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.autoshape().eval()

st.sidebar.write('### Upload an image:')
uploaded_file = st.sidebar.file_uploader('',type=['png', 'jpg', 'jpeg'],accept_multiple_files=False)
st.sidebar.write('Created by Michaelia Tan Tong for DYOM Final Assignment using Streamlit, Pytorch and YOLOV5.')
st.sidebar.write('[Github Repo](https://github.com/Michaeliaaa/Object-Detection-Web-App-Tool)')

st.write('# Object Detection Tool')
st.write('Welcome to the interactive object detection tool. Feel free to try it out!')
st.write('To get started, upload an image to the sidebar on your left.')


if uploaded_file is None:
    url = 'https://source.unsplash.com/random/1000x1000?sig=incrementingIdentifier'
    image = Image.open(requests.get(url, stream=True).raw)
else:
    image = Image.open(uploaded_file)
    st.sidebar.success('Success!')

start_time = time.time()
img = model(image)
img.render()
result = Image.fromarray(img.imgs[0])
end_time = time.time()

with st.spinner('In progress...'):
    time.sleep(1)
st.write('### Result')

st.write('Inference time = {:.2f}s'.format(end_time - start_time))
st.write('\n')

# Display image.
st.image(result, use_column_width=True)