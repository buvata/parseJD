import streamlit as st
from train import *


def load_text(model_dir, text):
    JD = JDParser(model_dir)
    info = JD.parse(text)
    return info

def main():

    VALUES = ["Tên người liên hệ", "Email", "Địa điểm", "Thời gian tuyển", "SĐT", "Tên công ty", "Vị trí tuyển", "Số lượng tuyển", "Mức lương"]

    st.title('Phân Tích Post Tuyển Dụng')
    model_dir = "nlp_jd_model_v2/"
    text = st.text_area("Nhập văn bản:")
    if st.button("Submit"):
        info_new = {}
        info = load_text(model_dir, text)
        for i, tag in enumerate(VALUE_ENTITIES):
            if VALUES[i] not in info_new:
                info_new[VALUES[i]] = []
            info_new[VALUES[i]].append(info[tag])

        for i in info_new:
            st.subheader(i)
            st.text(info_new[i][0])



if __name__ == '__main__':
    main()
