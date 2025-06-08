# app.py
import streamlit as st
import cv2, tempfile
import numpy as np

from measure import measure_from_img, draw_landmarks

st.set_page_config(page_title="AI 身形量測 MVP", page_icon="📐", layout="centered")

st.title("📐 AI 身形量測 MVP")
st.markdown("上傳一張**完整全身照**（建議旁邊放 A4 紙作比例尺），"
            "點 **開始量測** 後即可得到肩寬、骨盆寬等數據。")

uploaded = st.file_uploader("請選擇圖片", type=["jpg", "jpeg", "png"])

if uploaded:
    # 先把 Bytes 轉成 OpenCV 影像
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="原始影像", use_column_width=True)

    if st.button("開始量測"):
        with st.spinner("模型推論中…"):
            try:
                info = measure_from_img(img_bgr, ref_method="head")
            except ValueError as e:
                st.error(str(e))
            else:
                # 顯示數值
                st.success("量測完成！")
                st.subheader("📏 量測結果（公分）")
                st.json(info["cm"])

                # 再畫一次骨架給使用者看
                import mediapipe as mp
                with mp.solutions.pose.Pose(static_image_mode=True) as pose:
                    res = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                annotated = draw_landmarks(img_bgr, res)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         caption="骨架偵測", use_column_width=True)
