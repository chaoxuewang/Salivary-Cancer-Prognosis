# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import joblib


# 初始化基础特征
def get_base_info():
    return {key: 0 for key in [
        '性别-男', '性别-女', '年龄', '发病部位-腮腺', '发病部位-颌下腺', '发病部位-舌下腺+口底',
        '发病部位-腭', '发病部位-磨牙后区', '发病部位-颊', '发病部位-舌', '发病部位-唇', '发病部位-上颌',
        '发病部位-其他', '病理类型-鳞状细胞癌', '病理类型-淋巴上皮癌', '病理类型-上皮-肌上皮癌',
        '病理类型-嗜酸细胞腺癌', '病理类型-透明细胞癌', '病理类型-其他', '病理类型-高分化粘表',
        '病理类型-中分化粘表', '病理类型-低分化粘表', '病理类型-腺样囊性癌', '病理类型-癌在多形性腺瘤中',
        '病理类型-非特异性腺癌', '病理类型-腺泡细胞癌', '病理类型-肌上皮癌', '病理类型-多型性腺癌',
        '病理类型-基底细胞腺癌', '病理类型-唾液腺导管癌', 'T1', 'T2', 'T3', 'T4', 'N0', 'N1', 'N2', 'N3'
    ]}


# 页面设置
st.set_page_config(page_title="模型预测应用", layout="centered")

st.title("患者信息预测工具")
st.markdown("输入患者基本信息，预测模型结果。")

# 输入表单
with st.form("patient_form"):
    性别 = st.radio("性别", ["男", "女"])
    年龄 = st.number_input("年龄", min_value=1, max_value=120, step=1, value=30)
    发病部位 = st.selectbox("发病部位",
                            ["腮腺", "颌下腺", "舌下腺+口底", "腭", "磨牙后区", "颊", "舌", "唇", "上颌", "其他"])
    病理类型 = st.selectbox("病理类型", [
        "鳞状细胞癌", "淋巴上皮癌", "上皮-肌上皮癌", "嗜酸细胞腺癌", "透明细胞癌", "其他",
        "高分化粘表", "中分化粘表", "低分化粘表", "腺样囊性癌", "癌在多形性腺瘤中",
        "非特异性腺癌", "腺泡细胞癌", "肌上皮癌", "多型性腺癌", "基底细胞腺癌", "唾液腺导管癌"
    ])
    T_stage = st.selectbox("T分期", ["T1", "T2", "T3", "T4"])
    N_stage = st.selectbox("N分期", ["N0", "N1", "N2", "N3"])
    submit = st.form_submit_button("预测")

# 当用户提交表单时执行
if submit:
    info_dict = get_base_info()
    info_dict[f"性别-{性别}"] = 1
    info_dict["年龄"] = 年龄
    info_dict[f"发病部位-{发病部位}"] = 1
    info_dict[f"病理类型-{病理类型}"] = 1
    info_dict[T_stage] = 1
    info_dict[N_stage] = 1

    # 加载模型和特征
    objective = "prognosis"
    model_path = f"checkpoint/{objective}/lightgbm_model.pkl"
    feature_path = f"lightgbm_selected_features.txt"

    try:
        with open(model_path, 'rb') as f:
            lgb_model = joblib.load(f)
    
        with open(feature_path, 'rb') as f:
            selected_features = [line.strip() for line in f]

        df_case_info = pd.DataFrame([info_dict])
        data = df_case_info[selected_features]
        predict_prob = lgb_model.predict_proba(data)[:, 1][0]

        st.success(f"预测完成，患者{objective}的概率为：{predict_prob:.2%}")
    except Exception as e:
        st.error(f"模型加载或预测失败：{e}")
