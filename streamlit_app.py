import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog
from fpdf import FPDF
import datetime
import pandas as pd

# Sahifa sozlamalari
st.set_page_config(page_title="Решатель ЛП", layout="wide")

# --- XOTIRA ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- TIL MANTIQI ---
if 'lang' not in st.session_state:
    st.session_state.lang = "Русский"

if st.session_state.lang == "O'zbekcha":
    t_title, t_target, t_cons, t_add, t_solve, t_pdf, t_hist, t_analysis, t_edit_done = \
    "📊 Chiziqli dasturlash", "🎯 Maqsad funksiyasi", "🚧 Cheklovlar", "+ Qo'shish", "🚀 Hisoblash", \
    "📥 PDF yuklash", "📜 Tarix", "🔍 Tahlil", "✅ Yakunlash"
else:
    t_title, t_target, t_cons, t_add, t_solve, t_pdf, t_hist, t_analysis, t_edit_done = \
    "📊 Решатель ЛП", "🎯 Целевая функция", "🚧 Ограничения", "+ Добавить", "🚀 Решить", \
    "📥 Скачать PDF", "📜 История", "🔍 Анализ", "✅ Завершить"

st.markdown(f"<h1 style='text-align: center;'>{t_title}</h1>", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header(t_target)
    col_v1, col_x, col_v2, col_y, col_t = st.columns([2, 1, 2, 1, 3])
    with col_v1: c_main1 = st.number_input("C1", value=1.0, key="main_c1", label_visibility="collapsed")
    with col_v2: c_main2 = st.number_input("C2", value=2.0, key="main_c2", label_visibility="collapsed")
    with col_t: obj_type = st.selectbox("Type", ("max", "min"), key="main_type", label_visibility="collapsed")
    
    st.header(t_cons)
    if 'constraints' not in st.session_state:
        st.session_state.constraints = [{'a': 1.0, 'b': 2.0, 'op': '≤', 'c': 6.0}, {'a': 2.0, 'b': 1.0, 'op': '≥', 'c': 6.0}]
    
    new_cons = []
    for i, cons in enumerate(st.session_state.constraints):
        cl1, cl2, cl3, cl4, cl5 = st.columns([2, 2, 1.5, 2, 1])
        with cl1: a_val = st.number_input(f"a{i}", value=float(cons['a']), key=f"inp_a{i}", label_visibility="collapsed")
        with cl2: b_val = st.number_input(f"b{i}", value=float(cons['b']), key=f"inp_b{i}", label_visibility="collapsed")
        with cl3: op_val = st.selectbox(f"op{i}", ("≤", "≥", "="), index=("≤", "≥", "=").index(cons['op']), key=f"inp_op{i}", label_visibility="collapsed")
        with cl4: c_val = st.number_input(f"c{i}", value=float(cons['c']), key=f"inp_c{i}", label_visibility="collapsed")
        with cl5: 
            if st.button("🗑️", key=f"btn_del{i}"):
                st.session_state.constraints.pop(i); st.rerun()
        new_cons.append({'a': a_val, 'b': b_val, 'op': op_val, 'c': c_val})
    
    st.session_state.constraints = new_cons
    if st.button(t_add): st.session_state.constraints.append({'a': 1.0, 'b': 1.0, 'op': '≤', 'c': 10.0}); st.rerun()
    
    edit_done = st.checkbox(t_edit_done, value=False)
    solve_btn = st.button(t_solve, type="primary", use_container_width=True) if edit_done else False
    st.session_state.lang = st.radio("🌐 Til", ("Русский", "O'zbekcha"), horizontal=True)

# --- YECHIM VA GRAFIK ---
if solve_btn:
    coeffs = [-c_main1 if obj_type == "max" else c_main1, -c_main2 if obj_type == "max" else c_main2]
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for c in st.session_state.constraints:
        if c['op'] == '≤': A_ub.append([c['a'], c['b']]); b_ub.append(c['c'])
        elif c['op'] == '≥': A_ub.append([-c['a'], -c['b']]); b_ub.append(-c['c'])
        else: A_eq.append([c['a'], c['b']]); b_eq.append(c['c'])
    
    res = linprog(coeffs, A_ub=A_ub or None, b_ub=b_ub or None, A_eq=A_eq or None, b_eq=b_eq or None, bounds=(0, None), method='highs')
    
    fig = go.Figure()
    limit = 10
    x_vals = np.linspace(0, limit, 400)

    # ODR hisoblash (Faqat x1, x2 >= 0 uchun)
    for i, c in enumerate(st.session_state.constraints):
        if abs(c['b']) > 1e-7:
            y_line = (c['c'] - c['a'] * x_vals) / c['b']
            fig.add_trace(go.Scatter(x=x_vals, y=y_line, mode='lines', name=f"L{i+1}"))

    if res.success:
        opt_x, opt_y = res.x
        
        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(showgrid=False, zeroline=True, zerolinecolor='black', zerolinewidth=2, 
                       range=[-0.5, limit], dtick=1, ticks="outside", ticklen=10),
            yaxis=dict(showgrid=False, zeroline=True, zerolinecolor='black', zerolinewidth=2, 
                       range=[-0.5, limit], dtick=1, ticks="outside", ticklen=10),
            showlegend=True, height=700
        )

        # --- STRELKALARNI TO'G'RILASH ---
        # X o'qi strelkasi (aynan chiziq uchiga)
        fig.add_trace(go.Scatter(
            x=[limit, limit-0.2, limit, limit-0.2], 
            y=[0, 0.15, 0, -0.15],
            mode='lines', line=dict(color='black', width=2), showlegend=False
        ))
        # Y o'qi strelkasi (aynan chiziq uchiga)
        fig.add_trace(go.Scatter(
            x=[0, 0.15, 0, -0.15], 
            y=[limit, limit-0.2, limit, limit-0.2],
            mode='lines', line=dict(color='black', width=2), showlegend=False
        ))

        # O'q nomlari
        fig.add_annotation(x=limit, y=-0.4, text="<b>x₁</b>", showarrow=False, font=dict(size=16))
        fig.add_annotation(x=-0.4, y=limit, text="<b>x₂</b>", showarrow=False, font=dict(size=16))

        # Optimal nuqta
        fig.add_trace(go.Scatter(x=[opt_x], y=[opt_y], mode='markers+text', 
                                 text=["Opt"], textposition="top right",
                                 marker=dict(color='gold', size=15, symbol='star')))

        st.plotly_chart(fig, use_container_width=True)
        st.success(f"### Z* = {c_main1*opt_x + c_main2*opt_y:.2f} (X1={opt_x:.2f}, X2={opt_y:.2f})")
