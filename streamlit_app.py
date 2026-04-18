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
    t_title = "📊 Chiziqli dasturlash — Yechuvchi"
    t_target = "🎯 Maqsad funksiyasi"
    t_cons = "🚧 Cheklovlar"
    t_add = "+ Cheklov qo'shish"
    t_solve = "🚀 Hisoblash"
    t_pdf = "📥 PDF hisobotni yuklash (Barcha tarix)"
    t_hist = "📜 Yechimlar tarixi"
    t_analysis = "🔍 Masala tahlili va Sezgirlik"
    t_edit_done = "✅ Tahrirlashni yakunlash"
else:
    t_title = "📊 Линейное программирование — Решатель"
    t_target = "🎯 Целевая функция"
    t_cons = "🚧 Ограничения"
    t_add = "+ Добавить ограничение"
    t_solve = "🚀 Решить"
    t_pdf = "📥 Скачать отчёт PDF (Вся история)"
    t_hist = "📜 История решений"
    t_analysis = "🔍 Анализ задачи и Чувствительность"
    t_edit_done = "✅ Завершить редактирование"

st.markdown(f"<h1 style='text-align: center;'>{t_title}</h1>", unsafe_allow_html=True)

# --- PDF FUNKSIYASI ---
def create_pdf(history):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for i, item in enumerate(history):
        pdf.add_page()
        pdf.set_font("Arial", 'B', size=16)
        pdf.cell(200, 10, txt=f"Reshenie zadachi No{len(history)-i}", ln=True, align='C')
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt=f"Vremya: {item['time']}", ln=True, align='C')
        pdf.ln(5)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(200, 10, txt="1. Selevaya funksiya:", ln=True)
        pdf.set_font("Arial", size=12)
        target_txt = f"F(X) = {item.get('c1', 0)}x1 + ({item.get('c2', 0)})x2 -> {item['type']}"
        pdf.cell(200, 10, txt=target_txt, ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(200, 10, txt="2. Ogranicheniya:", ln=True)
        pdf.set_font("Arial", size=12)
        if 'constraints_text' in item:
            for cons in item['constraints_text']:
                safe_text = cons.replace('≤', '<=').replace('≥', '>=')
                pdf.cell(200, 8, txt=f"   {safe_text}", ln=True)
        pdf.cell(200, 8, txt="   x1 >= 0, x2 >= 0", ln=True)
        pdf.ln(10)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(200, 10, txt="3. Resultat:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 8, txt=f"Optimalnaya tochka: X1 = {item['x']:.2f}, X2 = {item['y']:.2f}", ln=True)
        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(200, 8, txt=f"Z* = {item['z']:.2f}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# --- SIDEBAR ---
with st.sidebar:
    st.header(t_target)
    col_v1, col_x, col_v2, col_y, col_t = st.columns([2, 1, 2, 1, 3])
    with col_v1: c_main1 = st.number_input("C1", value=1.0, format="%.1f", key="main_c1", label_visibility="collapsed")
    with col_x: st.markdown("<div style='margin-top: 5px;'><sup>*x</sup> +</div>", unsafe_allow_html=True)
    with col_v2: c_main2 = st.number_input("C2", value=2.0, format="%.1f", key="main_c2", label_visibility="collapsed")
    with col_y: st.markdown("<div style='margin-top: 5px;'><sup>*y</sup></div>", unsafe_allow_html=True)
    with col_t: obj_type = st.selectbox("Тип", ("max", "min"), key="main_type", label_visibility="collapsed")
    st.markdown("---")
    st.header(t_cons)
    if 'constraints' not in st.session_state:
        st.session_state.constraints = [{'a': 1.0, 'b': 2.0, 'op': '≤', 'c': 6.0}, {'a': 2.0, 'b': 1.0, 'op': '≥', 'c': 6.0}]
    
    new_cons = []
    for i, cons in enumerate(st.session_state.constraints):
        cl1, cl_x, cl2, cl_y, cl3, cl4, cl5 = st.columns([2, 1.2, 2, 1, 1.5, 2, 1])
        with cl1: a_val = st.number_input(f"a{i}", value=float(cons['a']), key=f"inp_a{i}", label_visibility="collapsed")
        with cl_x: st.markdown("<div style='margin-top: 5px;'><sup>*x</sup> +</div>", unsafe_allow_html=True)
        with cl2: b_val = st.number_input(f"b{i}", value=float(cons['b']), key=f"inp_b{i}", label_visibility="collapsed")
        with cl_y: st.markdown("<div style='margin-top: 5px;'><sup>*y</sup></div>", unsafe_allow_html=True)
        with cl3: op_val = st.selectbox(f"op{i}", ("≤", "≥", "="), index=("≤", "≥", "=").index(cons['op']), key=f"inp_op{i}", label_visibility="collapsed")
        with cl4: c_val = st.number_input(f"c{i}", value=float(cons['c']), key=f"inp_c{i}", label_visibility="collapsed")
        with cl5: 
            if st.button("🗑️", key=f"btn_del{i}"):
                st.session_state.constraints.pop(i); st.rerun()
        new_cons.append({'a': a_val, 'b': b_val, 'op': op_val, 'c': c_val})
    
    st.session_state.constraints = new_cons
    if st.button(t_add): st.session_state.constraints.append({'a': 1.0, 'b': 1.0, 'op': '≤', 'c': 10.0}); st.rerun()
    st.markdown("---")
    
    edit_done = st.checkbox(t_edit_done, value=False)
    solve_btn = False
    if edit_done:
        solve_btn = st.button(t_solve, type="primary", use_container_width=True)
    
    st.session_state.lang = st.radio("🌐 Til / Язык", ("Русский", "O'zbekcha"), horizontal=True)

# --- GRAFIK VA YECHIM ---
if solve_btn:
    coeffs = [-c_main1 if obj_type == "max" else c_main1, -c_main2 if obj_type == "max" else c_main2]
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for c in st.session_state.constraints:
        if c['op'] == '≤': A_ub.append([c['a'], c['b']]); b_ub.append(c['c'])
        elif c['op'] == '≥': A_ub.append([-c['a'], -c['b']]); b_ub.append(-c['c'])
        else: A_eq.append([c['a'], c['b']]); b_eq.append(c['c'])
    
    res = linprog(coeffs, A_ub=A_ub or None, b_ub=b_ub or None, A_eq=A_eq or None, b_eq=b_eq or None, bounds=(None, None), method='highs')
    
    fig = go.Figure()
    # Grafik chegarasini aniqlash (o'qlar uzunligi uchun)
    graph_limit = 12 
    x_range = np.linspace(-2, graph_limit, 1000)

    # ODR chizish
    corner_points = []
    lines = st.session_state.constraints
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            try:
                A = np.array([[lines[i]['a'], lines[i]['b']], [lines[j]['a'], lines[j]['b']]])
                B = np.array([lines[i]['c'], lines[j]['c']])
                p = np.linalg.solve(A, B)
                valid = True
                for check in lines:
                    val = check['a']*p[0] + check['b']*p[1]
                    if check['op'] == '≤' and val > check['c'] + 1e-5: valid = False
                    elif check['op'] == '≥' and val < check['c'] - 1e-5: valid = False
                    elif check['op'] == '=' and abs(val - check['c']) > 1e-5: valid = False
                if valid: corner_points.append(p)
            except: continue

    if corner_points:
        pts = np.array(corner_points)
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:,1]-center[1], pts[:,0]-center[0])
        pts = pts[np.argsort(angles)]
        fig.add_trace(go.Scatter(x=pts[:,0], y=pts[:,1], fill="toself", fillcolor='rgba(0, 100, 255, 0.2)', line=dict(color='rgba(255,255,255,0)'), name="ОДР"))
        fig.add_trace(go.Scatter(x=pts[:,0], y=pts[:,1], mode='markers', marker=dict(color='red', size=8), name="Угловые точки"))

    # Cheklov chiziqlarini chizish
    for i, c in enumerate(st.session_state.constraints):
        if abs(c['b']) > 1e-7:
            y_vals = (c['c'] - c['a'] * x_range) / c['b']
            fig.add_trace(go.Scatter(x=x_range, y=y_vals, mode='lines', name=f"L{i+1}", line=dict(width=1)))

    if res.success:
        opt_x, opt_y = res.x
        
        # --- GRAFIK KO'RINISHINI SOZLASH (ASOSIY O'ZGARISHLAR) ---
        fig.update_layout(
            # Oq fon va setkani o'chirish
            plot_bgcolor='white',
            paper_bgcolor='white',
            
            xaxis=dict(
                showgrid=False,      # Setkani o'chirish
                zeroline=True,       # Nol chizig'ini yoqish
                zerolinecolor='black', # Nol chizig'i rangi
                zerolinewidth=2,     # Nol chizig'i qalinligi
                showticklabels=True, # Raqamlarni ko'rsatish
                dtick=1,             # Har bir birlikda raqam
                range=[-1, graph_limit], # Ko'rinish oraliqlari
                ticks="outside",     # Chiziqchalar tashqariga
                tickcolor='black'
            ),
            yaxis=dict(
                showgrid=False,      # Setkani o'chirish
                zeroline=True,       # Nol chizig'ini yoqish
                zerolinecolor='black', # Nol chizig'i rangi
                zerolinewidth=2,     # Nol chizig'i qalinligi
                showticklabels=True, # Raqamlarni ko'rsatish
                dtick=1,             # Har bir birlikda raqam
                range=[-1, graph_limit], # Ko'rinish oraliqlari
                ticks="outside",
                tickcolor='black'
            ),
            legend=dict(x=0.5, y=-0.1, orientation="h", xanchor="center"),
            height=800,
            margin=dict(l=20, r=20, t=20, b=20) # Chetki bo'shliqlar
        )

        # --- O'QLARGA UCHLI STRELKALAR VA X, Y BELGILARINI QO'SHISH ---
        # X o'qi uchun strelka
        fig.add_annotation(
            x=graph_limit - 0.2, y=0,    # Strelka uchi koordinatasi
            ax=-40, ay=0,               # Strelka tanasining uzunligi va yo'nalishi relative piksellarda
            xref="x", yref="y",
            axref="pixel", ayref="pixel",
            showarrow=True,
            arrowhead=2,                # Strelka uchi turi (uchli)
            arrowsize=1.5,              # Strelka uchi kattaligi
            arrowwidth=2,               # Strelka chizig'i qalinligi
            arrowcolor="black"
        )
        # X belgisi
        fig.add_annotation(
            x=graph_limit - 0.5, y=0.5,
            text="<b>X</b>",
            showarrow=False,
            font=dict(size=16, color="black")
        )

        # Y o'qi uchun strelka
        fig.add_annotation(
            x=0, y=graph_limit - 0.2,    # Strelka uchi koordinatasi
            ax=0, ay=40,                # Strelka tanasining uzunligi (pastga qarab ax=-40 edi, ayref pixel bo'lsa tepaga ay=40)
            xref="x", yref="y",
            axref="pixel", ayref="pixel",
            showarrow=True,
            arrowhead=2,                # Strelka uchi turi (uchli)
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="black"
        )
        # Y belgisi
        fig.add_annotation(
            x=0.5, y=graph_limit - 0.5,
            text="<b>Y</b>",
            showarrow=False,
            font=dict(size=16, color="black")
        )

        # Optimal nuqtani belgilash
        fig.add_trace(go.Scatter(x=[opt_x], y=[opt_y], mode='markers+text', text=[f"Opt ({opt_x:.2f}; {opt_y:.2f})"], textposition="top right", marker=dict(color='gold', size=18, symbol='star', line=dict(color='black', width=1)), name="Оптимум"))

        st.plotly_chart(fig, use_container_width=True)

        # --- TAHLIL JADVALI ---
        st.markdown(f"### {t_analysis}")
        shadow_prices = res.get('ineqlin', {}).get('marginals', np.zeros(len(A_ub))) if A_ub else []
        analysis_data = []
        for i, c in enumerate(st.session_state.constraints):
            val_at_opt = c['a'] * opt_x + c['b'] * opt_y
            slack = abs(c['c'] - val_at_opt)
            s_price = abs(shadow_prices[i]) if slack < 1e-5 and i < len(shadow_prices) else 0
            status = "Активно" if slack < 1e-5 else "Запас"
            analysis_data.append({
                "№": f"L{i+1}", 
                "Уравнение": f"{c['a']}x1 + {c['b']}x2 {c['op']} {c['c']}", 
                "Остаток": round(slack, 4), 
                "Статус": status,
                "Shadow Price": round(s_price, 4)
            })
        st.table(pd.DataFrame(analysis_data))

        st.session_state.history.insert(0, {'time': datetime.datetime.now().strftime("%H:%M:%S"), 'c1': c_main1, 'c2': c_main2, 'constraints_text': [f"{c['a']}x1 + ({c['b']})x2 {c['op']} {c['c']}" for c in st.session_state.constraints], 'x': opt_x, 'y': opt_y, 'z': c_main1 * opt_x + c_main2 * opt_y, 'type': obj_type})
        st.success(f"### {('Результат' if st.session_state.lang == 'Русский' else 'Natija')}: X = {opt_x:.2f}, Y = {opt_y:.2f}, Z = {c_main1 * opt_x + c_main2 * opt_y:.2f}")
    else:
        st.error("Yechim topilmadi.")

# --- TARIX ---
if st.session_state.history:
    st.markdown("---")
    st.header(t_hist)
    pdf_file = create_pdf(st.session_state.history)
    st.download_button(t_pdf, data=pdf_file, file_name="lp_report.pdf", mime="application/pdf")
    for h in st.session_state.history:
        st.info(f"🕒 `{h['time']}` | **Z: {h['z']:.2f}** | X: {h['x']:.2f}, Y: {h['y']:.2f}")
