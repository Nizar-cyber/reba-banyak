"""
REBA Ergonomics Analyzer — Streamlit Web App
- Login sederhana (session-based)
- Multi upload foto + batch analisis
- Export 1 Excel untuk semua foto
- REBA Tables (Hignett & McAtamney, 2000)
Tested target: Streamlit Cloud + Python 3.11 + mediapipe==0.10.21
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from io import BytesIO
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="REBA Analyzer",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS
st.markdown(
    """
    <style>
      footer {visibility: hidden;}
      #MainMenu {visibility: hidden;}
      .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# LOGIN (simple, session-based)
# ─────────────────────────────────────────────────────────────────────────────
LOGIN_USER = "Kaizen"
LOGIN_PASS = "Toyota2026"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_ui():
    st.title("🔐 Login REBA System")
    st.caption("Internal Kaizen Tool")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    colA, colB = st.columns([1, 1])
    with colA:
        btn = st.button("Login", type="primary", use_container_width=True)
    with colB:
        st.button("Reset", use_container_width=True, on_click=lambda: None)

    if btn:
        if user == LOGIN_USER and pwd == LOGIN_PASS:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("❌ Username atau Password salah")

if not st.session_state.logged_in:
    login_ui()
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE (cache supaya tidak reload tiap interaksi)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pose_model():
    mp_pose = mp.solutions.pose
    detector = mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
    )
    return mp_pose, detector

mp_pose, pose_detector = load_pose_model()

# ─────────────────────────────────────────────────────────────────────────────
# REBA TABLES (Hignett & McAtamney, 2000)
# ─────────────────────────────────────────────────────────────────────────────
TABLE_A = [
    [[1,2,3,4],[1,2,3,4],[3,3,5,6]],
    [[2,3,4,5],[3,4,5,6],[4,5,6,7]],
    [[2,4,5,6],[4,5,6,7],[5,6,7,8]],
    [[3,5,6,7],[5,6,7,8],[6,7,8,9]],
    [[4,6,7,8],[6,7,8,9],[7,8,9,9]],
]

TABLE_B = [
    [[1,2,2],[1,2,3]],
    [[1,2,3],[2,3,4]],
    [[3,4,5],[4,5,5]],
    [[4,5,5],[5,6,7]],
    [[6,7,8],[7,8,8]],
    [[7,8,8],[8,9,9]],
]

TABLE_C = [
    [ 1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    [ 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [ 2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
    [ 3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [ 4, 4, 4, 5, 6, 7, 8, 8, 9, 9,10,10],
    [ 6, 6, 6, 7, 8, 8, 9, 9,10,10,11,11],
    [ 7, 7, 7, 8, 9, 9,10,10,11,11,11,12],
    [ 8, 8, 8, 9,10,10,11,11,12,12,13,13],
    [ 9, 9, 9,10,10,11,11,12,13,13,13,13],
    [10,10,10,11,11,12,12,13,13,14,14,14],
    [11,11,11,11,12,12,13,13,14,14,15,15],
    [12,12,12,12,12,13,13,14,14,15,15,15],
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS: risk, scoring, angles
# ─────────────────────────────────────────────────────────────────────────────
def risk_cat(s):
    if s == 1:
        return "Dapat Diabaikan", "🟢", "Tidak perlu tindakan", "#27AE60"
    elif s <= 3:
        return "Rendah", "🟡", "Perubahan mungkin diperlukan", "#2ECC71"
    elif s <= 7:
        return "Sedang", "🟠", "Investigasi & perubahan segera", "#F39C12"
    elif s <= 10:
        return "Tinggi", "🔴", "Investigasi & implementasi segera", "#E67E22"
    else:
        return "Sangat Tinggi", "🚨", "Implementasi perubahan SEGERA!", "#E74C3C"

def calc_angle(a, b, c):
    a = np.array(a, float); b = np.array(b, float); c = np.array(c, float)
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
    return round(np.degrees(np.arccos(np.clip(cos, -1, 1))), 1)

def trunk_flexion(shoulder, hip):
    v = np.array([hip[0]-shoulder[0], hip[1]-shoulder[1]], float)
    cos = np.dot(v, [0,1]) / (np.linalg.norm(v) + 1e-10)
    return round(np.degrees(np.arccos(np.clip(cos, -1, 1))), 1)

def neck_flexion(ear, shoulder, hip):
    nv = np.array([ear[0]-shoulder[0], ear[1]-shoulder[1]], float)
    tv = np.array([shoulder[0]-hip[0], shoulder[1]-hip[1]], float)
    cos = np.dot(nv, tv) / (np.linalg.norm(nv)*np.linalg.norm(tv) + 1e-10)
    return round(np.degrees(np.arccos(np.clip(cos, -1, 1))), 1)

def upper_arm_angle(shoulder, elbow, hip):
    av = np.array([elbow[0]-shoulder[0], elbow[1]-shoulder[1]], float)
    tv = np.array([hip[0]-shoulder[0], hip[1]-shoulder[1]], float)
    cos = np.dot(av, tv) / (np.linalg.norm(av)*np.linalg.norm(tv) + 1e-10)
    return round(np.degrees(np.arccos(np.clip(cos, -1, 1))), 1)

def score_neck(a): return 1 if a <= 20 else 2

def score_trunk(a):
    if a < 5: return 1
    elif a <= 20: return 2
    elif a <= 60: return 3
    else: return 4

def score_legs(knee_angle):
    b = 1
    f = 180 - knee_angle
    if 30 <= f <= 60: b += 1
    elif f > 60: b += 2
    return min(b, 4)

def score_ua(a):
    if a <= 20: return 1
    elif a <= 45: return 2
    elif a <= 90: return 3
    else: return 4

def score_la(a):
    f = 180 - a
    return 1 if 60 <= f <= 100 else 2

def score_wrist(d): return 1 if d <= 15 else 2

def tbl_a(t,n,l):
    return TABLE_A[max(1,min(5,t))-1][max(1,min(3,n))-1][max(1,min(4,l))-1]

def tbl_b(u,l,w):
    return TABLE_B[max(1,min(6,u))-1][max(1,min(2,l))-1][max(1,min(3,w))-1]

def tbl_c(a,b):
    return TABLE_C[max(1,min(12,a))-1][max(1,min(12,b))-1]

def force_score(kg):
    try: v = float(kg)
    except: v = 0
    return 0 if v < 5 else (1 if v <= 10 else 2)

# ─────────────────────────────────────────────────────────────────────────────
# DRAW (simple overlay) — biar ringan untuk batch
# ─────────────────────────────────────────────────────────────────────────────
def draw_overlay(img_bgr, pts, reba_score, category, color_hex):
    out = img_bgr.copy()
    color_hex = color_hex.lstrip("#")
    bgr = (int(color_hex[4:6],16), int(color_hex[2:4],16), int(color_hex[0:2],16))

    pairs = [
        ("ear","shoulder"),
        ("shoulder","elbow"),
        ("elbow","wrist"),
        ("shoulder","hip"),
        ("hip","knee"),
        ("knee","ankle"),
    ]
    for a,b in pairs:
        if a in pts and b in pts:
            p1 = tuple(np.array(pts[a], int))
            p2 = tuple(np.array(pts[b], int))
            cv2.line(out, p1, p2, (0,200,120), 2, cv2.LINE_AA)

    for k,v in pts.items():
        p = tuple(np.array(v, int))
        cv2.circle(out, p, 4, (50,220,255), -1, cv2.LINE_AA)
        cv2.circle(out, p, 5, (255,255,255), 1, cv2.LINE_AA)

    h, w = out.shape[:2]
    cv2.rectangle(out, (0,0), (w,48), bgr, -1)
    cv2.putText(out, f"REBA: {reba_score} | {category}", (12,32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2, cv2.LINE_AA)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# ANALYZE 1 IMAGE
# ─────────────────────────────────────────────────────────────────────────────
def analyze_pose(image_bgr, beban, aktivitas, activity_score):
    results = pose_detector.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None, None

    h, w, _ = image_bgr.shape
    lm = results.pose_landmarks.landmark
    LP = mp_pose.PoseLandmark

    def gp(idx): return [lm[idx].x*w, lm[idx].y*h]

    # sama seperti versi kamu: pakai sisi kiri
    ear      = gp(LP.LEFT_EAR.value)
    nose     = gp(LP.NOSE.value)
    shoulder = gp(LP.LEFT_SHOULDER.value)
    elbow    = gp(LP.LEFT_ELBOW.value)
    wrist    = gp(LP.LEFT_WRIST.value)
    hip      = gp(LP.LEFT_HIP.value)
    knee     = gp(LP.LEFT_KNEE.value)
    ankle    = gp(LP.LEFT_ANKLE.value)

    ta  = trunk_flexion(shoulder, hip)
    na  = neck_flexion(ear, shoulder, hip)
    uaa = upper_arm_angle(shoulder, elbow, hip)
    laa = calc_angle(shoulder, elbow, wrist)
    ka  = calc_angle(hip, knee, ankle)
    wr  = calc_angle(elbow, wrist, shoulder)
    wd  = round(abs(180 - wr), 1)

    ns  = score_neck(na)
    ts  = score_trunk(ta)
    ls  = score_legs(ka)
    us  = score_ua(uaa)
    las = score_la(laa)
    ws  = score_wrist(wd)

    tA = tbl_a(ts, ns, ls)
    tB = tbl_b(us, las, ws)
    fs = force_score(beban)
    sA = tA + fs
    sB = tB + 1  # coupling = 1 (Fair) seperti app kamu
    sC = tbl_c(sA, sB)

    final = max(1, min(15, sC + int(activity_score)))
    kat, icon, tindakan, warna = risk_cat(final)

    pts = {
        "nose": nose, "ear": ear, "shoulder": shoulder, "elbow": elbow,
        "wrist": wrist, "hip": hip, "knee": knee, "ankle": ankle
    }
    annotated = draw_overlay(image_bgr, pts, final, kat, warna)

    result = dict(
        Timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        Beban_kg=float(beban),
        Aktivitas=aktivitas,
        Activity_Score=int(activity_score),

        Sudut_Leher=na,
        Sudut_Tubuh=ta,
        Sudut_LenganAtas=uaa,
        Sudut_LenganBawah=laa,
        Deviasi_Pergelangan=wd,
        Sudut_Lutut=ka,

        Skor_Leher=ns,
        Skor_Tubuh=ts,
        Skor_Kaki=ls,
        Skor_LenganAtas=us,
        Skor_LenganBawah=las,
        Skor_Pergelangan=ws,

        Table_A=tA,
        Table_B=tB,
        Force_Score=fs,
        Score_A=sA,
        Score_B=sB,
        Score_C=sC,

        REBA_Final=final,
        Kategori=kat,
        Tindakan=tindakan,
        Warna=warna,
        Icon=icon,
    )
    return annotated, result

# ─────────────────────────────────────────────────────────────────────────────
# EXCEL EXPORT (BATCH)
# ─────────────────────────────────────────────────────────────────────────────
def build_excel_batch(results: list, nama_laporan: str) -> bytes:
    """
    results: list of dict (each dict already includes 'Nama File')
    """
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    df = pd.DataFrame(results)

    # rapikan urutan kolom (opsional)
    prefer = [
        "Nama File","Timestamp","Aktivitas","Beban_kg","Activity_Score",
        "REBA_Final","Kategori","Tindakan",
        "Sudut_Leher","Sudut_Tubuh","Sudut_LenganAtas","Sudut_LenganBawah","Deviasi_Pergelangan","Sudut_Lutut",
        "Skor_Leher","Skor_Tubuh","Skor_Kaki","Skor_LenganAtas","Skor_LenganBawah","Skor_Pergelangan",
        "Table_A","Force_Score","Score_A","Table_B","Score_B","Score_C",
    ]
    cols = [c for c in prefer if c in df.columns] + [c for c in df.columns if c not in prefer]
    df = df[cols]

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as wr:
        df.to_excel(wr, sheet_name="Hasil REBA", index=False, startrow=2)
        ws = wr.sheets["Hasil REBA"]

        # Header besar
        ws.merge_cells("A1:H1")
        ws["A1"] = f"Laporan REBA (Batch) — {nama_laporan} — {datetime.now().strftime('%d %B %Y')}"
        ws["A1"].font = Font(bold=True, size=13, color="FFFFFF")
        ws["A1"].fill = PatternFill("solid", fgColor="1A3A6A")
        ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 28

        # Header table (row=3)
        for cell in ws[3]:
            cell.font = Font(bold=True, color="FFFFFF", size=10)
            cell.fill = PatternFill("solid", fgColor="2E4A7A")
            cell.alignment = Alignment(horizontal="center")

        # Border + autosize
        thin = Border(
            left=Side(style="thin"), right=Side(style="thin"),
            top=Side(style="thin"), bottom=Side(style="thin")
        )
        max_row = 3 + len(df)
        max_col = len(df.columns)

        for r in range(3, max_row+1):
            for c in range(1, max_col+1):
                ws.cell(r, c).border = thin

        for col in ws.columns:
            ml = max((len(str(c.value)) if c.value else 0) for c in col)
            ws.column_dimensions[get_column_letter(col[0].column)].width = max(ml + 3, 12)

        # Highlight kolom REBA_Final
        if "REBA_Final" in df.columns:
            reba_idx = df.columns.get_loc("REBA_Final") + 1  # 1-based
            cmap = {(1,1):"27AE60",(2,3):"2ECC71",(4,7):"F39C12",(8,10):"E67E22",(11,15):"E74C3C"}
            for rr in range(4, 4+len(df)):
                val = ws.cell(rr, reba_idx).value
                try:
                    score = int(val)
                except:
                    continue
                cc = "FFFFFF"
                for (lo,hi),hx in cmap.items():
                    if lo <= score <= hi:
                        cc = hx
                        break
                ws.cell(rr, reba_idx).fill = PatternFill("solid", fgColor=cc)
                ws.cell(rr, reba_idx).font = Font(bold=True, color="000000")

    return buf.getvalue()

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE for batch
# ─────────────────────────────────────────────────────────────────────────────
for k, d in [
    ("batch_results", []),
    ("batch_images", []),
    ("batch_failed", []),
    ("analyzed", False),
    ("selected_file", None),
]:
    if k not in st.session_state:
        st.session_state[k] = d

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR UI
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🦴 REBA Analyzer")
    st.caption("Ergonomics Risk Assessment · Hignett & McAtamney, 2000")
    st.divider()

    st.subheader("📁 Upload Foto (Multi)")
    uploaded_files = st.file_uploader(
        "Pilih foto postur kerja (boleh banyak sekaligus)",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        help="Pastikan seluruh tubuh (kepala hingga kaki) terlihat jelas",
    )

    st.divider()
    st.subheader("⚖️ Beban")
    beban = st.number_input("Berat yang diangkat (kg)", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    fs_val = force_score(beban)
    st.caption(["🟢 Ringan < 5 kg (Force Score 0)", "🟡 Sedang 5–10 kg (Force Score 1)", "🔴 Berat > 10 kg (Force Score 2)"][fs_val])

    st.divider()
    st.subheader("🏭 Aktivitas")
    AKTIVITAS_LIST = [
        "Pengangkatan Manual","Menurunkan Beban","Mendorong / Menarik",
        "Perakitan (Assembly)","Pengelasan","Pengepakan / Packaging",
        "Inspeksi Visual","Pengoperasian Mesin","Pekerjaan Kantor / Duduk",
        "Lainnya..."
    ]
    aktivitas_sel = st.selectbox("Pilih jenis aktivitas", AKTIVITAS_LIST)
    if aktivitas_sel == "Lainnya...":
        aktivitas = st.text_input("Nama aktivitas (isi manual)")
    else:
        aktivitas = aktivitas_sel

    st.divider()
    st.subheader("🔢 Activity Score")
    st.caption("Tiap kondisi yang berlaku = +1 ke Score C")
    act1 = st.checkbox("🧍 Bagian tubuh statis > 1 menit")
    act2 = st.checkbox("🔄 Gerakan berulang > 4× per menit")
    act3 = st.checkbox("⚡ Postur berubah cepat / tidak stabil")
    activity_score = int(act1) + int(act2) + int(act3)
    st.info(f"Activity Score: **+{activity_score}**", icon="📊")

    st.divider()
    analyze_btn = st.button(
        "🔍 Analisis REBA (Batch)",
        type="primary",
        use_container_width=True,
        disabled=(not uploaded_files),
    )

    # Logout
    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
col_img, col_res = st.columns([3, 2], gap="large")

with col_img:
    st.subheader("Visualisasi Skeleton (Batch)")

    if st.session_state.analyzed and st.session_state.batch_images:
        file_names = [x[0] for x in st.session_state.batch_images]
        if st.session_state.selected_file not in file_names:
            st.session_state.selected_file = file_names[0]

        pick = st.selectbox("Pilih file untuk ditampilkan", file_names, index=file_names.index(st.session_state.selected_file))
        st.session_state.selected_file = pick

        img = dict(st.session_state.batch_images).get(pick)
        if img is not None:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"Skeleton — {pick}")

        # tampilkan daftar error (jika ada)
        if st.session_state.batch_failed:
            st.warning("Pose tidak terdeteksi pada file: " + ", ".join(st.session_state.batch_failed))
    else:
        if uploaded_files:
            st.image(Image.open(uploaded_files[0]).convert("RGB"), use_container_width=True,
                     caption="Pratinjau file pertama — klik Analisis untuk memproses semua")
        else:
            st.info("⬅️ Upload foto postur kerja (boleh banyak), lalu klik **Analisis REBA (Batch)**.")

with col_res:
    st.subheader("Hasil Analisis (Batch)")

    if not (st.session_state.analyzed and st.session_state.batch_results):
        st.info("Hasil akan tampil setelah foto dianalisis.", icon="📋")
    else:
        df = pd.DataFrame(st.session_state.batch_results)

        # Summary table utama
        show_cols = [c for c in ["Nama File","REBA_Final","Kategori","Tindakan","Beban_kg","Aktivitas","Timestamp"] if c in df.columns]
        st.markdown("**📊 Ringkasan Batch**")
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

        # Pilih satu file untuk detail (pakai Nama File)
        st.divider()
        st.markdown("**🔎 Detail per File**")
        file_list = df["Nama File"].tolist() if "Nama File" in df.columns else []
        if file_list:
            if st.session_state.selected_file not in file_list:
                st.session_state.selected_file = file_list[0]
            pick2 = st.selectbox("Pilih file untuk detail", file_list, index=file_list.index(st.session_state.selected_file))
            st.session_state.selected_file = pick2
            r = df[df["Nama File"] == pick2].iloc[0].to_dict()
        else:
            r = df.iloc[0].to_dict()

        reba = int(r["REBA_Final"])
        kat, icon, tindakan, warna = risk_cat(reba)

        st.metric(
            label=f"{icon} Skor REBA Final",
            value=reba,
            delta=f"Risiko {kat} · {tindakan}",
            delta_color="off",
        )

        # Sudut & skor segmen (detail)
        st.markdown("**📐 Sudut Terukur**")
        st.dataframe(
            pd.DataFrame({
                "Segmen": ["Leher","Batang Tubuh","Lengan Atas","Lengan Bawah","Pergelangan","Lutut"],
                "Sudut (°)": [r["Sudut_Leher"], r["Sudut_Tubuh"], r["Sudut_LenganAtas"], r["Sudut_LenganBawah"], r["Deviasi_Pergelangan"], r["Sudut_Lutut"]],
                "Skor": [r["Skor_Leher"], r["Skor_Tubuh"], r["Skor_LenganAtas"], r["Skor_LenganBawah"], r["Skor_Pergelangan"], r["Skor_Kaki"]],
            }),
            hide_index=True, use_container_width=True,
        )

        st.markdown("**🔢 Alur Perhitungan**")
        st.dataframe(
            pd.DataFrame({
                "Parameter": ["Table A","Force Score","Score A","Table B","Coupling","Score B","Score C","Activity Score","REBA FINAL"],
                "Nilai": [r["Table_A"], r["Force_Score"], r["Score_A"], r["Table_B"], "1 (Fair)", r["Score_B"], r["Score_C"], f"+{r['Activity_Score']}", r["REBA_Final"]],
            }),
            hide_index=True, use_container_width=True,
        )

        st.caption(f"🕐 {r['Timestamp']} · ⚖️ {r['Beban_kg']} kg · 🏭 {r['Aktivitas']} · 📄 {r.get('Nama File','-')}")

        # Export Excel batch
        st.divider()
        st.markdown("**📤 Export Excel (Semua Foto)**")
        nama_laporan = st.text_input("Nama laporan batch", value=r.get("Aktivitas","Batch REBA"), key="nama_laporan_batch")
        if nama_laporan.strip():
            st.download_button(
                label="⬇️ Download Excel (Batch)",
                data=build_excel_batch(st.session_state.batch_results, nama_laporan.strip()),
                file_name=f"REBA_Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# RUN ANALYSIS (BATCH)
# ─────────────────────────────────────────────────────────────────────────────
if analyze_btn and uploaded_files:
    batch_results = []
    batch_images = []
    batch_failed = []

    with st.spinner("Mendeteksi pose & menghitung skor REBA (batch)..."):
        for f in uploaded_files:
            img_bgr = cv2.cvtColor(np.array(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR)
            annotated, result = analyze_pose(img_bgr, beban, aktivitas, activity_score)

            if result is None:
                batch_failed.append(f.name)
                continue

            result["Nama File"] = f.name
            batch_results.append(result)
            batch_images.append((f.name, annotated))

    if not batch_results:
        st.error(
            "**Tidak ada pose yang berhasil terdeteksi dari semua foto.**\n\n"
            "Pastikan seluruh tubuh (kepala–kaki) terlihat jelas, pencahayaan cukup, dan tidak terhalang.",
            icon="⚠️",
        )
        st.session_state.analyzed = False
        st.session_state.batch_results = []
        st.session_state.batch_images = []
        st.session_state.batch_failed = batch_failed
    else:
        st.session_state.batch_results = batch_results
        st.session_state.batch_images = batch_images
        st.session_state.batch_failed = batch_failed
        st.session_state.analyzed = True
        st.session_state.selected_file = batch_images[0][0]
        st.rerun()
