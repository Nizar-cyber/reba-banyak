"""
REBA Ergonomics Analyzer — Web App
Deploy: Streamlit Community Cloud (GitHub)
GUI: Native Streamlit — bersih, ringan, kompatibel penuh
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
# PAGE CONFIG  (harus baris pertama Streamlit)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="REBA Analyzer",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS — hanya sembunyikan footer & rapikan padding
st.markdown("""
<style>
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
.block-container {padding-top: 1.5rem;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE  (cache agar tidak reload tiap interaksi)
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

# =============================================
# REBA LOOKUP TABLES  (Hignett & McAtamney, 2000)
# =============================================
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

# =============================================
# HELPERS
# =============================================
def risk_cat(s):
    if s == 1:    return "Dapat Diabaikan", "🟢", "Tidak perlu tindakan"
    elif s <= 3:  return "Rendah",          "🟡", "Perubahan mungkin diperlukan"
    elif s <= 7:  return "Sedang",          "🟠", "Investigasi & perubahan segera"
    elif s <= 10: return "Tinggi",          "🔴", "Investigasi & implementasi segera"
    else:         return "Sangat Tinggi",   "🚨", "Implementasi perubahan SEGERA!"

def risk_color_bgr(s):
    if s <= 1:    return (39, 174, 39)
    elif s <= 3:  return (50, 205, 50)
    elif s <= 7:  return (0, 165, 255)
    elif s <= 10: return (0, 100, 230)
    else:         return (0, 50, 220)

def seg_color(s):
    if s <= 1:   return (50, 200, 50)
    elif s == 2: return (0, 200, 200)
    elif s == 3: return (0, 165, 255)
    else:        return (0, 80, 220)

# =============================================
# KALKULASI SUDUT
# =============================================
def calc_angle(a, b, c):
    a=np.array(a,float); b=np.array(b,float); c=np.array(c,float)
    ba=a-b; bc=c-b
    cos=np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-10)
    return round(np.degrees(np.arccos(np.clip(cos,-1,1))),1)

def trunk_flexion(shoulder, hip):
    v=np.array([hip[0]-shoulder[0], hip[1]-shoulder[1]])
    cos=np.dot(v,[0,1])/(np.linalg.norm(v)+1e-10)
    return round(np.degrees(np.arccos(np.clip(cos,-1,1))),1)

def neck_flexion(ear, shoulder, hip):
    nv=np.array([ear[0]-shoulder[0], ear[1]-shoulder[1]])
    tv=np.array([shoulder[0]-hip[0], shoulder[1]-hip[1]])
    cos=np.dot(nv,tv)/(np.linalg.norm(nv)*np.linalg.norm(tv)+1e-10)
    return round(np.degrees(np.arccos(np.clip(cos,-1,1))),1)

def upper_arm_angle(shoulder, elbow, hip):
    av=np.array([elbow[0]-shoulder[0], elbow[1]-shoulder[1]])
    tv=np.array([hip[0]-shoulder[0],   hip[1]-shoulder[1]])
    cos=np.dot(av,tv)/(np.linalg.norm(av)*np.linalg.norm(tv)+1e-10)
    return round(np.degrees(np.arccos(np.clip(cos,-1,1))),1)

# =============================================
# REBA SCORING
# =============================================
def score_neck(a):  return 1 if a<=20 else 2
def score_trunk(a):
    if a<5:     return 1
    elif a<=20: return 2
    elif a<=60: return 3
    else:       return 4
def score_legs(ka):
    b=1; f=180-ka
    if 30<=f<=60: b+=1
    elif f>60:    b+=2
    return min(b,4)
def score_ua(a):
    if a<=20:   return 1
    elif a<=45: return 2
    elif a<=90: return 3
    else:       return 4
def score_la(a):    return 1 if 60<=(180-a)<=100 else 2
def score_wrist(d): return 1 if d<=15 else 2

def tbl_a(t,n,l): return TABLE_A[max(1,min(5,t))-1][max(1,min(3,n))-1][max(1,min(4,l))-1]
def tbl_b(u,l,w): return TABLE_B[max(1,min(6,u))-1][max(1,min(2,l))-1][max(1,min(3,w))-1]
def tbl_c(a,b):   return TABLE_C[max(1,min(12,a))-1][max(1,min(12,b))-1]

def force_score(kg):
    try: v=float(kg)
    except: v=0
    return 0 if v<5 else (1 if v<=10 else 2)

# =============================================
# SKELETON DRAWING
# =============================================
def get_scale_params(w, h):
    s = min(w, h)/600.0
    return dict(
        bone=max(1,int(2*s)), joint=max(3,int(4*s)),
        arc=max(12,int(18*s)), font=max(0.28,min(0.45,0.35*s)),
        offset=max(35,int(45*s)), head=0.7,
        banner=max(0.38,min(0.55,0.45*s)),
        legend=max(0.28,min(0.38,0.32*s)),
    )

def draw_label(img, text, pos, color, scale=0.35):
    x,y=int(pos[0]),int(pos[1]); f=cv2.FONT_HERSHEY_SIMPLEX
    (tw,th),bl=cv2.getTextSize(text,f,scale,1); p=2
    cv2.rectangle(img,(x-p,y-th-p),(x+tw+p,y+bl+p),(8,8,8),-1)
    cv2.rectangle(img,(x-p,y-th-p),(x+tw+p,y+bl+p),color,1)
    cv2.putText(img,text,(x,y),f,scale,color,1,cv2.LINE_AA)

def draw_bone(img, p1, p2, color, thick=2):
    cv2.line(img,tuple(np.array(p1,int)),tuple(np.array(p2,int)),color,thick,cv2.LINE_AA)

def draw_joint(img, pt, color, r=4):
    cv2.circle(img,tuple(np.array(pt,int)),r,color,-1,cv2.LINE_AA)
    cv2.circle(img,tuple(np.array(pt,int)),r+1,(255,255,255),1,cv2.LINE_AA)

def draw_arc(img, vertex, p1, p2, color, radius=18):
    v=np.array(vertex,int)
    v1=np.array(p1,float)-v; v2=np.array(p2,float)-v
    a1=np.degrees(np.arctan2(v1[1],v1[0])); a2=np.degrees(np.arctan2(v2[1],v2[0]))
    if a1>a2: a1,a2=a2,a1
    if a2-a1>180: a1,a2=a2,a1+360
    cv2.ellipse(img,tuple(v),(radius,radius),0,a1,a2,color,1,cv2.LINE_AA)

def offset_from(vertex, ref, dist=45):
    v=np.array(vertex,float); r=np.array(ref,float); d=v-r; nm=np.linalg.norm(d)
    if nm<1: return v+np.array([dist,0])
    return v+(d/nm)*dist

def draw_skeleton(image, lmd, scores, reba_final):
    img=image.copy(); h,w=img.shape[:2]; sp=get_scale_params(w,h)
    ear=lmd['ear']; nose=lmd['nose']; shoulder=lmd['shoulder']
    elbow=lmd['elbow']; wrist=lmd['wrist']; hip=lmd['hip']
    knee=lmd['knee']; ankle=lmd['ankle']
    neck_s,trunk_s,leg_s,ua_s,la_s,w_s=scores
    main_c=risk_color_bgr(reba_final); bt=sp['bone']

    draw_bone(img,ear,shoulder,  seg_color(neck_s),  bt)
    draw_bone(img,shoulder,hip,  seg_color(trunk_s), bt+1)
    draw_bone(img,shoulder,elbow,seg_color(ua_s),    bt)
    draw_bone(img,elbow,wrist,   seg_color(la_s),    bt)
    draw_bone(img,hip,knee,      seg_color(leg_s),   bt+1)
    draw_bone(img,knee,ankle,    seg_color(leg_s),   bt+1)
    draw_bone(img,nose,ear,      (160,160,160),       max(1,bt-1))

    head_r=max(8,int(abs(nose[1]-ear[1])*sp['head']))
    cv2.circle(img,tuple(np.array(nose,int)),head_r,(200,200,200),1,cv2.LINE_AA)

    jr=sp['joint']
    for pt,c in [(ear,seg_color(neck_s)),(shoulder,(220,220,100)),
                 (elbow,seg_color(ua_s)),(wrist,seg_color(la_s)),
                 (hip,(220,220,100)),(knee,seg_color(leg_s)),(ankle,seg_color(leg_s))]:
        draw_joint(img,pt,c,jr)

    ar=sp['arc']
    draw_arc(img,shoulder,ear,hip,    seg_color(neck_s),  ar)
    draw_arc(img,hip,shoulder,knee,   seg_color(trunk_s), int(ar*1.2))
    draw_arc(img,shoulder,hip,elbow,  seg_color(ua_s),    ar)
    draw_arc(img,elbow,shoulder,wrist,seg_color(la_s),    int(ar*0.9))
    draw_arc(img,knee,hip,ankle,      seg_color(leg_s),   ar)

    sc=sp['font']; off=sp['offset']
    na=lmd['neck_ang']; ta=lmd['trunk_ang']; uaa=lmd['ua_ang']
    laa=lmd['la_ang']; wd=lmd['wrist_dev']; ka=lmd['knee_ang']

    def place(vertex, ref, text, color):
        pos=offset_from(vertex,ref,off)
        pos[0]=max(5,min(w-130,pos[0])); pos[1]=max(14,min(h-5,pos[1]))
        draw_label(img,text,pos,color,sc)

    place(shoulder,ear,    f"Leher:{na:.1f}d[S{neck_s}]",  seg_color(neck_s))
    place(hip,knee,        f"Tubuh:{ta:.1f}d[S{trunk_s}]", seg_color(trunk_s))
    place(elbow,wrist,     f"L.Atas:{uaa:.1f}d[S{ua_s}]",  seg_color(ua_s))
    place(wrist,elbow,     f"L.Bwh:{laa:.1f}d[S{la_s}]",   seg_color(la_s))
    place(knee,ankle,      f"Lutut:{ka:.1f}d[S{leg_s}]",   seg_color(leg_s))
    place(wrist,shoulder,  f"Prglgn:{wd:.1f}d[S{w_s}]",    seg_color(w_s))

    kat,_,_=risk_cat(reba_final); bfont=sp['banner']
    banner=f"  REBA: {reba_final}  |  {kat}  "
    (bw2,bh),_=cv2.getTextSize(banner,cv2.FONT_HERSHEY_SIMPLEX,bfont,1)
    bx=(w-bw2)//2; pad_v=max(3,int(6*bfont))
    cv2.rectangle(img,(bx-8,3),(bx+bw2+8,bh+pad_v*2),(12,12,12),-1)
    cv2.rectangle(img,(bx-8,3),(bx+bw2+8,bh+pad_v*2),main_c,1)
    cv2.putText(img,banner,(bx,bh+pad_v),cv2.FONT_HERSHEY_SIMPLEX,bfont,main_c,1,cv2.LINE_AA)

    lfont=sp['legend']
    legend=[((50,200,50),"S1"),((0,200,200),"S2"),((0,165,255),"S3"),((0,80,220),"S4+")]
    lx=6; ly=h-6
    for bgr,lbl in reversed(legend):
        (tw2,th2),_=cv2.getTextSize(f" {lbl}",cv2.FONT_HERSHEY_SIMPLEX,lfont,1)
        cv2.rectangle(img,(lx-2,ly-th2-3),(lx+tw2+14,ly+3),(12,12,12),-1)
        cv2.circle(img,(lx+4,ly-th2//2),3,bgr,-1)
        cv2.putText(img,f"  {lbl}",(lx,ly),cv2.FONT_HERSHEY_SIMPLEX,lfont,bgr,1,cv2.LINE_AA)
        ly-=th2+7
    return img

# =============================================
# ANALISIS POSE
# =============================================
def analyze_pose(image_bgr, beban, aktivitas, activity_score):
    results = pose_detector.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None, None

    h,w,_=image_bgr.shape; lm=results.pose_landmarks.landmark; LP=mp_pose.PoseLandmark
    def gp(idx): return [lm[idx].x*w, lm[idx].y*h]

    ear=gp(LP.LEFT_EAR.value); nose=gp(LP.NOSE.value)
    shoulder=gp(LP.LEFT_SHOULDER.value); elbow=gp(LP.LEFT_ELBOW.value)
    wrist=gp(LP.LEFT_WRIST.value); hip=gp(LP.LEFT_HIP.value)
    knee=gp(LP.LEFT_KNEE.value); ankle=gp(LP.LEFT_ANKLE.value)

    ta=trunk_flexion(shoulder,hip); na=neck_flexion(ear,shoulder,hip)
    uaa=upper_arm_angle(shoulder,elbow,hip); laa=calc_angle(shoulder,elbow,wrist)
    ka=calc_angle(hip,knee,ankle); wr=calc_angle(elbow,wrist,shoulder)
    wd=round(abs(180-wr),1)

    ns=score_neck(na); ts=score_trunk(ta); ls=score_legs(ka)
    us=score_ua(uaa); las=score_la(laa); ws=score_wrist(wd)

    tA=tbl_a(ts,ns,ls); tB=tbl_b(us,las,ws)
    fs=force_score(beban); sA=tA+fs; sB=tB+1
    sC=tbl_c(sA,sB); final=max(1,min(15,sC+activity_score))
    kat,icon,tindakan=risk_cat(final)

    lmd=dict(
        ear=ear, nose=nose, shoulder=shoulder, elbow=elbow, wrist=wrist,
        hip=hip, knee=knee, ankle=ankle, neck_ang=na, trunk_ang=ta,
        ua_ang=uaa, la_ang=laa, wrist_dev=wd, knee_ang=ka,
    )
    annotated=draw_skeleton(image_bgr, lmd, (ns,ts,ls,us,las,ws), final)

    result=dict(
        Timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        Beban_kg=beban, Aktivitas=aktivitas,
        Sudut_Leher=na, Sudut_Tubuh=ta, Sudut_LenganAtas=uaa,
        Sudut_LenganBawah=laa, Deviasi_Pergelangan=wd, Sudut_Lutut=ka,
        Skor_Leher=ns, Skor_Tubuh=ts, Skor_Kaki=ls,
        Skor_LenganAtas=us, Skor_LenganBawah=las, Skor_Pergelangan=ws,
        Table_A=tA, Table_B=tB, Force_Score=fs,
        Score_A=sA, Score_B=sB, Score_C=sC,
        Activity_Score=activity_score,
        REBA_Final=final, Kategori=kat, Tindakan=tindakan,
    )
    return annotated, result

# =============================================
# EXPORT EXCEL
# =============================================
def build_excel(result: dict, nama_laporan: str) -> bytes:
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    label_map = {
        "Timestamp":"Timestamp", "Beban_kg":"Beban (kg)", "Aktivitas":"Aktivitas",
        "Sudut_Leher":"Sudut Leher (°)", "Sudut_Tubuh":"Sudut Batang Tubuh (°)",
        "Sudut_LenganAtas":"Sudut Lengan Atas (°)", "Sudut_LenganBawah":"Sudut Lengan Bawah (°)",
        "Deviasi_Pergelangan":"Deviasi Pergelangan (°)", "Sudut_Lutut":"Sudut Lutut (°)",
        "Skor_Leher":"Skor Leher", "Skor_Tubuh":"Skor Batang Tubuh", "Skor_Kaki":"Skor Kaki",
        "Skor_LenganAtas":"Skor Lengan Atas", "Skor_LenganBawah":"Skor Lengan Bawah",
        "Skor_Pergelangan":"Skor Pergelangan", "Table_A":"Table A", "Table_B":"Table B",
        "Force_Score":"Force Score", "Score_A":"Score A", "Score_B":"Score B",
        "Score_C":"Score C", "Activity_Score":"Activity Score",
        "REBA_Final":"Skor REBA Final", "Kategori":"Kategori Risiko", "Tindakan":"Tindakan",
    }
    row = {label_map[k]: v for k,v in result.items() if k in label_map}
    row["Nama Laporan"] = nama_laporan
    df = pd.DataFrame([row])

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as wr:
        df.to_excel(wr, sheet_name="Hasil REBA", index=False, startrow=2)
        ws = wr.sheets["Hasil REBA"]

        ws.merge_cells("A1:E1")
        ws["A1"] = f"Laporan REBA — {nama_laporan} — {datetime.now().strftime('%d %B %Y')}"
        ws["A1"].font      = Font(bold=True, size=13, color="FFFFFF")
        ws["A1"].fill      = PatternFill("solid", fgColor="1A3A6A")
        ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 28

        for cell in ws[3]:
            cell.font      = Font(bold=True, color="FFFFFF", size=10)
            cell.fill      = PatternFill("solid", fgColor="2E4A7A")
            cell.alignment = Alignment(horizontal="center")

        for i, cell in enumerate(ws[3], 1):
            if cell.value == "Skor REBA Final":
                score = result["REBA_Final"]
                cmap  = {(1,1):"27AE60",(2,3):"2ECC71",(4,7):"F39C12",
                         (8,10):"E67E22",(11,15):"E74C3C"}
                cc = "FFFFFF"
                for (lo,hi),hx in cmap.items():
                    if lo<=score<=hi: cc=hx; break
                dc = ws.cell(row=4, column=i)
                dc.fill = PatternFill("solid", fgColor=cc)
                dc.font = Font(bold=True, color="000000")
                break

        thin = Border(
            left=Side(style="thin"), right=Side(style="thin"),
            top=Side(style="thin"),  bottom=Side(style="thin"),
        )
        for row_cells in ws.iter_rows(min_row=3, max_row=4):
            for cell in row_cells: cell.border = thin

        for col in ws.columns:
            ml = max((len(str(c.value)) if c.value else 0) for c in col)
            ws.column_dimensions[get_column_letter(col[0].column)].width = max(ml+3, 12)

    return buf.getvalue()


def build_excel_multi(results: list, nama_laporan: str) -> bytes:
    """Export semua hasil analisis ke satu file Excel — tiap baris = satu foto."""
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    label_map = {
        "Nama_File":"Nama File",
        "Timestamp":"Timestamp", "Beban_kg":"Beban (kg)", "Aktivitas":"Aktivitas",
        "Sudut_Leher":"Sudut Leher (°)", "Sudut_Tubuh":"Sudut Batang Tubuh (°)",
        "Sudut_LenganAtas":"Sudut Lengan Atas (°)", "Sudut_LenganBawah":"Sudut Lengan Bawah (°)",
        "Deviasi_Pergelangan":"Deviasi Pergelangan (°)", "Sudut_Lutut":"Sudut Lutut (°)",
        "Skor_Leher":"Skor Leher", "Skor_Tubuh":"Skor Batang Tubuh", "Skor_Kaki":"Skor Kaki",
        "Skor_LenganAtas":"Skor Lengan Atas", "Skor_LenganBawah":"Skor Lengan Bawah",
        "Skor_Pergelangan":"Skor Pergelangan", "Table_A":"Table A", "Table_B":"Table B",
        "Force_Score":"Force Score", "Score_A":"Score A", "Score_B":"Score B",
        "Score_C":"Score C", "Activity_Score":"Activity Score",
        "REBA_Final":"Skor REBA Final", "Kategori":"Kategori Risiko", "Tindakan":"Tindakan",
    }

    rows = []
    for r in results:
        row = {label_map[k]: v for k,v in r.items() if k in label_map}
        row["Nama Laporan"] = nama_laporan
        rows.append(row)

    df = pd.DataFrame(rows)

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as wr:
        df.to_excel(wr, sheet_name="Hasil REBA", index=False, startrow=2)
        ws = wr.sheets["Hasil REBA"]

        last_col = get_column_letter(len(df.columns))
        ws.merge_cells(f"A1:{last_col}1")
        ws["A1"] = f"Laporan REBA — {nama_laporan} — {datetime.now().strftime('%d %B %Y')} — {len(results)} Foto"
        ws["A1"].font      = Font(bold=True, size=13, color="FFFFFF")
        ws["A1"].fill      = PatternFill("solid", fgColor="1A3A6A")
        ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 28

        for cell in ws[3]:
            cell.font      = Font(bold=True, color="FFFFFF", size=10)
            cell.fill      = PatternFill("solid", fgColor="2E4A7A")
            cell.alignment = Alignment(horizontal="center")

        reba_col_idx = None
        for i, cell in enumerate(ws[3], 1):
            if cell.value == "Skor REBA Final":
                reba_col_idx = i
                break

        cmap = {(1,1):"27AE60",(2,3):"2ECC71",(4,7):"F39C12",(8,10):"E67E22",(11,15):"E74C3C"}
        for row_idx in range(4, 4+len(results)):
            if reba_col_idx:
                dc = ws.cell(row=row_idx, column=reba_col_idx)
                try:
                    score = int(dc.value)
                    cc = "FFFFFF"
                    for (lo,hi),hx in cmap.items():
                        if lo<=score<=hi: cc=hx; break
                    dc.fill = PatternFill("solid", fgColor=cc)
                    dc.font = Font(bold=True, color="000000")
                except: pass

        thin = Border(
            left=Side(style="thin"), right=Side(style="thin"),
            top=Side(style="thin"),  bottom=Side(style="thin"),
        )
        for row_cells in ws.iter_rows(min_row=3, max_row=3+len(results)):
            for cell in row_cells: cell.border = thin

        for col in ws.columns:
            ml = max((len(str(c.value)) if c.value else 0) for c in col)
            ws.column_dimensions[get_column_letter(col[0].column)].width = max(ml+3, 12)

    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# LOGIN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
VALID_USERS = {"Kaizen": "Toyota2026"}

def check_login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    return st.session_state.logged_in

def show_login_page():
    col_center = st.columns([1, 1.5, 1])[1]
    with col_center:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center; margin-bottom:1.5rem;'>
            <span style='font-size:3rem;'>🦴</span>
            <h2 style='margin:0.3rem 0 0.1rem;'>REBA Analyzer</h2>
            <p style='color:gray; font-size:0.9rem;'>Ergonomics Risk Assessment</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Masukkan username")
            password = st.text_input("🔒 Password", type="password", placeholder="Masukkan password")
            submit   = st.form_submit_button("🔑 Login", use_container_width=True, type="primary")

        if submit:
            if username in VALID_USERS and VALID_USERS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username  = username
                st.rerun()
            else:
                st.error("❌ Username atau password salah. Coba lagi.", icon="🚫")

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [
    ("results_list", []),      # list of (result_dict, annotated_img, filename)
    ("analyzed", False),
    ("logged_in", False),
    ("username", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# GUARD: tampilkan login jika belum login
# ─────────────────────────────────────────────────────────────────────────────
if not check_login():
    show_login_page()
    st.stop()

# =============================================
# SIDEBAR
# =============================================
with st.sidebar:
    st.title("🦴 REBA Analyzer")
    st.caption("Ergonomics Risk Assessment · Hignett & McAtamney, 2000")
    st.markdown(f"👤 **{st.session_state.get('username','')}**")
    if st.button("🚪 Logout", use_container_width=True):
        for k in ["logged_in","username","results_list","analyzed"]:
            st.session_state[k] = False if k in ("logged_in","analyzed") else ([] if k=="results_list" else "")
        st.rerun()
    st.divider()

    st.subheader("📁 Upload Foto")
    uploaded_files = st.file_uploader(
        "Pilih satu atau lebih foto postur kerja",
        type=["jpg","jpeg","png","bmp"],
        accept_multiple_files=True,
        help="Pastikan seluruh tubuh (kepala hingga kaki) terlihat jelas",
    )
    if uploaded_files:
        st.caption(f"✅ {len(uploaded_files)} foto dipilih")

    st.divider()

    st.subheader("⚖️ Beban")
    beban = st.number_input(
        "Berat yang diangkat (kg)",
        min_value=0.0, max_value=500.0, value=0.0, step=0.5,
    )
    fs_val = force_score(beban)
    st.caption(["🟢 Ringan < 5 kg  (Force Score 0)",
                "🟡 Sedang 5–10 kg (Force Score 1)",
                "🔴 Berat > 10 kg  (Force Score 2)"][fs_val])

    st.divider()

    st.subheader("🏭 Aktivitas")
    AKTIVITAS_LIST = [
        "Pengangkatan Manual","Menurunkan Beban","Mendorong / Menarik",
        "Perakitan (Assembly)","Pengelasan","Pengepakan / Packaging",
        "Inspeksi Visual","Pengoperasian Mesin","Pekerjaan Kantor / Duduk",
        "Lainnya...",
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
        "🔍 Analisis Semua Foto",
        type="primary",
        use_container_width=True,
        disabled=(not uploaded_files),
    )

    if st.session_state.analyzed and st.session_state.results_list:
        st.button(
            "🗑️ Reset / Hapus Semua",
            use_container_width=True,
            on_click=lambda: st.session_state.update({"results_list": [], "analyzed": False}),
        )

    st.divider()
    st.caption("**Alur kerja:**\n"
               "1. Upload foto (bisa banyak)\n"
               "2. Isi beban & aktivitas\n"
               "3. Centang activity score\n"
               "4. Klik Analisis Semua Foto\n"
               "5. Download Excel (semua hasil)")

# =============================================
# MAIN AREA
# =============================================

# ── JALANKAN ANALISIS ──────────────────────────────────────────────────────
if analyze_btn and uploaded_files:
    results_list = []
    progress = st.progress(0, text="Memproses foto...")
    for i, uf in enumerate(uploaded_files):
        progress.progress((i) / len(uploaded_files), text=f"Memproses {uf.name} ({i+1}/{len(uploaded_files)})...")
        img_bgr = cv2.cvtColor(
            np.array(Image.open(uf).convert("RGB")), cv2.COLOR_RGB2BGR
        )
        annotated, result = analyze_pose(img_bgr, beban, aktivitas, activity_score)
        if result is not None:
            result["Nama_File"] = uf.name
            results_list.append((result, annotated, uf.name))
        else:
            results_list.append((None, None, uf.name))
    progress.progress(1.0, text="Selesai!")
    st.session_state.results_list = results_list
    st.session_state.analyzed     = True
    st.rerun()

# ── TAMPILKAN HASIL ────────────────────────────────────────────────────────
if st.session_state.analyzed and st.session_state.results_list:
    results_list = st.session_state.results_list

    ok_results   = [(r, ann, fn) for r, ann, fn in results_list if r is not None]
    fail_results = [fn for r, ann, fn in results_list if r is None]

    st.subheader(f"📊 Hasil Analisis — {len(ok_results)} dari {len(results_list)} foto berhasil")

    if fail_results:
        st.warning(f"⚠️ Pose tidak terdeteksi pada: **{', '.join(fail_results)}**", icon="⚠️")

    # Tab per foto + tab ringkasan
    if ok_results:
        tab_labels = [f"📷 {fn}" for _, _, fn in ok_results] + ["📋 Ringkasan & Export"]
        tabs = st.tabs(tab_labels)

        for tab, (r, annotated, fn) in zip(tabs[:-1], ok_results):
            with tab:
                col_img, col_res = st.columns([3, 2], gap="large")
                with col_img:
                    st.subheader("Visualisasi Skeleton")
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, use_container_width=True,
                             caption=f"Skeleton: {fn}")

                with col_res:
                    st.subheader("Hasil Analisis")
                    reba = r["REBA_Final"]
                    kat, icon_r, tindakan = risk_cat(reba)
                    st.metric(
                        label=f"{icon_r} Skor REBA Final",
                        value=reba,
                        delta=f"Risiko {kat} · {tindakan}",
                        delta_color="off",
                    )
                    st.divider()

                    st.markdown("**📐 Sudut Terukur**")
                    st.dataframe(
                        pd.DataFrame({
                            "Segmen":    ["Leher","Batang Tubuh","Lengan Atas",
                                          "Lengan Bawah","Pergelangan","Lutut"],
                            "Sudut (°)": [r["Sudut_Leher"], r["Sudut_Tubuh"],
                                          r["Sudut_LenganAtas"], r["Sudut_LenganBawah"],
                                          r["Deviasi_Pergelangan"], r["Sudut_Lutut"]],
                            "Skor":      [r["Skor_Leher"], r["Skor_Tubuh"],
                                          r["Skor_LenganAtas"], r["Skor_LenganBawah"],
                                          r["Skor_Pergelangan"], r["Skor_Kaki"]],
                        }),
                        hide_index=True, use_container_width=True,
                    )

                    st.markdown("**🔢 Alur Perhitungan**")
                    st.dataframe(
                        pd.DataFrame({
                            "Parameter": ["Table A","Force Score","Score A",
                                          "Table B","Coupling","Score B",
                                          "Score C","Activity Score","REBA FINAL"],
                            "Nilai":     [r["Table_A"], r["Force_Score"], r["Score_A"],
                                          r["Table_B"], "1 (Fair)", r["Score_B"],
                                          r["Score_C"], f"+{r['Activity_Score']}", reba],
                        }),
                        hide_index=True, use_container_width=True,
                    )
                    st.caption(f"🕐 {r['Timestamp']}  ·  ⚖️ {r['Beban_kg']} kg  ·  🏭 {r['Aktivitas']}")

        # ── TAB RINGKASAN & EXPORT ────────────────────────────────────────
        with tabs[-1]:
            st.subheader("📋 Ringkasan Semua Foto")
            summary_rows = []
            for r, _, fn in ok_results:
                kat, icon_r, tindakan = risk_cat(r["REBA_Final"])
                summary_rows.append({
                    "No": len(summary_rows)+1,
                    "File": fn,
                    "Aktivitas": r["Aktivitas"],
                    "REBA Final": r["REBA_Final"],
                    "Kategori": f"{icon_r} {kat}",
                    "Tindakan": tindakan,
                })
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

            st.divider()
            st.markdown("**📤 Export Semua Hasil ke Excel**")
            nama_laporan = st.text_input(
                "Nama laporan",
                value=aktivitas if 'aktivitas' in dir() else "",
                placeholder="Contoh: Pengangkatan Barang Gudang A",
                key="nama_laporan_multi",
            )
            if nama_laporan.strip():
                excel_data = build_excel_multi([r for r, _, _ in ok_results], nama_laporan.strip())
                st.download_button(
                    label=f"⬇️ Download Excel ({len(ok_results)} hasil)",
                    data=excel_data,
                    file_name=f"REBA_Multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    type="primary",
                )

else:
    st.info("⬅️ Upload foto postur kerja (bisa banyak sekaligus), lalu klik **Analisis Semua Foto**.")
