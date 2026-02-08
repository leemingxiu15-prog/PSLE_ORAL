import streamlit as st
import pandas as pd
import json
import tempfile
import os
import datetime
import base64
import wave
import contextlib
from openai import OpenAI

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(page_title="PSLE Coach", page_icon="🦁", layout="wide")

# Load Secrets
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    sheet_url = st.secrets["GOOGLE_SHEET_URL"]
except Exception:
    st.error("❌ Secrets missing! Check .streamlit/secrets.toml")
    st.stop()

client = OpenAI(api_key=api_key)

# Define File Paths
MISTAKES_FILE = "mistakes.json"
HISTORY_FILE = "history.json"
WORD_BANK_FILE = "word_bank.json"
IMAGES_FOLDER = "images" 

# Ensure Image Folder Exists
if not os.path.exists(IMAGES_FOLDER):
    os.makedirs(IMAGES_FOLDER)

# Custom CSS for Cleaner UI
st.markdown("""
    <style>
        .main .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
        .stAudioInput { margin-top: 0px !important; }
        .stAlert { padding: 0.5rem; }
        h1, h2, h3 { font-family: sans-serif; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DEFINITIONS (FRAMEWORKS)
# ==========================================
FRAMEWORKS = {
    "PACT": {
        "name": "Picture Description",
        "parts": {
            "People": "Who is in the scene? (Appearance, Age, Roles)",
            "Actions": "What are they doing? (Precise verbs & continuous tense)",
            "Context": "Where is this? What is the atmosphere/weather?",
            "Thoughts": "What are they likely feeling? What might happen next?"
        }
    },
    "PEEL": {
        "name": "Opinion Question",
        "parts": {
            "Point": "Direct Answer. (Yes/No/I agree because...)",
            "Explanation": "Why do you say that? (Give reasons)",
            "Example": "Share a specific example (Personal story or General knowledge)",
            "Link": "Link back to the question (Therefore, I believe...)"
        }
    },
    "SAFE": {
        "name": "Recount Question",
        "parts": {
            "Setting": "When and Where did it happen? Who was there?",
            "Action": "What happened? Describe the sequence of events (The Climax).",
            "Feeling": "How did you feel at that moment? (Use rich adjectives)",
            "Explanation": "What did you learn? How did it end? (The Lesson)"
        }
    }
}

MOE_CRITERIA = """
CRITICAL GRADING CONTEXT (2025 SYLLABUS):
Score each component out of 5 marks (Total 20).
1. AL1 STANDARD (Distinction):
   - "Thick" answers with personal details.
   - Specific vocabulary (e.g. "overjoyed" not "happy").
   - Clear structure linking back to the question.
"""

# ==========================================
# 3. DATA MANAGEMENT (NO-CACHE VERSION)
# ==========================================

def load_all_data():
    """
    Loads data from Google Sheets with AGGRESSIVE cleaning.
    No Caching used to ensure fresh data on every reload.
    """
    try:
        # 1. Read CSV directly from URL
        df = pd.read_csv(sheet_url)
        
        # 2. Clean Headers
        df.columns = df.columns.str.strip()
        
        # 3. Drop "Ghost Rows" (Rows where Theme is empty/NaN)
        df = df.dropna(subset=['Theme'])
        
        # 4. Filter out string-based "nan" or empty strings
        df = df[df['Theme'].astype(str).str.lower() != 'nan']
        df = df[df['Theme'].astype(str).str.strip() != ""]
        
        # 5. Fill remaining missing values with empty strings
        df = df.fillna("")
        
        # 6. Verify Required Columns
        required = ["Theme", "Topic", "Type", "Question", "Framework", "Model_Part", "AL2_Sample", "AL1_Sample"]
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            st.error(f"❌ Missing columns in Google Sheet: {missing}")
            return pd.DataFrame()
            
        return df
    except Exception as e:
        st.error(f"Error reading Google Sheet. Check your URL in secrets.toml. Error: {e}")
        return pd.DataFrame()

def load_data(filename, default=[]):
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f: return json.load(f)
        except: return default
    return default

def save_data(filename, data):
    with open(filename, "w") as f: json.dump(data, f)

def log_mistakes(mistakes_list, theme, topic, framework, part):
    current_log = load_data(MISTAKES_FILE, [])
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for m in mistakes_list:
        entry = {
            "timestamp": timestamp, "theme": theme, "topic": topic,
            "framework": framework, "part": part,
            "type": m.get('type', 'General'), "original": m['original'],
            "fixed": m['fixed'], "issue": m['issue']
        }
        current_log.append(entry)
    save_data(MISTAKES_FILE, current_log)

def log_history(theme, topic, framework_key, total_score, full_details):
    history = load_data(HISTORY_FILE, [])
    record = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "theme": theme, "topic": topic, "model": framework_key,
        "score": total_score, "details": full_details
    }
    history.append(record)
    save_data(HISTORY_FILE, history)

def save_to_word_bank(word, category):
    bank = load_data(WORD_BANK_FILE, [])
    if not any(w['word'] == word for w in bank):
        bank.append({"date": datetime.datetime.now().strftime("%Y-%m-%d"), "word": word, "category": category})
        save_data(WORD_BANK_FILE, bank)
        return True
    return False

# ==========================================
# 4. AI & HELPER FUNCTIONS
# ==========================================

def transcribe_audio(audio_file):
    audio_file.seek(0)
    return client.audio.transcriptions.create(
        model="whisper-1", file=audio_file, language="en"
    ).text

def encode_image_from_path(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def grade_component_based(question, parts_data, framework_type, image_base64=None):
    content_text = f"Question: {question}\nFRAMEWORK: {framework_type}\n\nSTUDENT ANSWER BROKEN DOWN:\n"
    for part, text in parts_data.items():
        content_text += f"- {part.upper()}: {text}\n"

    user_content = [{"type": "text", "text": content_text}]
    if image_base64:
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})

    keys_list = list(parts_data.keys())
    prompt = f"""
    You are a Singapore MOE Teacher grading PSLE Oral.
    {MOE_CRITERIA}
    
    TASK:
    Analyze the student's answer component by component.
    For EACH component ({', '.join(keys_list)}):
    1. Score it out of 5.
    2. Identify specific errors (Grammar/Vocab/Tense).
    3. Write a concise Model Answer segment (Level Up AL1 Standard).
    4. Give a brief reason why the correction is needed.
    
    OUTPUT JSON FORMAT:
    {{
        "{keys_list[0]}": {{
            "score": 4,
            "mistakes": [{{"original": "...", "fixed": "...", "issue": "...", "type": "Grammar"}}],
            "model_part": "...",
            "feedback": "..."
        }},
        ...
    }}
    """
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "system", "content": "JSON only."}, {"role": "user", "content": prompt}, {"role": "user", "content": user_content}], 
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def grade_drill(question, transcript, drill_type, image_base64=None):
    user_content = [{"type": "text", "text": f"Question: {question}\nStudent Answer: {transcript}"}]
    if image_base64:
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})

    prompt = f"""
    You are a Creativity Coach. SPEED DRILL for "{drill_type}".
    Rules: 1. Count distinct ideas. 2. Ignore grammar. 3. Suggest 3 better vocab words.
    
    Output JSON:
    {{ "count": 0, "items_found": ["item1"], "feedback": "...", "suggestions": ["Word1", "Word2"] }}
    """
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "system", "content": "JSON only."}, {"role": "user", "content": prompt}, {"role": "user", "content": user_content}], 
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# ==========================================
# 5. MAIN UI APPLICATION
# ==========================================

# Initialize Session State
if "exam_results" not in st.session_state: st.session_state.exam_results = {}
if "selection" not in st.session_state: st.session_state.selection = {"theme": None, "topic": None}

with st.sidebar:
    st.header("🦁 Navigation")
    
    # --- DEBUGGER (Verify Data Loading) ---
    with st.expander("🛠️ Data Inspector (Debug)", expanded=False):
        if st.button("🔄 Force Refresh"):
            st.rerun()
        df_debug = load_all_data()
        st.write(f"Rows Loaded: {len(df_debug)}")
        if not df_debug.empty:
            st.write("Themes:", df_debug['Theme'].unique())
    # --------------------------------------

    mode = st.radio("Go to:", ["Practice Exam", "⚡ Speed Brainstorm", "Review Mistakes", "📈 Progress Report"])

    # Load Data if needed for exam/drill
    if mode in ["Practice Exam", "⚡ Speed Brainstorm"]:
        st.divider()
        df = load_all_data()
        
        if not df.empty:
            # Theme Selector
            themes = sorted(df['Theme'].unique())
            sel_theme = st.selectbox("1. Select Theme:", themes)
            
            # Topic Selector (Filtered by Theme)
            topics = sorted(df[df['Theme'] == sel_theme]['Topic'].unique())
            sel_topic = st.selectbox("2. Select Specific Topic:", topics)
            
            # Reset results if selection changes
            current_selection = f"{sel_theme}-{sel_topic}"
            if st.session_state.selection["topic"] != current_selection:
                st.session_state.exam_results = {}
                st.session_state.selection["topic"] = current_selection
                st.rerun()

# ------------------------------------------------------------------
# MODE 1: PRACTICE EXAM (Full Features)
# ------------------------------------------------------------------
if mode == "Practice Exam":
    if df.empty: st.warning("Please check your database connection."); st.stop()
    
    st.title(f"📝 {sel_theme}: {sel_topic}")
    
    # Filter Data for current topic
    topic_data = df[(df['Theme'] == sel_theme) & (df['Topic'] == sel_topic)]
    
    # Separate Rows by Question Type
    pic_rows = topic_data[topic_data['Type'].str.contains("Picture", case=False)]
    opi_rows = topic_data[topic_data['Type'].str.contains("Opinion", case=False)]
    rec_rows = topic_data[topic_data['Type'].str.contains("Recount", case=False)]
    
    col_left, col_right = st.columns([1, 1])
    
    # LEFT: Image Display
    img_b64 = None
    with col_left:
        st.markdown("### 🖼️ Stimulus")
        if not pic_rows.empty:
            img_file = pic_rows.iloc[0]['Image_File']
            if img_file and str(img_file).lower() not in ['nan', '']:
                img_path = os.path.join(IMAGES_FOLDER, str(img_file).strip())
                if os.path.exists(img_path):
                    st.image(img_path, width=350)
                    img_b64 = encode_image_from_path(img_path)
                else:
                    st.warning(f"Image not found: {img_file}")
            else:
                st.info("No image assigned.")

    # RIGHT: Tabs & Recorders
    with col_right:
        tab1, tab2, tab3 = st.tabs(["🖼️ Picture", "💭 Opinion", "📖 Recount"])

        def render_exam_tab(rows, framework_key, tab_id):
            if rows.empty: st.info("No question available."); return
            
            # Get Question
            q_text = rows.iloc[0]['Question']
            framework = FRAMEWORKS[framework_key]
            parts = framework["parts"]
            
            st.info(f"**Q:** {q_text}")
            
            # Render Audio Inputs
            audio_buffers = {}
            all_filled = True
            for part_key, part_desc in parts.items():
                c1, c2 = st.columns([6, 4])
                c1.markdown(f"**{part_key}**: {part_desc}")
                aud = c2.audio_input(f"rec_{part_key}", key=f"rec_{tab_id}_{part_key}", label_visibility="collapsed")
                if aud: audio_buffers[part_key] = aud
                else: all_filled = False
            
            # Submit Logic
            if all_filled:
                if st.button(f"🚀 Submit {framework_key}", key=f"btn_{tab_id}", use_container_width=True):
                    with st.spinner("Grading..."):
                        # 1. Transcribe
                        transcripts = {k: transcribe_audio(v) for k, v in audio_buffers.items()}
                        
                        # 2. Grade
                        res = grade_component_based(q_text, transcripts, framework_key, img_b64 if framework_key=="PACT" else None)
                        
                        # 3. Match References (AL1/AL2)
                        refs = {}
                        for _, row in rows.iterrows():
                            # Normalize key
                            p_key = str(row['Model_Part']).strip()
                            refs[p_key] = {"al2": row['AL2_Sample'], "al1": row['AL1_Sample']}
                        
                        # 4. Save Session
                        st.session_state.exam_results[tab_id] = {
                            "res": res, "transcripts": transcripts, "parts": list(parts.keys()), "refs": refs
                        }
                        
                        # 5. Log History & Mistakes
                        total = sum(d['score'] for d in res.values())
                        log_history(sel_theme, sel_topic, framework_key, total, res)
                        
                        for p_name, data in res.items():
                            if data.get('mistakes'):
                                log_mistakes(data['mistakes'], sel_theme, sel_topic, framework_key, p_name)
                        st.rerun()

        with tab1: render_exam_tab(pic_rows, "PACT", "tab_pic")
        with tab2: render_exam_tab(opi_rows, "PEEL", "tab_opi")
        with tab3: render_exam_tab(rec_rows, "SAFE", "tab_rec")

    # RESULTS DISPLAY
    if st.session_state.exam_results:
        st.divider()
        st.markdown("## 📊 Grading Results")
        res_tabs = st.tabs([k for k in ["tab_pic", "tab_opi", "tab_rec"] if k in st.session_state.exam_results])
        
        for idx, (tab_key, data) in enumerate(st.session_state.exam_results.items()):
            with res_tabs[idx]:
                st.markdown(f"### Total Score: {sum(d['score'] for d in data['res'].values())}/20")
                for part in data["parts"]:
                    if part in data["res"]:
                        p_res = data["res"][part]
                        p_refs = data["refs"].get(part, {"al2": "-", "al1": "-"})
                        
                        # Fuzzy match check
                        if p_refs["al2"] == "-": 
                            match = next((k for k in data["refs"] if k.lower() == part.lower()), None)
                            if match: p_refs = data["refs"][match]

                        with st.container():
                            st.markdown(f"#### 🔹 {part} (Score: {p_res['score']}/5)")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("**Your Answer:**")
                                st.write(data["transcripts"][part])
                                if p_res.get('mistakes'):
                                    for m in p_res['mistakes']: st.error(f"❌ {m.get('original')} → ✅ {m.get('fixed')}")
                            with c2:
                                st.markdown("**AI Improvement:**")
                                st.info(p_res['model_part'])
                                st.caption(p_res['feedback'])
                            
                            with st.expander("📚 Compare Samples"):
                                r1, r2 = st.columns(2)
                                r1.text(f"Standard (AL2): {p_refs['al2']}")
                                r2.success(f"Distinction (AL1): {p_refs['al1']}")
                            st.divider()

# ------------------------------------------------------------------
# MODE 2: SPEED BRAINSTORM
# ------------------------------------------------------------------
elif mode == "⚡ Speed Brainstorm":
    if df.empty: st.warning("No data."); st.stop()
    st.title(f"⚡ Brainstorm: {sel_topic}")
    
    topic_data = df[(df['Theme'] == sel_theme) & (df['Topic'] == sel_topic)]
    pic_rows = topic_data[topic_data['Type'].str.contains("Picture", case=False)]
    
    c1, c2 = st.columns(2)
    img_b64 = None
    with c1:
        if not pic_rows.empty:
            img_file = pic_rows.iloc[0]['Image_File']
            if img_file and str(img_file).lower() not in ['nan', '']:
                img_path = os.path.join(IMAGES_FOLDER, str(img_file).strip())
                if os.path.exists(img_path):
                    st.image(img_path, width=350)
                    img_b64 = encode_image_from_path(img_path)

    with c2:
        drill = st.radio("Target:", ["Actions", "Feelings", "Perspectives", "Consequences"])
        st.subheader(f"List 3 {drill}!")
        aud = st.audio_input("Go!")
        if aud and st.button("Check"):
            with st.spinner("Analyzing..."):
                txt = transcribe_audio(aud)
                res = grade_drill(sel_topic, txt, drill, img_b64)
                st.metric("Score", f"{res['count']}/3")
                st.success(f"Ideas: {', '.join(res['items_found'])}")
                
                # Word Bank Integration
                st.write("### Suggestions")
                cols = st.columns(3)
                for i, word in enumerate(res['suggestions']):
                    if cols[i].button(f"➕ {word}", key=f"wb_{word}"):
                        save_to_word_bank(word, drill)
                        st.toast(f"Saved {word}!")

# ------------------------------------------------------------------
# MODE 3: REVIEW MISTAKES (Rebuilt)
# ------------------------------------------------------------------
elif mode == "Review Mistakes":
    st.title("📚 Mistakes Bank")
    mistakes = load_data(MISTAKES_FILE, [])
    if mistakes:
        df_m = pd.DataFrame(mistakes)
        # Clean NaNs
        df_m = df_m[df_m['theme'].notna() & (df_m['theme'] != "nan")]
        
        c1, c2, c3 = st.columns(3)
        with c1: f_thm = st.multiselect("Theme", df_m['theme'].unique())
        with c2: f_mod = st.multiselect("Model", df_m['framework'].unique())
        with c3: f_typ = st.multiselect("Type", df_m['type'].unique())
        
        if f_thm: df_m = df_m[df_m['theme'].isin(f_thm)]
        if f_mod: df_m = df_m[df_m['framework'].isin(f_mod)]
        if f_typ: df_m = df_m[df_m['type'].isin(f_typ)]
        
        if not df_m.empty:
            st.dataframe(df_m[['timestamp', 'theme', 'framework', 'part', 'original', 'fixed']], use_container_width=True)
        else:
            st.info("No mistakes match filters.")
    else: st.info("No mistakes recorded yet.")

# ------------------------------------------------------------------
# MODE 4: PROGRESS REPORT (Rebuilt & Enhanced)
# ------------------------------------------------------------------
elif mode == "📈 Progress Report":
    st.title("📈 Analytics")
    history = load_data(HISTORY_FILE, [])
    
    if history:
        df_h = pd.DataFrame(history)
        
        # 1. Clean Dates
        df_h['date'] = pd.to_datetime(df_h['date'], errors='coerce')
        df_h = df_h.dropna(subset=['date'])
        
        # 2. Clean Strings
        df_h = df_h[~df_h['theme'].astype(str).str.lower().isin(['nan', 'unknown'])]
        
        # 3. Filters
        c1, c2, c3 = st.columns(3)
        with c1: f_thm = st.multiselect("Theme", sorted(df_h['theme'].unique()))
        with c2: f_mod = st.multiselect("Model", sorted(df_h['model'].unique()))
        
        # Extract Parts dynamically
        all_parts = set()
        for d in df_h['details']: 
            if isinstance(d, dict): all_parts.update(d.keys())
        with c3: f_prt = st.multiselect("Part", sorted(list(all_parts)))
        
        # Apply Filters
        if f_thm: df_h = df_h[df_h['theme'].isin(f_thm)]
        if f_mod: df_h = df_h[df_h['model'].isin(f_mod)]
        if f_prt: 
            # Filter rows that contain ANY of the selected parts
            df_h = df_h[df_h['details'].apply(lambda x: any(p in x for p in f_prt) if isinstance(x, dict) else False)]
            
        if not df_h.empty:
            # 4. Metrics Calculation
            avg = df_h['score'].mean()
            
            # If Parts Selected -> Re-calculate Average based ONLY on those parts
            if f_prt:
                p_scores = []
                for det in df_h['details']:
                    if isinstance(det, dict):
                        for p in f_prt:
                            if p in det: p_scores.append(det[p]['score'])
                if p_scores: 
                    avg = sum(p_scores)/len(p_scores)
                    max_score = 5
                else: max_score = 5
            else:
                max_score = 20
            
            st.metric("Average Score", f"{avg:.1f}/{max_score}")
            
            # 5. Breakdown Table
            part_stats = {}
            for det in df_h['details']:
                if isinstance(det, dict):
                    for p, info in det.items():
                        if f_prt and p not in f_prt: continue
                        if p not in part_stats: part_stats[p] = []
                        part_stats[p].append(info.get('score', 0))
            
            if part_stats:
                st.write("#### Component Weaknesses")
                data = [{"Part": k, "Avg": round(sum(v)/len(v), 1), "Count": len(v)} for k, v in part_stats.items()]
                st.table(pd.DataFrame(data).set_index("Part"))
            
            # 6. Detailed History
            st.divider()
            for _, row in df_h.sort_values('date', ascending=False).iterrows():
                date_str = row['date'].strftime('%d %b %H:%M')
                with st.expander(f"{date_str} | {row['theme']} | Score: {row['score']}"):
                    if isinstance(row['details'], dict):
                        show = f_prt if f_prt else row['details'].keys()
                        for p in show:
                            if p in row['details']:
                                info = row['details'][p]
                                st.markdown(f"**{p} ({info.get('score')}/5)**")
                                
                                col_fb, col_md = st.columns(2)
                                with col_fb:
                                    if 'feedback' in info: st.caption(f"💡 {info['feedback']}")
                                    if info.get('mistakes'):
                                        for m in info['mistakes']: st.error(f"❌ {m.get('original')} → {m.get('fixed')}")
                                with col_md:
                                    if 'model_part' in info: st.success(f"**Model:** {info['model_part']}")
                                st.markdown("---")
        else: st.warning("No data matches filters.")
    else: st.info("No history found.")