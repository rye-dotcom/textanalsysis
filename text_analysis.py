# app.py
import json
import csv
import io
import math
from collections import Counter

import streamlit as st
import spacy
from spacy import displacy
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import annotated_text
from annotated_text import annotated_text as ann_text
from fpdf import FPDF

# ── NLTK bootstrap ──────────────────────────────────────────────────────────
for pkg in ("punkt", "stopwords", "punkt_tab"):
    nltk.download(pkg, quiet=True)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Text Analyser",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session-state defaults ───────────────────────────────────────────────────
DEFAULTS = {
    "selected_style": "dep",
    "analysis_done": False,
    "doc": None,
    "text": "",
    "results": {},
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# Core analysis class


class TextAnalysisApp:
    ENTITY_COLORS = {
        "PERSON": "#f9a8d4",
        "ORG": "#86efac",
        "GPE": "#93c5fd",
        "LOC": "#fcd34d",
        "DATE": "#c4b5fd",
        "MONEY": "#6ee7b7",
    }
    ENTITY_TYPES = list(ENTITY_COLORS.keys())

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words("english"))

    # ── NLP helpers ──────────────────────────────────────────────────────────
    def extract_keywords(self, text, top_n=15):
        tokens = word_tokenize(text.lower())
        filtered = [w for w in tokens if w.isalpha() and w not in self.stop_words]
        return Counter(filtered).most_common(top_n)

    def get_sentiment(self, text):
        blob = TextBlob(text)
        pol = blob.sentiment.polarity
        sub = blob.sentiment.subjectivity
        label = "Positive" if pol > 0.1 else ("Negative" if pol < -0.1 else "Neutral")
        return pol, sub, label

    def sentence_sentiments(self, text):
        rows = []
        for sent in sent_tokenize(text):
            pol, sub, label = self.get_sentiment(sent)
            rows.append(
                {
                    "Sentence": sent,
                    "Polarity": round(pol, 3),
                    "Subjectivity": round(sub, 3),
                    "Label": label,
                }
            )
        return rows

    def readability_scores(self, text):
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        words_alpha = [w for w in words if w.isalpha()]
        syllables = sum(self._count_syllables(w) for w in words_alpha)

        n_sent = max(len(sentences), 1)
        n_words = max(len(words_alpha), 1)
        n_syl = max(syllables, 1)

        fk_ease = 206.835 - 1.015 * (n_words / n_sent) - 84.6 * (n_syl / n_words)
        fk_grade = 0.39 * (n_words / n_sent) + 11.8 * (n_syl / n_words) - 15.59
        avg_sent_len = n_words / n_sent
        avg_syl = n_syl / n_words

        return {
            "Flesch Reading Ease": round(fk_ease, 2),
            "Flesch-Kincaid Grade": round(fk_grade, 2),
            "Avg Sentence Length": round(avg_sent_len, 2),
            "Avg Syllables/Word": round(avg_syl, 2),
            "Total Sentences": n_sent,
            "Total Words": n_words,
        }

    @staticmethod
    def _count_syllables(word):
        word = word.lower()
        vowels = "aeiouy"
        count = prev_vowel = 0
        for ch in word:
            is_v = ch in vowels
            if is_v and not prev_vowel:
                count += 1
            prev_vowel = is_v
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    # ── Charts ───────────────────────────────────────────────────────────────
    def plotly_keyword_bar(self, keywords):
        words, counts = zip(*keywords) if keywords else ([], [])
        fig = px.bar(
            x=list(counts),
            y=list(words),
            orientation="h",
            labels={"x": "Frequency", "y": "Keyword"},
            title="Top Keywords",
            color=list(counts),
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            height=420,
        )
        return fig

    def plotly_entity_pie(self, ents, selected_types):
        filtered = [e for e in ents if e.label_ in selected_types]
        if not filtered:
            return None
        counts = Counter(e.label_ for e in filtered)
        fig = px.pie(
            values=list(counts.values()),
            names=list(counts.keys()),
            title="Named Entity Distribution",
            color=list(counts.keys()),
            color_discrete_map=self.ENTITY_COLORS,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        return fig

    def plotly_sentiment_scatter(self, sent_rows):
        fig = px.scatter(
            sent_rows,
            x="Polarity",
            y="Subjectivity",
            color="Label",
            hover_data=["Sentence"],
            color_discrete_map={
                "Positive": "#22c55e",
                "Negative": "#ef4444",
                "Neutral": "#94a3b8",
            },
            title="Sentence-level Sentiment",
        )
        fig.add_vline(x=0.1, line_dash="dot", line_color="gray")
        fig.add_vline(x=-0.1, line_dash="dot", line_color="gray")
        fig.update_layout(height=420)
        return fig

    def plotly_readability_gauge(self, scores):
        fke = scores["Flesch Reading Ease"]
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=fke,
                title={"text": "Flesch Reading Ease"},
                delta={"reference": 60},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#6366f1"},
                    "steps": [
                        {"range": [0, 30], "color": "#fca5a5"},
                        {"range": [30, 60], "color": "#fde68a"},
                        {"range": [60, 100], "color": "#86efac"},
                    ],
                    "threshold": {
                        "line": {"color": "#1e293b", "width": 4},
                        "thickness": 0.75,
                        "value": 60,
                    },
                },
            )
        )
        fig.update_layout(height=300)
        return fig

    def wordcloud_figure(self, text):
        wc = WordCloud(
            width=900,
            height=400,
            background_color="white",
            colormap="viridis",
            random_state=42,
        ).generate(text)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        return fig

    def plotly_dependency_graph(self, doc):
        edges, labels = [], {}
        for token in doc:
            labels[token.i] = f"{token.text}\n({token.pos_})"
            for child in token.children:
                edges.append((token.i, child.i))
        G = nx.DiGraph(edges)
        pos = nx.spring_layout(G, seed=7)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_text = [labels.get(n, str(n)) for n in G.nodes()]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"width": 1, "color": "#94a3b8"},
                hoverinfo="none",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=node_text,
                textposition="top center",
                marker={
                    "size": 14,
                    "color": "#6366f1",
                    "line": {"width": 1, "color": "#fff"},
                },
                hoverinfo="text",
            )
        )
        fig.update_layout(
            showlegend=False,
            height=500,
            title="Dependency Graph",
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        )
        return fig

    # ── Annotated text ───────────────────────────────────────────────────────
    def render_annotated(self, doc, selected_types):
        parts = []
        cursor = 0
        text = doc.text
        filtered_ents = [e for e in doc.ents if e.label_ in selected_types]
        for ent in filtered_ents:
            if ent.start_char > cursor:
                parts.append(text[cursor : ent.start_char])
            color = self.ENTITY_COLORS.get(ent.label_, "#e2e8f0")
            parts.append((ent.text, ent.label_, color))
            cursor = ent.end_char
        if cursor < len(text):
            parts.append(text[cursor:])
        return parts

    # ── Export helpers ────────────────────────────────────────────────────────
    @staticmethod
    def to_json(results: dict) -> str:
        safe = {
            k: v
            for k, v in results.items()
            if isinstance(v, (dict, list, str, int, float, bool))
        }
        return json.dumps(safe, indent=2, ensure_ascii=False)

    @staticmethod
    def to_csv(rows: list[dict]) -> str:
        if not rows:
            return ""
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
        return buf.getvalue()

    @staticmethod
    def to_pdf(results: dict, text: str) -> bytes:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Text Analysis Report", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.ln(4)

        # Readability
        if "readability" in results:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Readability Scores", ln=True)
            pdf.set_font("Helvetica", size=10)
            for k, v in results["readability"].items():
                pdf.cell(0, 6, f"  {k}: {v}", ln=True)
            pdf.ln(4)

        # Sentiment
        if "sentiment" in results:
            pol, sub, label = results["sentiment"]
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Overall Sentiment", ln=True)
            pdf.set_font("Helvetica", size=10)
            pdf.cell(
                0,
                6,
                f"  Polarity: {pol:.3f}  Subjectivity: {sub:.3f}  Label: {label}",
                ln=True,
            )
            pdf.ln(4)

        # Keywords
        if "keywords" in results:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Top Keywords", ln=True)
            pdf.set_font("Helvetica", size=10)
            for word, freq in results["keywords"]:
                pdf.cell(0, 6, f"  {word}: {freq}", ln=True)
            pdf.ln(4)

        # Text excerpt
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Text (first 1000 chars)", ln=True)
        pdf.set_font("Helvetica", size=9)
        excerpt = text[:1000].replace("\n", " ")
        pdf.multi_cell(0, 5, excerpt)

        return pdf.output()


# Page renderers

app = TextAnalysisApp()


def page_analysis():
    st.header("📝 Text Analysis")

    # ── Input ────────────────────────────────────────────────────────────────
    col_up, col_txt = st.columns([1, 2])
    with col_up:
        uploaded = st.file_uploader("Upload .txt file", type=["txt"])
    with col_txt:
        default_txt = uploaded.getvalue().decode("utf-8") if uploaded else ""
        text = st.text_area("…or paste text here:", value=default_txt, height=180)

    if st.button("🔍 Analyse", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please provide some text first.")
            return
        with st.spinner("Running NLP pipeline…"):
            doc = app.nlp(text)
            pol, sub, label = app.get_sentiment(text)
            keywords = app.extract_keywords(text)
            sent_rows = app.sentence_sentiments(text)
            readability = app.readability_scores(text)

            st.session_state.update(
                {
                    "doc": doc,
                    "text": text,
                    "analysis_done": True,
                    "results": {
                        "sentiment": (pol, sub, label),
                        "keywords": keywords,
                        "sent_rows": sent_rows,
                        "readability": readability,
                        "pos_data": [
                            (t.text, t.pos_, t.dep_, t.head.text) for t in doc
                        ],
                    },
                }
            )

    if not st.session_state.analysis_done:
        st.info("Enter text and click **Analyse** to begin.")
        return

    doc = st.session_state.doc
    text = st.session_state.text
    res = st.session_state.results
    pol, sub, label = res["sentiment"]

    # ── Metric strip ─────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sentiment", label, f"{pol:+.2f} polarity")
    m2.metric("Subjectivity", f"{sub:.2f}", "0=objective · 1=subjective")
    m3.metric("Flesch Ease", res["readability"]["Flesch Reading Ease"])
    m4.metric("FK Grade", res["readability"]["Flesch-Kincaid Grade"])

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tabs = st.tabs(
        [
            "🏷️ Entities",
            "📊 Keywords",
            "😊 Sentiment",
            "🔗 Dependency",
            "☁️ Word Cloud",
            "📖 Readability",
            "🔬 POS Table",
        ]
    )

    # ── Entities ─────────────────────────────────────────────────────────────
    with tabs[0]:
        sel_ents = st.multiselect(
            "Filter entity types",
            app.ENTITY_TYPES,
            default=app.ENTITY_TYPES,
            key="ent_filter",
        )
        st.subheader("Annotated Text")
        parts = app.render_annotated(doc, sel_ents)
        ann_text(*parts)

        if doc.ents:
            st.subheader("Entity Distribution")
            fig = app.plotly_entity_pie(doc.ents, sel_ents)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Entity Table")
            ent_rows = [
                {
                    "Text": e.text,
                    "Label": e.label_,
                    "Start": e.start_char,
                    "End": e.end_char,
                }
                for e in doc.ents
                if e.label_ in sel_ents
            ]
            if ent_rows:
                gb = GridOptionsBuilder.from_dataframe(
                    __import__("pandas").DataFrame(ent_rows)
                )
                gb.configure_pagination(paginationAutoPageSize=True)
                gb.configure_default_column(filter=True, sortable=True)
                AgGrid(
                    __import__("pandas").DataFrame(ent_rows),
                    gridOptions=gb.build(),
                    update_mode=GridUpdateMode.NO_UPDATE,
                    height=300,
                    fit_columns_on_grid_load=True,
                )

    # ── Keywords ─────────────────────────────────────────────────────────────
    with tabs[1]:
        top_n = st.slider("Top N keywords", 5, 30, 15)
        kws = app.extract_keywords(text, top_n)
        st.plotly_chart(app.plotly_keyword_bar(kws), use_container_width=True)

        import pandas as pd

        kw_df = pd.DataFrame(kws, columns=["Keyword", "Frequency"])
        gb2 = GridOptionsBuilder.from_dataframe(kw_df)
        gb2.configure_default_column(sortable=True, filter=True)
        AgGrid(
            kw_df,
            gridOptions=gb2.build(),
            update_mode=GridUpdateMode.NO_UPDATE,
            height=300,
            fit_columns_on_grid_load=True,
        )

    # ── Sentiment ─────────────────────────────────────────────────────────────
    with tabs[2]:
        st.plotly_chart(
            app.plotly_sentiment_scatter(res["sent_rows"]), use_container_width=True
        )
        import pandas as pd

        sdf = pd.DataFrame(res["sent_rows"])
        gb3 = GridOptionsBuilder.from_dataframe(sdf)
        gb3.configure_pagination(paginationAutoPageSize=True)
        gb3.configure_default_column(filter=True, sortable=True, resizable=True)
        gb3.configure_column("Sentence", wrapText=True, autoHeight=True)
        AgGrid(
            sdf,
            gridOptions=gb3.build(),
            update_mode=GridUpdateMode.NO_UPDATE,
            height=350,
            fit_columns_on_grid_load=True,
        )

    # ── Dependency ────────────────────────────────────────────────────────────
    with tabs[3]:
        viz_style = st.radio(
            "displaCy style", ["dep", "ent"], horizontal=True, key="dep_radio"
        )
        html = displacy.render(doc, style=viz_style, jupyter=False)
        st.write(f"<div style='overflow-x:auto'>{html}</div>", unsafe_allow_html=True)
        st.subheader("Interactive Dependency Graph")
        st.plotly_chart(app.plotly_dependency_graph(doc), use_container_width=True)

    # ── Word Cloud ────────────────────────────────────────────────────────────
    with tabs[4]:
        st.pyplot(app.wordcloud_figure(text))

    # ── Readability ───────────────────────────────────────────────────────────
    with tabs[5]:
        st.plotly_chart(
            app.plotly_readability_gauge(res["readability"]), use_container_width=True
        )
        import pandas as pd

        r_df = pd.DataFrame(res["readability"].items(), columns=["Metric", "Value"])
        gb5 = GridOptionsBuilder.from_dataframe(r_df)
        gb5.configure_default_column(sortable=True)
        AgGrid(
            r_df,
            gridOptions=gb5.build(),
            update_mode=GridUpdateMode.NO_UPDATE,
            height=260,
            fit_columns_on_grid_load=True,
        )

        st.markdown("""
**Flesch Reading Ease guide**
| Score | Level |
|-------|-------|
| 90–100 | Very Easy |
| 70–90  | Easy |
| 60–70  | Standard |
| 30–60  | Difficult |
| 0–30   | Very Difficult |
""")

    # ── POS Table ─────────────────────────────────────────────────────────────
    with tabs[6]:
        import pandas as pd

        pos_df = pd.DataFrame(res["pos_data"], columns=["Token", "POS", "Dep", "Head"])
        gb6 = GridOptionsBuilder.from_dataframe(pos_df)
        gb6.configure_pagination(paginationAutoPageSize=True)
        gb6.configure_default_column(filter=True, sortable=True)
        AgGrid(
            pos_df,
            gridOptions=gb6.build(),
            update_mode=GridUpdateMode.NO_UPDATE,
            height=400,
            fit_columns_on_grid_load=True,
        )


def page_export():
    st.header("📤 Export Results")
    if not st.session_state.analysis_done:
        st.warning("Run an analysis first on the **Text Analysis** page.")
        return

    res = st.session_state.results
    text = st.session_state.text

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📄 JSON")
        json_str = app.to_json(
            {
                "readability": res["readability"],
                "sentiment": {
                    "polarity": res["sentiment"][0],
                    "subjectivity": res["sentiment"][1],
                    "label": res["sentiment"][2],
                },
                "keywords": res["keywords"],
            }
        )
        st.download_button(
            "⬇️ Download JSON",
            json_str,
            "analysis.json",
            "application/json",
            use_container_width=True,
        )
        with st.expander("Preview"):
            st.code(json_str, language="json")

    with col2:
        st.subheader("📊 CSV (Sentences)")
        csv_str = app.to_csv(res["sent_rows"])
        st.download_button(
            "⬇️ Download CSV",
            csv_str,
            "sentiments.csv",
            "text/csv",
            use_container_width=True,
        )
        with st.expander("Preview"):
            st.code(csv_str[:800])

    with col3:
        st.subheader("🖨️ PDF Report")
        pdf_bytes = app.to_pdf(res, text)
        st.download_button(
            "⬇️ Download PDF",
            pdf_bytes,
            "report.pdf",
            "application/pdf",
            use_container_width=True,
        )
        st.caption("Includes readability, sentiment, keywords and text excerpt.")


def page_about():
    st.header("ℹ️ About")
    st.markdown("""
This app provides a full NLP pipeline for English text using:

| Tool | Purpose |
|------|---------|
| **spaCy** `en_core_web_sm` | Tokenisation, POS, NER, dependency parsing |
| **TextBlob** | Sentiment polarity & subjectivity |
| **NLTK** | Stopword removal, sentence tokenisation |
| **Plotly** | Interactive charts |
| **AG Grid** | Sortable / filterable data tables |
| **annotated-text** | Inline entity highlighting |
| **WordCloud** | Word frequency visualisation |
| **fpdf2** | PDF report generation |

### Setup (uv)
```bash
uv init nlp-app && cd nlp-app
uv add streamlit spacy textblob nltk plotly networkx wordcloud \\
       matplotlib seaborn streamlit-aggrid annotated-text fpdf2
uv run python -m spacy download en_core_web_sm
uv run streamlit run app.py
```
""")


# Sidebar navigation

st.sidebar.title("🔍 NLP Analyser")
st.sidebar.divider()
page = st.sidebar.radio(
    "Navigate",
    ["📝 Text Analysis", "📤 Export Results", "ℹ️ About"],
    label_visibility="collapsed",
)

if page == "📝 Text Analysis":
    page_analysis()
elif page == "📤 Export Results":
    page_export()
elif page == "ℹ️ About":
    page_about()
