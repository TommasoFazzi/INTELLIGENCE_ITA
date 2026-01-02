"""
Daily Briefing - Report Editor with Split View

Refactored editor for reviewing and approving intelligence reports.
Features:
- Split View: Original AI draft vs Human Editor
- Sidebar Navigator: Date filter + selectbox
- Ticker Detection: Whitelist-based highlighting
- Compact Header: Key metrics at a glance
"""

import sys
from pathlib import Path
from datetime import datetime

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

from src.hitl.streamlit_utils import (
    get_db_manager,
    inject_custom_css,
    get_status_badge,
    get_status_emoji,
    extract_tickers,
    init_session_state
)
from src.llm.report_generator import ReportGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Page Configuration
st.set_page_config(
    page_title="Daily Briefing | INTELLIGENCE_ITA",
    page_icon="📝",
    layout="wide"
)

# Inject CSS and initialize
inject_custom_css()
init_session_state()
db = get_db_manager()


# =============================================================================
# SIDEBAR NAVIGATOR
# =============================================================================

def sidebar_navigator():
    """Render sidebar with date filter and report selector."""
    st.sidebar.title("🗂️ Archivio Report")

    # Date filter
    selected_date = st.sidebar.date_input(
        "Filtra per data",
        datetime.now(),
        help="Seleziona una data per vedere i report di quel giorno"
    )

    # Fetch reports
    try:
        all_reports = db.get_all_reports(limit=100)
    except Exception as e:
        st.sidebar.error(f"Errore DB: {e}")
        all_reports = []

    # Filter by selected date
    filtered_reports = []
    for r in all_reports:
        report_date = r.get('report_date')
        if report_date:
            if hasattr(report_date, 'strftime'):
                report_date_str = report_date.strftime('%Y-%m-%d')
            else:
                report_date_str = str(report_date)[:10]

            if report_date_str == selected_date.strftime('%Y-%m-%d'):
                filtered_reports.append(r)

    # Report selector
    if filtered_reports:
        options = {}
        for r in filtered_reports:
            emoji = get_status_emoji(r.get('status', 'draft'))
            options[r['id']] = f"{emoji} ID {r['id']} | {r.get('status', 'draft').upper()}"

        selected_id = st.sidebar.selectbox(
            "Seleziona Report",
            options=list(options.keys()),
            format_func=lambda x: options[x],
            index=0
        )

        if selected_id:
            st.session_state.current_report_id = selected_id
    else:
        st.sidebar.warning(f"Nessun report per il {selected_date.strftime('%d/%m/%Y')}")

        # Show all reports option
        if st.sidebar.checkbox("Mostra tutti i report"):
            if all_reports:
                options = {}
                for r in all_reports[:20]:  # Limit to 20
                    emoji = get_status_emoji(r.get('status', 'draft'))
                    report_date = r.get('report_date', datetime.now())
                    if hasattr(report_date, 'strftime'):
                        date_str = report_date.strftime('%d/%m')
                    else:
                        date_str = str(report_date)[:10]
                    options[r['id']] = f"{emoji} {date_str} | ID {r['id']}"

                selected_id = st.sidebar.selectbox(
                    "Tutti i Report",
                    options=list(options.keys()),
                    format_func=lambda x: options[x]
                )

                if selected_id:
                    st.session_state.current_report_id = selected_id

    st.sidebar.divider()

    # Generate new report button
    if st.sidebar.button("➕ Genera Nuovo Report", type="primary", width="stretch"):
        generate_new_report()

    # Back to War Room
    st.sidebar.divider()
    if st.sidebar.button("🏠 Torna alla War Room", width="stretch"):
        st.switch_page("Home.py")


def generate_new_report():
    """Generate a new intelligence report."""
    with st.spinner("🚀 Avvio Intelligence Engine..."):
        try:
            generator = ReportGenerator()

            # Use project's standard focus areas
            focus_areas = [
                "cybersecurity threats, state-sponsored cyber attacks, ransomware campaigns",
                "breakthroughs in artificial intelligence, semiconductor supply chain shifts",
                "escalation of military conflicts, diplomatic ruptures, NATO, Russia, China",
                "territorial control changes, strategic military movements, maritime security",
                "global economic impact of sanctions, energy market volatility"
            ]

            report = generator.generate_report(
                focus_areas=focus_areas,
                days=1,
                rag_top_k=5
            )

            if report.get('success'):
                report_id = db.save_report(report)
                if report_id:
                    st.session_state.current_report_id = report_id
                    st.success(f"Report generato! ID: {report_id}")
                    st.rerun()
                else:
                    st.error("Errore nel salvataggio del report")
            else:
                st.error(f"Errore generazione: {report.get('error', 'Unknown')}")

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            st.error(f"Errore: {e}")


# =============================================================================
# MAIN EDITOR INTERFACE
# =============================================================================

def editor_interface():
    """Render the main editor with split view."""
    if not st.session_state.current_report_id:
        render_welcome_screen()
        return

    # Fetch report
    try:
        report = db.get_report(st.session_state.current_report_id)
    except Exception as e:
        st.error(f"Errore nel recupero del report: {e}")
        return

    if not report:
        st.error("Report non trovato.")
        st.session_state.current_report_id = None
        return

    # Header with metrics
    render_compact_header(report)

    st.divider()

    # Split View
    render_split_view(report)

    st.divider()

    # Action bar
    render_action_bar(report)


def render_welcome_screen():
    """Render welcome screen when no report is selected."""
    st.title("📝 Daily Briefing")

    st.info("""
    👈 **Seleziona un report dalla sidebar** o generane uno nuovo.

    **Workflow:**
    1. Seleziona una data per filtrare i report
    2. Scegli un report dal menu a tendina
    3. Rivedi la bozza AI nella colonna sinistra
    4. Modifica nella colonna destra (Editor)
    5. Approva quando soddisfatto
    """)

    # Quick stats
    try:
        stats = db.get_statistics()
        col1, col2, col3 = st.columns(3)
        col1.metric("Articoli", stats.get('total_articles', 0))
        col2.metric("Chunks", stats.get('total_chunks', 0))
        col3.metric("Recenti (7gg)", stats.get('recent_articles', 0))
    except Exception:
        pass


def render_compact_header(report: dict):
    """Render compact header with key metrics."""
    st.title("📝 Daily Briefing")

    # Extract data
    status = report.get('status', 'draft')
    metadata = report.get('metadata', {})
    report_date = report.get('report_date', datetime.now())

    if hasattr(report_date, 'strftime'):
        date_str = report_date.strftime('%d/%m/%Y')
    else:
        date_str = str(report_date)[:10]

    draft_content = report.get('draft_content', '')
    detected_tickers = extract_tickers(draft_content)

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(get_status_badge(status), unsafe_allow_html=True)

    with c2:
        articles_count = metadata.get('recent_articles_count', 0)
        st.metric("Articoli Analizzati", articles_count)

    with c3:
        st.metric("Data Report", date_str)

    with c4:
        st.metric(
            "Tickers Rilevati",
            len(detected_tickers),
            help=", ".join(detected_tickers[:10]) if detected_tickers else "Nessuno"
        )


def render_split_view(report: dict):
    """Render split view: Original vs Editor."""
    draft_content = report.get('draft_content', '')
    final_content = report.get('final_content') or draft_content

    col_orig, col_edit = st.columns([1, 1])

    with col_orig:
        st.subheader("📄 Originale (AI)")
        st.markdown(
            f"""<div style='
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                height: 500px;
                overflow-y: auto;
                border: 1px solid #dee2e6;
                font-size: 14px;
                line-height: 1.6;
            '>{draft_content}</div>""",
            unsafe_allow_html=True
        )

    with col_edit:
        st.subheader("✏️ Editor (Versione Finale)")
        edited_text = st.text_area(
            "Modifica il report",
            value=final_content,
            height=500,
            label_visibility="collapsed",
            key="editor_textarea"
        )

        # Store in session state for save
        st.session_state.edited_content = edited_text


def render_action_bar(report: dict):
    """Render action bar with save/approve buttons."""
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        reviewer_name = st.text_input(
            "Nome Revisore",
            value="Human Lead",
            key="reviewer_name"
        )

    with col2:
        rating = st.slider(
            "Qualità (1-5)",
            min_value=1,
            max_value=5,
            value=3,
            key="quality_rating"
        )

    with col3:
        feedback_comment = st.text_input(
            "Note",
            placeholder="Commenti opzionali...",
            key="feedback_comment"
        )

    # Action buttons
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])

    with btn_col1:
        if st.button("💾 Salva Bozza", width="stretch"):
            save_report(report['id'], 'reviewed', reviewer_name, rating, feedback_comment)

    with btn_col2:
        if st.button("✅ Approva", type="primary", width="stretch"):
            save_report(report['id'], 'approved', reviewer_name, rating, feedback_comment)

    with btn_col3:
        # Word count comparison
        draft_len = len(report.get('draft_content', '').split())
        edited_len = len(st.session_state.get('edited_content', '').split())
        diff = edited_len - draft_len
        diff_str = f"+{diff}" if diff > 0 else str(diff)

        st.caption(f"📊 Parole: {draft_len} → {edited_len} ({diff_str})")


def save_report(report_id: int, status: str, reviewer: str, rating: int, comment: str):
    """Save the edited report."""
    edited_content = st.session_state.get('edited_content', '')

    if not edited_content:
        st.error("Nessun contenuto da salvare")
        return

    try:
        # Update report
        success = db.update_report(
            report_id=report_id,
            final_content=edited_content,
            status=status,
            reviewer=reviewer
        )

        if success:
            # Save feedback
            try:
                db.upsert_approval_feedback(
                    report_id=report_id,
                    rating=rating,
                    comment=comment
                )
            except Exception as e:
                logger.warning(f"Failed to save feedback: {e}")

            st.toast(f"Report {'approvato' if status == 'approved' else 'salvato'}!", icon="✅")

            if status == 'approved':
                st.balloons()

            st.rerun()
        else:
            st.error("Errore nel salvataggio")

    except Exception as e:
        logger.error(f"Save failed: {e}")
        st.error(f"Errore: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    sidebar_navigator()
    editor_interface()


if __name__ == "__main__":
    main()
