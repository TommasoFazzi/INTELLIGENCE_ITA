"""
Streamlit Dashboard for Human-in-the-Loop Report Review

This dashboard allows human reviewers to:
1. View LLM-generated intelligence reports
2. Edit and correct report content
3. Rate report quality
4. Approve final versions
5. Provide feedback for improving future reports
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.storage.database import DatabaseManager
from src.llm.report_generator import ReportGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Page configuration
st.set_page_config(
    page_title="Intelligence Report Review Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.3rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-draft {
        background-color: #fff3cd;
        color: #856404;
    }
    .status-reviewed {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    .status-approved {
        background-color: #d4edda;
        color: #155724;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .feedback-box {
        background-color: #fff9e6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'db' not in st.session_state:
        st.session_state.db = DatabaseManager()

    if 'current_report_id' not in st.session_state:
        st.session_state.current_report_id = None

    if 'edited_content' not in st.session_state:
        st.session_state.edited_content = ""

    if 'feedback_list' not in st.session_state:
        st.session_state.feedback_list = []

    if 'show_success_message' not in st.session_state:
        st.session_state.show_success_message = None


def generate_new_report() -> Optional[int]:
    """Generate a new intelligence report."""
    try:
        with st.spinner("Generazione report in corso... (10-20 secondi)"):
            generator = ReportGenerator()

            # Generate report
            report = generator.generate_report(
                focus_areas=[
                    "cybersecurity threats, data breaches, and vulnerabilities",
                    "artificial intelligence and technology developments",
                    "geopolitical tensions and conflicts",
                    "economic policy changes and market trends"
                ],
                days=1,
                rag_top_k=5
            )

            if not report['success']:
                st.error(f"Errore nella generazione del report: {report.get('error')}")
                return None

            # Save to database
            report_id = st.session_state.db.save_report(report)

            if report_id:
                st.success(f"✓ Report generato con successo (ID: {report_id})")
                return report_id
            else:
                st.error("Errore nel salvataggio del report al database")
                return None

    except Exception as e:
        st.error(f"Errore: {str(e)}")
        logger.error(f"Error generating report: {e}", exc_info=True)
        return None


def export_approved_report_to_file(report_date: datetime, final_content: str, reviewer: str = None) -> Optional[Path]:
    """
    Export approved report to markdown file, overwriting the original draft.

    Args:
        report_date: Date of the report
        final_content: Approved report content
        reviewer: Name of the reviewer who approved it

    Returns:
        Path to the exported file or None if failed
    """
    try:
        # Format date for filename
        date_str = report_date.strftime("%Y%m%d")

        # Find the original report file
        reports_dir = project_root / "reports"
        reports_dir.mkdir(exist_ok=True)

        # Look for existing report file with this date
        pattern = f"intelligence_report_{date_str}_*.md"
        existing_files = list(reports_dir.glob(pattern))

        if existing_files:
            # Use the first matching file (should only be one per day)
            output_file = existing_files[0]
            logger.info(f"Overwriting existing report file: {output_file}")
        else:
            # Create new file if no existing one found
            timestamp = report_date.strftime("%H%M%S")
            output_file = reports_dir / f"intelligence_report_{date_str}_{timestamp}.md"
            logger.info(f"Creating new report file: {output_file}")

        # Prepare approved header
        approval_header = "---\n"
        approval_header += "**STATUS: APPROVATO**\n"
        if reviewer:
            approval_header += f"**Revisore**: {reviewer}\n"
        approval_header += f"**Data Approvazione**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        approval_header += "---\n\n"

        # Write approved content
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(approval_header)
            f.write(final_content)

        logger.info(f"✓ Approved report exported to: {output_file}")
        return output_file

    except Exception as e:
        logger.error(f"Error exporting approved report: {e}", exc_info=True)
        return None


def display_report_selector():
    """Display sidebar for selecting reports."""
    st.sidebar.markdown("## 📋 Gestione Report")
    
    # Generate new report button
    if st.sidebar.button("➕ Genera Nuovo Report", use_container_width=True):
        report_id = generate_new_report()
        if report_id:
            st.session_state.current_report_id = report_id
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Get all reports
    reports = st.session_state.db.get_all_reports(limit=20)
    
    if not reports:
        st.sidebar.info("Nessun report disponibile. Genera il primo report!")
        return
    
    # Display reports list
    st.sidebar.markdown("### Report Disponibili")
    
    for report in reports:
        report_date = report['report_date']
        if hasattr(report_date, 'strftime'):
            date_str = report_date.strftime('%Y-%m-%d')
        else:
            date_str = str(report_date)
        
        status = report['status']
        status_icon = {
            'draft': '📝',
            'reviewed': '👁️',
            'approved': '✅'
        }.get(status, '❓')
        
        # Article count from metadata
        article_count = report.get('metadata', {}).get('recent_articles_count', 0)
        
        # Button to select report
        button_label = f"{status_icon} {date_str} ({article_count} articoli)"
        if st.sidebar.button(button_label, key=f"select_{report['id']}", use_container_width=True):
            st.session_state.current_report_id = report['id']
            st.rerun()


def get_status_badge(status: str) -> str:
    """Get HTML badge for report status."""
    status_map = {
        'draft': ('Draft', 'status-draft'),
        'reviewed': ('Revisionato', 'status-reviewed'),
        'approved': ('Approvato', 'status-approved')
    }
    label, css_class = status_map.get(status, (status, 'status-draft'))
    return f'<span class="status-badge {css_class}">{label}</span>'


def display_report_viewer(report: Dict[str, Any]):
    """Display report viewer and editor."""

    # Display success message if present
    if st.session_state.show_success_message:
        msg = st.session_state.show_success_message
        if msg['type'] == 'approved':
            st.success("✅ **Report approvato e salvato con successo!**", icon="✅")
        else:
            st.info("💾 **Report salvato come bozza revisionata**", icon="💾")
        # Clear message after display
        st.session_state.show_success_message = None

    # Report header
    report_date = report['report_date']
    if hasattr(report_date, 'strftime'):
        date_str = report_date.strftime('%d %B %Y')
    else:
        date_str = str(report_date)
    
    st.markdown(f'<div class="main-header">Intelligence Report - {date_str}</div>', unsafe_allow_html=True)
    
    # Status badge
    st.markdown(get_status_badge(report['status']), unsafe_allow_html=True)
    
    # Metadata
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Articoli Analizzati", report.get('metadata', {}).get('recent_articles_count', 0))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Context Storico", report.get('metadata', {}).get('historical_chunks_count', 0))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        model = report.get('model_used', 'N/A')
        st.metric("Modello LLM", model.replace('gemini-', ''))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        gen_time = report.get('generated_at')
        if hasattr(gen_time, 'strftime'):
            time_str = gen_time.strftime('%H:%M')
        else:
            time_str = 'N/A'
        st.metric("Generato alle", time_str)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")

    # Tabs for draft vs final
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Bozza LLM", "✏️ Versione Finale", "📊 Fonti", "💬 Feedback"])
    
    with tab1:
        st.markdown('<div class="section-header">Bozza Originale (Generata da LLM)</div>', unsafe_allow_html=True)
        st.markdown(report['draft_content'])
    
    with tab2:
        st.markdown('<div class="section-header">Modifica Report</div>', unsafe_allow_html=True)
        
        # Initialize edited content
        if st.session_state.current_report_id != report['id']:
            # Reset edited content when switching reports
            st.session_state.edited_content = report.get('final_content') or report['draft_content']
            st.session_state.current_report_id = report['id']
        
        # Editable text area
        edited_content = st.text_area(
            "Modifica il report qui sotto:",
            value=st.session_state.edited_content,
            height=500,
            key="report_editor"
        )
        
        # Update session state
        st.session_state.edited_content = edited_content
        
        # Rating and reviewer
        col1, col2 = st.columns([1, 3])
        
        with col1:
            rating = st.slider("Valutazione Qualità", 1, 5, 3, help="1 = Scarso, 5 = Eccellente")
        
        with col2:
            reviewer_name = st.text_input("Nome Revisore (opzionale)", value=os.getenv('USER', ''))
        
        # Feedback comment
        feedback_comment = st.text_area(
            "Note / Commenti (opzionale)",
            placeholder="Inserisci note sulle modifiche effettuate, suggerimenti per migliorare i futuri report, ecc.",
            height=100
        )
        
        # Action buttons
        col1, col2, col3 = st.columns([2, 2, 6])
        
        with col1:
            if st.button("💾 Salva Bozza", use_container_width=True):
                success = st.session_state.db.update_report(
                    report_id=report['id'],
                    final_content=edited_content,
                    status='reviewed',
                    reviewer=reviewer_name or None
                )

                if success:
                    # Set success message for display after rerun
                    st.session_state.show_success_message = {
                        'type': 'saved',
                        'timestamp': datetime.now()
                    }
                    st.rerun()
                else:
                    st.error("Errore nel salvataggio del report")
        
        with col2:
            if st.button("✅ Approva", use_container_width=True):
                success = st.session_state.db.update_report(
                    report_id=report['id'],
                    final_content=edited_content,
                    status='approved',
                    reviewer=reviewer_name or None
                )

                if success:
                    # Upsert approval feedback (update if exists, insert if not)
                    st.session_state.db.upsert_approval_feedback(
                        report_id=report['id'],
                        rating=rating,
                        comment=feedback_comment or "Report approvato"
                    )

                    # Export approved report to file
                    exported_file = export_approved_report_to_file(
                        report_date=report['report_date'],
                        final_content=edited_content,
                        reviewer=reviewer_name
                    )

                    if exported_file:
                        logger.info(f"✓ Report approvato ed esportato: {exported_file}")
                    else:
                        logger.warning("⚠ Report approvato ma export fallito")

                    # Set success message for display after rerun
                    st.session_state.show_success_message = {
                        'type': 'approved',
                        'timestamp': datetime.now()
                    }
                    st.balloons()
                    st.rerun()
                else:
                    st.error("Errore nell'approvazione del report")
        
        # Show differences
        if edited_content != report['draft_content']:
            st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
            st.warning(f"⚠️ Hai apportato modifiche alla bozza originale")
            
            # Calculate basic diff stats
            original_words = len(report['draft_content'].split())
            edited_words = len(edited_content.split())
            word_diff = edited_words - original_words
            
            st.caption(f"Parole: {original_words} → {edited_words} ({word_diff:+d})")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        # Sources
        st.markdown('<div class="section-header">Fonti Utilizzate</div>', unsafe_allow_html=True)

        sources = report.get('sources', {})

        # Recent articles
        st.markdown("#### Articoli Recenti")
        recent_articles = sources.get('recent_articles', [])

        if recent_articles:
            total_articles = len(recent_articles)
            st.info(f"📚 Totale: {total_articles} articoli utilizzati")

            # Toggle per espandere/collassare tutti
            col_btn1, col_btn2 = st.columns([1, 5])
            with col_btn1:
                if st.button("📖 Espandi Tutti", key="expand_all_articles", use_container_width=True):
                    st.session_state.articles_expanded = True
                    st.rerun()
            with col_btn2:
                if st.button("📕 Riduci Tutti", key="collapse_all_articles", use_container_width=True):
                    st.session_state.articles_expanded = False
                    st.rerun()

            # Stato di espansione (default: primi 3 espansi)
            if 'articles_expanded' not in st.session_state:
                st.session_state.articles_expanded = None

            # Mostra tutti gli articoli
            for i, article in enumerate(recent_articles, 1):
                # Determina se questo articolo deve essere espanso
                if st.session_state.articles_expanded is True:
                    is_expanded = True
                elif st.session_state.articles_expanded is False:
                    is_expanded = False
                else:
                    # Default: primi 3 espansi
                    is_expanded = (i <= 3)

                with st.expander(f"[{i}] {article['title']}", expanded=is_expanded):
                    st.markdown(f"**Fonte:** {article['source']}")
                    st.markdown(f"**Data:** {article['published_date']}")
                    st.markdown(f"**Link:** [{article['link']}]({article['link']})")
        else:
            st.info("Nessun articolo recente trovato")

        st.markdown("---")

        # Historical context
        st.markdown("#### Context Storico (RAG)")
        historical = sources.get('historical_context', [])

        if historical:
            total_chunks = len(historical)
            st.info(f"🧠 Totale: {total_chunks} chunk storici recuperati via RAG")

            # Toggle per espandere/collassare tutti
            col_btn1, col_btn2 = st.columns([1, 5])
            with col_btn1:
                if st.button("📖 Espandi Tutti", key="expand_all_chunks", use_container_width=True):
                    st.session_state.chunks_expanded = True
                    st.rerun()
            with col_btn2:
                if st.button("📕 Riduci Tutti", key="collapse_all_chunks", use_container_width=True):
                    st.session_state.chunks_expanded = False
                    st.rerun()

            # Stato di espansione (default: primi 2 espansi)
            if 'chunks_expanded' not in st.session_state:
                st.session_state.chunks_expanded = None

            # Mostra tutti i chunk
            for i, ctx in enumerate(historical, 1):
                similarity = ctx.get('similarity', 0)
                sim_percent = f"{similarity * 100:.1f}%"

                # Determina se questo chunk deve essere espanso
                if st.session_state.chunks_expanded is True:
                    is_expanded = True
                elif st.session_state.chunks_expanded is False:
                    is_expanded = False
                else:
                    # Default: primi 2 espansi
                    is_expanded = (i <= 2)

                with st.expander(f"[{i}] {ctx['title']} (Similarità: {sim_percent})", expanded=is_expanded):
                    st.markdown(f"**Link:** [{ctx['link']}]({ctx['link']})")
                    st.markdown(f"**Contenuto Chunk:**")
                    st.markdown(f"> {ctx.get('content', 'N/A')[:300]}...")
                    st.progress(similarity)
        else:
            st.info("Nessun context storico utilizzato")

    with tab4:
        display_feedback_tab(report, st.session_state.db)


def display_feedback_tab(report: Dict[str, Any], db: DatabaseManager):
    """
    Display comprehensive feedback view with report-specific and global feedback.

    Args:
        report: Current report dictionary
        db: Database manager instance
    """
    st.markdown("### 📋 Feedback di Questo Report")

    # Get feedback for current report
    current_feedback = db.get_report_feedback(report['id'])

    if current_feedback:
        for fb in current_feedback:
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"⭐ Rating: {fb['rating']}/5")
                with col2:
                    reviewer = fb.get('reviewer', 'Anonimo')
                    st.write(f"👤 {reviewer}")
                with col3:
                    created = fb['created_at']
                    if hasattr(created, 'strftime'):
                        time_str = created.strftime('%d/%m %H:%M')
                    else:
                        time_str = str(created)
                    st.write(f"🕐 {time_str}")

                if fb.get('comment'):
                    st.info(fb['comment'])
                st.divider()
    else:
        st.info("Nessun feedback ancora per questo report")

    st.markdown("---")
    st.markdown("### 🌐 Ultimi Feedback Globali (tutti i report)")

    # Get recent global feedback
    global_feedback = db.get_recent_feedback(limit=10)

    if global_feedback:
        for fb in global_feedback:
            # Format report date
            report_date = fb['report_date']
            if hasattr(report_date, 'strftime'):
                date_str = report_date.strftime('%d/%m/%Y')
            else:
                date_str = str(report_date)

            reviewer = fb.get('reviewer', 'Anonimo')

            with st.expander(
                f"Report {date_str} - ⭐ {fb['rating']}/5 - {reviewer}"
            ):
                created = fb['created_at']
                if hasattr(created, 'strftime'):
                    created_str = created.strftime('%d/%m/%Y %H:%M')
                else:
                    created_str = str(created)
                st.write(f"**Data creazione**: {created_str}")

                if fb.get('comment'):
                    # Show first 200 chars
                    comment_preview = fb['comment'][:200]
                    if len(fb['comment']) > 200:
                        comment_preview += "..."
                    st.write(f"**Commento**: {comment_preview}")

                # Link to open that report
                if st.button(f"📂 Apri Report", key=f"open_report_fb_{fb['id']}"):
                    st.session_state.current_report_id = fb['report_id']
                    st.rerun()
    else:
        st.info("Nessun feedback disponibile nel sistema")


def display_statistics():
    """Display database statistics in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 Statistiche Database")
    
    stats = st.session_state.db.get_statistics()
    
    if stats:
        st.sidebar.metric("Totale Articoli", stats.get('total_articles', 0))
        st.sidebar.metric("Totale Chunks", stats.get('total_chunks', 0))
        st.sidebar.metric("Articoli Recenti (7gg)", stats.get('recent_articles', 0))
        
        # Reports stats
        all_reports = st.session_state.db.get_all_reports(limit=100)
        if all_reports:
            st.sidebar.metric("Report Generati", len(all_reports))
            
            # Count by status
            status_counts = {}
            for r in all_reports:
                status = r['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            st.sidebar.caption(f"Draft: {status_counts.get('draft', 0)} | "
                             f"Revisionati: {status_counts.get('reviewed', 0)} | "
                             f"Approvati: {status_counts.get('approved', 0)}")


def main():
    """Main dashboard application."""
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    display_report_selector()
    display_statistics()
    
    # Main content
    if st.session_state.current_report_id is None:
        # Welcome screen
        st.markdown('<div class="main-header">Intelligence Report Review Dashboard</div>', unsafe_allow_html=True)
        st.markdown("""
        ### Benvenuto al Sistema di Revisione Report
        
        Questa dashboard ti permette di:
        - 📝 **Generare** nuovi report intelligence con LLM + RAG
        - 👁️ **Revisionare** le bozze generate automaticamente
        - ✏️ **Modificare** e correggere il contenuto
        - ✅ **Approvare** le versioni finali
        - 💬 **Fornire feedback** per migliorare i futuri report
        
        #### Come Iniziare
        
        1. Clicca su **"➕ Genera Nuovo Report"** nella barra laterale
        2. Attendi 10-20 secondi per la generazione
        3. Rivedi la bozza generata dall'LLM
        4. Apporta modifiche se necessario
        5. Salva come revisionato o approva direttamente
        
        #### Workflow HITL (Human-in-the-Loop)
        
        ```
        LLM genera bozza → Umano revisiona → Salva feedback → 
        Report approvato → Feedback usato per migliorare futuri prompt
        ```
        
        Il feedback che fornisci (modifiche, commenti, valutazioni) viene salvato nel database
        e può essere usato per:
        - Analizzare quali sezioni richiedono più correzioni
        - Migliorare i prompt per l'LLM
        - Creare esempi di "ground truth" per fine-tuning
        """)
        
        # Quick stats
        stats = st.session_state.db.get_statistics()
        if stats.get('total_articles', 0) > 0:
            st.markdown("---")
            st.markdown("### 📊 Dati Disponibili")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**{stats.get('total_articles', 0)}** articoli nel database")
            with col2:
                st.info(f"**{stats.get('total_chunks', 0)}** chunks per RAG")
            with col3:
                st.info(f"**{stats.get('recent_articles', 0)}** articoli ultimi 7 giorni")
        else:
            st.warning("⚠️ Database vuoto. Esegui prima l'ingestion e il processing NLP.")
    
    else:
        # Display selected report
        report = st.session_state.db.get_report(st.session_state.current_report_id)
        
        if report:
            display_report_viewer(report)
        else:
            st.error(f"Report ID {st.session_state.current_report_id} non trovato")
            st.session_state.current_report_id = None


if __name__ == "__main__":
    main()
