"""
The War Room - INTELLIGENCE_ITA Dashboard Home

Entry point for the Multi-Page App.
Displays situational awareness metrics and quick navigation.
"""

import sys
from pathlib import Path

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
from datetime import datetime

from src.hitl.streamlit_utils import (
    get_db_manager,
    inject_custom_css,
    get_status_emoji,
    init_session_state
)

# Page Configuration (must be first Streamlit call)
st.set_page_config(
    page_title="The War Room | INTELLIGENCE_ITA",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS
inject_custom_css()

# Initialize
init_session_state()
db = get_db_manager()


def render_ticker_tape():
    """Render the top metrics bar with real data."""
    st.markdown("### Situational Awareness Dashboard")

    # Get real statistics from DB
    try:
        stats = db.get_statistics()
    except Exception as e:
        st.error(f"Database connection error: {e}")
        stats = {
            'total_articles': 0,
            'total_chunks': 0,
            'recent_articles': 0,
            'by_category': {}
        }

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Articoli Totali",
            value=f"{stats.get('total_articles', 0):,}",
            help="Totale articoli nel database"
        )

    with col2:
        st.metric(
            label="Ultimi 7 Giorni",
            value=f"{stats.get('recent_articles', 0):,}",
            help="Articoli raccolti negli ultimi 7 giorni"
        )

    with col3:
        st.metric(
            label="Chunks RAG",
            value=f"{stats.get('total_chunks', 0):,}",
            help="Chunks indicizzati per semantic search"
        )

    with col4:
        st.metric(
            label="System Status",
            value="Online",
            delta="Operational",
            delta_color="normal"
        )


def render_signal_radar():
    """Render the Signal Radar table."""
    st.subheader("📡 Signal Radar")
    st.caption("Trade signals estratti dai report (ultimi 7 giorni)")

    # Try to fetch from trade_signals table if exists
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        ts.ticker,
                        ts.signal,
                        ts.timeframe,
                        ts.confidence,
                        ts.rationale,
                        ts.created_at::date as signal_date
                    FROM trade_signals ts
                    WHERE ts.created_at > NOW() - INTERVAL '7 days'
                    ORDER BY ts.created_at DESC
                    LIMIT 10
                """)
                rows = cur.fetchall()

                if rows:
                    df = pd.DataFrame(rows, columns=[
                        'Ticker', 'Signal', 'Timeframe', 'Confidence', 'Rationale', 'Date'
                    ])

                    # Format confidence as percentage
                    df['Confidence'] = df['Confidence'].apply(
                        lambda x: f"{float(x)*100:.0f}%" if x else "N/A"
                    )

                    # Add emoji to signal
                    signal_emoji = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '🟡', 'WATCHLIST': '👁️'}
                    df['Signal'] = df['Signal'].apply(
                        lambda x: f"{signal_emoji.get(x, '')} {x}"
                    )

                    st.dataframe(
                        df,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Rationale": st.column_config.TextColumn(
                                "Rationale",
                                width="large"
                            )
                        }
                    )
                else:
                    st.info("Nessun segnale negli ultimi 7 giorni. Esegui `--macro-first` per generare segnali.")

    except Exception as e:
        # trade_signals table might not exist
        st.warning(f"Tabella trade_signals non disponibile. Esegui migration 005.")
        st.code("psql -d intelligence_ita -f migrations/005_add_trade_signals.sql")


def render_report_status():
    """Render recent reports status."""
    st.subheader("📋 Report Recenti")

    try:
        reports = db.get_all_reports(limit=5)

        if reports:
            for report in reports:
                status_emoji = get_status_emoji(report.get('status', 'draft'))
                report_date = report.get('report_date', datetime.now())
                if hasattr(report_date, 'strftime'):
                    date_str = report_date.strftime('%d/%m/%Y')
                else:
                    date_str = str(report_date)[:10]

                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    st.write(f"{status_emoji} **ID {report['id']}**")
                with col2:
                    st.write(f"📅 {date_str}")
                with col3:
                    st.write(f"`{report.get('status', 'draft').upper()}`")
        else:
            st.info("Nessun report generato. Vai al Daily Briefing per crearne uno.")

    except Exception as e:
        st.error(f"Errore nel recupero report: {e}")


def render_quick_actions():
    """Render quick action buttons."""
    st.subheader("🚀 Quick Actions")

    if st.button("📝 Vai al Daily Briefing", width="stretch", type="primary"):
        st.switch_page("pages/1_Daily_Briefing.py")

    if st.button("🧠 Interroga The Oracle", width="stretch"):
        st.switch_page("pages/2_The_Oracle.py")

    st.divider()

    st.caption("Pipeline Commands")
    st.code("""
# Esegui pipeline completa
python -m src.ingestion.pipeline
python scripts/process_nlp.py
python scripts/load_to_database.py
python scripts/generate_report.py --macro-first
    """, language="bash")


def render_category_breakdown():
    """Render article breakdown by category."""
    st.subheader("📊 Articoli per Categoria")

    try:
        stats = db.get_statistics()
        by_category = stats.get('by_category', {})

        if by_category:
            df = pd.DataFrame([
                {'Categoria': k, 'Articoli': v}
                for k, v in by_category.items()
            ])
            df = df.sort_values('Articoli', ascending=False)

            st.bar_chart(df.set_index('Categoria'))
        else:
            st.info("Nessun dato di categoria disponibile.")

    except Exception as e:
        st.warning(f"Errore: {e}")


def main():
    """Main application entry point."""
    st.title("🏠 The War Room")

    # Top metrics
    render_ticker_tape()
    st.divider()

    # Main content layout
    col_left, col_right = st.columns([2, 1])

    with col_left:
        render_signal_radar()
        st.divider()
        render_category_breakdown()

    with col_right:
        render_report_status()
        st.divider()
        render_quick_actions()

    # Footer
    st.divider()
    st.caption(f"INTELLIGENCE_ITA v3.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
