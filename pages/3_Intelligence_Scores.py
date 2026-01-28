"""
Intelligence Scores - Scoring Breakdown Dashboard

Displays intelligence scores with full breakdown of scoring components:
- Technical analysis (SMA200 deviation)
- Fundamental analysis (PE ratio, valuation)
- LLM confidence and data quality
- Historical trend analysis
"""

import sys
from pathlib import Path
from datetime import datetime

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd

from src.hitl.streamlit_utils import (
    get_db_manager,
    inject_custom_css,
    init_session_state
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Page Configuration
st.set_page_config(
    page_title="Intelligence Scores | INTELLIGENCE_ITA",
    page_icon="📊",
    layout="wide"
)

# Inject CSS and initialize
inject_custom_css()
init_session_state()
db = get_db_manager()

# Global variable to hold full history for trend calculation
_df_full_history = None


def fetch_intelligence_scores_history(days: int = 30, min_score: int = 0):
    """Fetch complete historical intelligence scores (not just latest)."""
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        ts.ticker,
                        ts.signal,
                        ts.timeframe,
                        ts.intelligence_score,
                        ts.confidence,
                        ts.sma_200_deviation,
                        ts.pe_rel_valuation,
                        ts.valuation_rating,
                        ts.data_quality,
                        ts.category,
                        ts.rationale,
                        ts.created_at,
                        cf.company_name,
                        cf.sector,
                        cf.pe_ratio,
                        md.close_price,
                        ts.price_source,
                        ts.sma_source,
                        ts.pe_source,
                        ts.sector_pe_source,
                        ts.fetched_at,
                        ts.days_of_history
                    FROM trade_signals ts
                    LEFT JOIN company_fundamentals cf
                        ON UPPER(ts.ticker) = UPPER(cf.ticker)
                    LEFT JOIN LATERAL (
                        SELECT close_price
                        FROM market_data
                        WHERE UPPER(ticker) = UPPER(ts.ticker)
                        ORDER BY date DESC
                        LIMIT 1
                    ) md ON true
                    WHERE ts.intelligence_score IS NOT NULL
                      AND ts.intelligence_score >= %s
                      AND ts.created_at > NOW() - INTERVAL '%s days'
                    ORDER BY ts.ticker ASC, ts.created_at DESC
                """, (min_score, days))

                rows = cur.fetchall()
                columns = [
                    'ticker', 'signal', 'timeframe', 'intelligence_score', 'confidence',
                    'sma_200_deviation', 'pe_rel_valuation', 'valuation_rating',
                    'data_quality', 'category', 'rationale', 'created_at',
                    'company_name', 'sector', 'pe_ratio', 'price',
                    'price_source', 'sma_source', 'pe_source', 'sector_pe_source',
                    'fetched_at', 'days_of_history'
                ]

                return pd.DataFrame(rows, columns=columns)

    except Exception as e:
        logger.error(f"Error fetching intelligence scores history: {e}")
        st.error(f"Errore nel recupero dati storici: {e}")
        return pd.DataFrame()


def get_latest_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Get only the latest signal per ticker."""
    if df.empty:
        return df

    return df.sort_values('created_at', ascending=False).groupby('ticker', as_index=False).first()


def calculate_trend(df_full: pd.DataFrame, ticker: str) -> str:
    """Calculate trend indicator for ticker based on historical data."""
    ticker_data = df_full[df_full['ticker'] == ticker].sort_values('created_at')
    if len(ticker_data) < 2:
        return "➡️"

    first_score = ticker_data.iloc[0]['intelligence_score']  # oldest
    last_score = ticker_data.iloc[-1]['intelligence_score']  # newest
    delta = last_score - first_score

    if delta > 5:
        return "🔼"
    elif delta < -5:
        return "🔽"
    return "➡️"


def calculate_sma_200(row):
    """Calculate SMA 200 from price and deviation."""
    if row['price'] and row['sma_200_deviation']:
        try:
            return round(float(row['price']) / (1 + float(row['sma_200_deviation']) / 100), 2)
        except:
            return None
    return None


def render_metrics(df: pd.DataFrame):
    """Render summary metrics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_score = df['intelligence_score'].mean() if not df.empty else 0
        st.metric(
            label="Avg Score",
            value=f"{avg_score:.0f}",
            help="Media degli intelligence scores"
        )

    with col2:
        total = len(df)
        st.metric(
            label="Unique Tickers",
            value=total,
            help="Numero di ticker unici con score"
        )

    with col3:
        high_conv = len(df[df['intelligence_score'] >= 70]) if not df.empty else 0
        st.metric(
            label="High Conviction",
            value=high_conv,
            delta=f"score >= 70",
            help="Segnali con score >= 70"
        )

    with col4:
        full_quality = len(df[df['data_quality'] == 'FULL']) if not df.empty else 0
        st.metric(
            label="Data Quality FULL",
            value=full_quality,
            delta=f"of {total}",
            help="Segnali con dati completi"
        )


def render_filters():
    """Render filter controls in sidebar."""
    st.sidebar.title("Filtri")

    days = st.sidebar.slider(
        "Giorni Storico",
        min_value=7,
        max_value=90,
        value=30,
        help="Periodo di lookback per lo storico"
    )

    min_score = st.sidebar.slider(
        "Score Minimo",
        min_value=0,
        max_value=100,
        value=50,
        step=10,
        help="Filtra per score minimo"
    )

    signal_filter = st.sidebar.multiselect(
        "Tipo Segnale",
        options=['BULLISH', 'BEARISH', 'NEUTRAL', 'WATCHLIST'],
        default=[],
        help="Filtra per tipo di segnale"
    )

    quality_filter = st.sidebar.multiselect(
        "Data Quality",
        options=['FULL', 'PARTIAL', 'INSUFFICIENT'],
        default=[],
        help="""Filtra per qualità dati:

- FULL: Tutti i dati disponibili (prezzo, SMA200 su 200+ giorni, PE, settore PE)
- PARTIAL: Alcuni dati mancanti o SMA su <200 giorni (proxy mean)
- INSUFFICIENT: Solo prezzo o storico <100 giorni"""
    )

    st.sidebar.divider()

    if st.sidebar.button("Torna alla War Room", use_container_width=True):
        st.switch_page("Home.py")

    return days, min_score, signal_filter, quality_filter


def format_dataframe(df: pd.DataFrame, df_full_history: pd.DataFrame) -> pd.DataFrame:
    """Format dataframe for display with trend indicators."""
    if df.empty:
        return df

    # Calculate SMA 200
    df = df.copy()
    df['sma_200'] = df.apply(calculate_sma_200, axis=1)

    # Format columns
    display_df = pd.DataFrame()

    # Signal with emoji
    signal_emoji = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '🟡', 'WATCHLIST': '👁️'}
    display_df['Ticker'] = df['ticker']
    display_df['Company'] = df['company_name'].fillna('-')
    display_df['Score'] = df['intelligence_score']

    # Trend indicator based on historical data
    display_df['Trend'] = df['ticker'].apply(lambda t: calculate_trend(df_full_history, t))

    display_df['Signal'] = df['signal'].apply(lambda x: f"{signal_emoji.get(x, '')} {x}" if x else '-')
    display_df['Confidence'] = df['confidence'].apply(lambda x: f"{float(x)*100:.0f}%" if x else '-')

    # Price data
    display_df['Price'] = df['price'].apply(lambda x: f"${float(x):.2f}" if x else '-')
    display_df['SMA 200'] = df['sma_200'].apply(lambda x: f"${x:.2f}" if x else '-')
    display_df['SMA Dev %'] = df['sma_200_deviation'].apply(
        lambda x: f"{float(x):+.1f}%" if x is not None else '-'
    )

    # Fundamental data
    display_df['P/E'] = df['pe_ratio'].apply(lambda x: f"{float(x):.1f}" if x else '-')
    display_df['PE Rel'] = df['pe_rel_valuation'].apply(lambda x: f"{float(x):.2f}" if x else '-')

    # Valuation with color coding
    valuation_emoji = {
        'UNDERVALUED': '🟢',
        'FAIR': '🟡',
        'OVERVALUED': '🟠',
        'BUBBLE': '🔴',
        'LOSS_MAKING': '⚫',
        'UNKNOWN': '⚪'
    }
    display_df['Valuation'] = df['valuation_rating'].apply(
        lambda x: f"{valuation_emoji.get(x, '')} {x}" if x else '-'
    )

    # Data quality with emoji badges
    quality_emoji = {"FULL": "🟢", "PARTIAL": "🟡", "INSUFFICIENT": "🔴"}
    display_df['Quality'] = df['data_quality'].apply(
        lambda x: f"{quality_emoji.get(x, '⚪')} {x}" if x else '⚪ N/A'
    )
    display_df['Category'] = df['category'].fillna('-')
    display_df['Date'] = df['created_at'].apply(
        lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)[:10] if x else '-'
    )

    return display_df


def render_score_distribution(df: pd.DataFrame):
    """Render score distribution chart."""
    if df.empty:
        return

    st.subheader("Score Distribution")

    # Create bins for score distribution
    bins = [0, 50, 60, 70, 80, 90, 100]
    labels = ['0-50', '51-60', '61-70', '71-80', '81-90', '91-100']

    df_copy = df.copy()
    df_copy['score_range'] = pd.cut(
        df_copy['intelligence_score'],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    distribution = df_copy['score_range'].value_counts().sort_index()

    st.bar_chart(distribution)


def render_score_trend(df: pd.DataFrame, ticker: str):
    """Render line chart showing score evolution over time."""
    ticker_data = df[df['ticker'] == ticker].sort_values('created_at')

    if len(ticker_data) < 2:
        st.caption("📉 Dati insufficienti per visualizzare il trend (serve almeno 2 data points)")
        return

    # Prepare chart data
    chart_data = ticker_data[['created_at', 'intelligence_score']].copy()
    chart_data['created_at'] = pd.to_datetime(chart_data['created_at'])
    chart_data = chart_data.set_index('created_at')

    # Display chart
    st.line_chart(
        chart_data['intelligence_score'],
        use_container_width=True,
        color="#1f77b4"
    )

    # Show delta metrics
    first_score = ticker_data.iloc[0]['intelligence_score']   # oldest
    last_score = ticker_data.iloc[-1]['intelligence_score']   # newest
    delta = last_score - first_score

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Latest Score",
            f"{last_score}",
            f"{delta:+.0f} pts"
        )

    with col2:
        st.metric(
            "First Score",
            f"{first_score}"
        )

    with col3:
        st.metric(
            "Data Points",
            len(ticker_data)
        )


def render_signal_history(df: pd.DataFrame, ticker: str):
    """Render historical signals table for a ticker."""
    ticker_history = df[df['ticker'] == ticker].sort_values('created_at', ascending=False)

    if ticker_history.empty:
        return

    # Format history table
    history_display = pd.DataFrame()
    history_display['Date'] = ticker_history['created_at'].apply(
        lambda x: x.strftime('%Y-%m-%d %H:%M') if hasattr(x, 'strftime') else str(x)
    )
    history_display['Score'] = ticker_history['intelligence_score']

    signal_emoji = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '🟡', 'WATCHLIST': '👁️'}
    history_display['Signal'] = ticker_history['signal'].apply(
        lambda x: f"{signal_emoji.get(x, '')} {x}" if x else '-'
    )

    history_display['Confidence'] = ticker_history['confidence'].apply(
        lambda x: f"{float(x)*100:.0f}%" if x else '-'
    )

    valuation_emoji = {
        'UNDERVALUED': '🟢',
        'FAIR': '🟡',
        'OVERVALUED': '🟠',
        'BUBBLE': '🔴',
        'LOSS_MAKING': '⚫',
        'UNKNOWN': '⚪'
    }
    history_display['Valuation'] = ticker_history['valuation_rating'].apply(
        lambda x: f"{valuation_emoji.get(x, '')} {x}" if x else '-'
    )

    history_display['Quality'] = ticker_history['data_quality']

    st.dataframe(
        history_display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score",
                min_value=0,
                max_value=100,
                format="%d"
            )
        }
    )


def render_detail_expander(ticker: str, latest_row: dict, history_df: pd.DataFrame):
    """Render detailed view for a single signal with historical data."""
    company = latest_row.get('Company', 'N/A')
    score = latest_row.get('Score', 'N/A')

    with st.expander(f"📋 {ticker} - {company} (Score: {score})"):

        # Latest signal details
        st.markdown("### 📊 Latest Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Technical Analysis**")
            st.write(f"- Price: {latest_row.get('Price', '-')}")
            st.write(f"- SMA 200: {latest_row.get('SMA 200', '-')}")
            st.write(f"- Deviation: {latest_row.get('SMA Dev %', '-')}")

        with col2:
            st.markdown("**Fundamental Analysis**")
            st.write(f"- P/E Ratio: {latest_row.get('P/E', '-')}")
            st.write(f"- PE Relative: {latest_row.get('PE Rel', '-')}")
            st.write(f"- Valuation: {latest_row.get('Valuation', '-')}")

        # Data Quality Assessment section
        st.markdown("**📊 Data Quality Assessment**")
        quality = latest_row.get('data_quality', 'UNKNOWN')
        quality_badge = {"FULL": "🟢", "PARTIAL": "🟡", "INSUFFICIENT": "🔴"}.get(quality, "⚪")
        st.markdown(f"**Qualità Dati**: {quality_badge} {quality}")

        with st.expander("🔍 Dettaglio Sorgenti Dati", expanded=False):
            src_col1, src_col2 = st.columns(2)
            with src_col1:
                st.write(f"- **Prezzo**: {latest_row.get('price_source', 'N/A')}")
                st.write(f"- **SMA 200**: {latest_row.get('sma_source', 'N/A')}")
            with src_col2:
                st.write(f"- **P/E Ratio**: {latest_row.get('pe_source', 'N/A')}")
                st.write(f"- **Settore PE**: {latest_row.get('sector_pe_source', 'N/A')}")

            days = latest_row.get('days_of_history', 0) or 0
            if days < 100:
                st.warning(f"⚠️ Solo {days} giorni di storico - dati insufficienti")
            elif days < 200:
                st.info(f"📊 {days} giorni di storico - SMA basata su proxy mean")
            else:
                st.success(f"✅ {days} giorni di storico - SMA calcolata correttamente")

            fetched = latest_row.get('fetched_at')
            if fetched:
                st.caption(f"Dati recuperati: {fetched}")

        st.markdown("**Rationale**")
        st.info(latest_row.get('rationale', 'Nessun rationale disponibile'))

        st.divider()

        # Historical trend
        st.markdown("### 📈 Score Trend")
        render_score_trend(history_df, ticker)

        st.divider()

        # Signal history
        st.markdown("### 📜 Signal History")
        render_signal_history(history_df, ticker)


def main():
    """Main entry point."""
    st.title("📊 Intelligence Scores")
    st.caption("Breakdown completo degli intelligence scores con trend storici")

    # Get filters from sidebar
    days, min_score, signal_filter, quality_filter = render_filters()

    # Fetch FULL historical data
    with st.spinner("Caricamento dati storici..."):
        df_full = fetch_intelligence_scores_history(days=days, min_score=min_score)

    if df_full.empty:
        st.warning("Nessun intelligence score trovato per i filtri selezionati.")
        st.info("""
        **Suggerimenti:**
        - Aumenta il periodo di lookback (giorni)
        - Riduci lo score minimo
        - Esegui `python scripts/generate_report.py --macro-first` per generare nuovi segnali
        """)
        return

    # Apply additional filters to full dataset
    df_filtered = df_full.copy()

    if signal_filter:
        df_filtered = df_filtered[df_filtered['signal'].isin(signal_filter)]

    if quality_filter:
        df_filtered = df_filtered[df_filtered['data_quality'].isin(quality_filter)]

    if df_filtered.empty:
        st.warning("Nessun risultato per i filtri applicati.")
        return

    # Get LATEST signals for table display
    df_latest = get_latest_signals(df_filtered)

    # Render metrics based on latest signals
    render_metrics(df_latest)

    st.divider()

    # Format and display table with ONLY latest signals (pass full history for trend)
    display_df = format_dataframe(df_latest, df_filtered)

    # Add export button
    col1, col2 = st.columns([4, 1])
    with col2:
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Export CSV",
            data=csv,
            file_name=f"intelligence_scores_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    # Display table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score",
                min_value=0,
                max_value=100,
                format="%d"
            ),
            "Company": st.column_config.TextColumn(
                "Company",
                width="medium"
            ),
            "Trend": st.column_config.TextColumn(
                "Trend",
                width="small",
                help="🔼 improving, 🔽 declining, ➡️ stable"
            )
        }
    )

    st.divider()

    # Score distribution chart (based on latest)
    render_score_distribution(df_latest)

    # Detail section with historical data
    st.divider()
    st.subheader("📋 Dettagli Segnali con Storico")
    st.caption("Espandi per vedere l'evoluzione dello score nel tempo")

    # For each ticker, show details with full history
    for ticker in df_latest['ticker'].unique():
        # Get latest row data for this ticker
        latest_mask = df_latest['ticker'] == ticker
        latest_idx = df_latest[latest_mask].index[0]

        # Find position in display_df
        df_latest_list = df_latest.index.tolist()
        display_idx = df_latest_list.index(latest_idx)
        display_row = display_df.iloc[display_idx].to_dict()

        # Get rationale and audit trail from original data
        latest_data = df_latest[latest_mask].iloc[0]
        display_row['rationale'] = latest_data['rationale']
        display_row['data_quality'] = latest_data.get('data_quality')
        display_row['price_source'] = latest_data.get('price_source')
        display_row['sma_source'] = latest_data.get('sma_source')
        display_row['pe_source'] = latest_data.get('pe_source')
        display_row['sector_pe_source'] = latest_data.get('sector_pe_source')
        display_row['fetched_at'] = latest_data.get('fetched_at')
        display_row['days_of_history'] = latest_data.get('days_of_history')

        # Render with full history
        render_detail_expander(ticker, display_row, df_filtered)


if __name__ == "__main__":
    main()
