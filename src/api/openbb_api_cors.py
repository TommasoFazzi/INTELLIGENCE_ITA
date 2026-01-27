"""
OpenBB Platform API with CORS enabled for OpenBB Workspace.

Run with: uvicorn src.api.openbb_api_cors:app --host 0.0.0.0 --port 6900
"""

import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Configure FRED API key for OpenBB
fred_key = os.getenv('FRED_API_KEY', '').strip()
if fred_key:
    os.environ['OPENBB_FRED_API_KEY'] = fred_key

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openbb_core.api.rest_api import app

# Add CORS middleware directly to the OpenBB app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Widget configuration for OpenBB Workspace
WIDGETS_CONFIG = {
    "fred_search": {
        "name": "FRED Search",
        "description": "Search FRED economic data series",
        "category": "FRED",
        "type": "table",
        "endpoint": "widget/fred_search",
        "gridData": {"w": 20, "h": 9},
        "params": [
            {"paramName": "query", "label": "Search Query", "type": "form", "value": "GDP"}
        ]
    },
    "fred_series": {
        "name": "FRED Series",
        "description": "Get FRED time series data",
        "category": "FRED",
        "type": "chart",
        "endpoint": "widget/fred_series",
        "gridData": {"w": 20, "h": 9},
        "params": [
            {"paramName": "symbol", "label": "Series ID", "type": "form", "value": "DGS10"}
        ]
    },
    "bond_indices": {
        "name": "Bond Indices",
        "description": "Treasury yields and bond indices",
        "category": "FRED",
        "type": "table",
        "endpoint": "widget/bond_indices",
        "gridData": {"w": 20, "h": 9}
    },
    "pce": {
        "name": "PCE",
        "description": "Personal Consumption Expenditures",
        "category": "FRED",
        "type": "table",
        "endpoint": "widget/pce",
        "gridData": {"w": 20, "h": 9}
    },
    "cpi": {
        "name": "CPI",
        "description": "Consumer Price Index",
        "category": "FRED",
        "type": "table",
        "endpoint": "widget/cpi",
        "gridData": {"w": 20, "h": 9}
    },
    "nonfarm_payrolls": {
        "name": "Nonfarm Payrolls",
        "description": "US Nonfarm Payrolls data",
        "category": "FRED",
        "type": "table",
        "endpoint": "widget/nonfarm_payrolls",
        "gridData": {"w": 20, "h": 9}
    },
    "retail_prices": {
        "name": "Retail Prices",
        "description": "Retail price indices",
        "category": "FRED",
        "type": "table",
        "endpoint": "widget/retail_prices",
        "gridData": {"w": 20, "h": 9}
    },
    "chicago_economic": {
        "name": "Chicago Economic Conditions",
        "description": "Chicago Fed economic conditions",
        "category": "FRED",
        "type": "table",
        "endpoint": "widget/chicago_economic",
        "gridData": {"w": 20, "h": 9}
    },
    "empire_state": {
        "name": "Empire State Manufacturing",
        "description": "NY Fed manufacturing outlook",
        "category": "FRED",
        "type": "table",
        "endpoint": "widget/empire_state",
        "gridData": {"w": 20, "h": 9}
    },
    "texas_manufacturing": {
        "name": "Texas Manufacturing Outlook",
        "description": "Dallas Fed manufacturing outlook",
        "category": "FRED",
        "type": "table",
        "endpoint": "widget/texas_manufacturing",
        "gridData": {"w": 20, "h": 9}
    },
    "sloos": {
        "name": "SLOOS",
        "description": "Senior Loan Officer Opinion Survey",
        "category": "FRED",
        "type": "table",
        "endpoint": "widget/sloos",
        "gridData": {"w": 20, "h": 9}
    },
    "unemployment": {
        "name": "Unemployment Rate",
        "description": "US unemployment statistics",
        "category": "FRED",
        "type": "table",
        "endpoint": "widget/unemployment",
        "gridData": {"w": 20, "h": 9}
    }
}

@app.get("/widgets.json")
def get_widgets():
    """Return widget configuration for OpenBB Workspace"""
    return JSONResponse(content=WIDGETS_CONFIG)


# ===================================================================
# Wrapper endpoints that return results array directly for tables
# ===================================================================
from openbb import obb

@app.get("/widget/fred_search")
def widget_fred_search(query: str = "GDP"):
    """FRED Search - returns results array directly for table widget"""
    try:
        result = obb.economy.fred_search(query=query, provider="fred")
        if result and result.results:
            # Return array of dicts with selected columns
            return [
                {
                    "Series ID": r.series_id,
                    "Title": r.title[:60] + "..." if len(r.title) > 60 else r.title,
                    "Frequency": r.frequency_short,
                    "Units": r.units_short,
                    "Last Updated": str(r.last_updated)[:10] if r.last_updated else "-"
                }
                for r in result.results[:50]  # Limit to 50 results
            ]
        return []
    except Exception as e:
        return [{"Error": str(e)}]


@app.get("/widget/fred_series")
def widget_fred_series(symbol: str = "DGS10"):
    """FRED Series - returns results for chart widget"""
    try:
        result = obb.economy.fred_series(symbol=symbol, provider="fred")
        if result and result.results:
            return [
                {"date": str(r.date), "value": getattr(r, symbol, None) or getattr(r, 'value', None)}
                for r in result.results[-252:]
            ]
        return []
    except Exception as e:
        return [{"Error": str(e)}]


@app.get("/widget/bond_indices")
def widget_bond_indices():
    """Bond Indices - Treasury yields"""
    try:
        series = ["DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
        rows = []
        for s in series:
            result = obb.economy.fred_series(symbol=s, provider="fred")
            if result and result.results:
                last = result.results[-1]
                val = getattr(last, s, None)
                rows.append({"Maturity": s.replace("DGS", "") + "Y", "Yield %": val, "Date": str(last.date)})
        return rows if rows else [{"Info": "No data"}]
    except Exception as e:
        return [{"Error": str(e)}]


@app.get("/widget/pce")
def widget_pce():
    """PCE - Personal Consumption Expenditures"""
    try:
        result = obb.economy.pce(provider="fred")
        if result and result.results:
            return [
                {"Date": str(r.date), "Value": r.value if hasattr(r, 'value') else None}
                for r in result.results[-24:]
            ]
        return [{"Info": "No data"}]
    except Exception as e:
        return [{"Error": str(e)}]


@app.get("/widget/cpi")
def widget_cpi():
    """CPI - Consumer Price Index"""
    try:
        result = obb.economy.cpi(provider="fred")
        if result and result.results:
            return [
                {"Date": str(r.date), "Country": getattr(r, 'country', 'US'), "Value": r.value}
                for r in result.results[-24:]
            ]
        return [{"Info": "No data"}]
    except Exception as e:
        return [{"Error": str(e)}]


@app.get("/widget/nonfarm_payrolls")
def widget_nonfarm_payrolls():
    """Nonfarm Payrolls"""
    try:
        result = obb.economy.survey.nonfarm_payrolls(provider="fred")
        if result and result.results:
            return [
                {"Date": str(r.date), "Category": getattr(r, 'category', '-'), "Value": getattr(r, 'value', None)}
                for r in result.results[-20:]
            ]
        return [{"Info": "No data"}]
    except Exception as e:
        return [{"Error": str(e)}]


@app.get("/widget/retail_prices")
def widget_retail_prices():
    """Retail Prices"""
    try:
        result = obb.economy.retail_prices(provider="fred")
        if result and result.results:
            return [
                {"Date": str(r.date), "Country": getattr(r, 'country', 'US'), "Value": getattr(r, 'value', None)}
                for r in result.results[-24:]
            ]
        return [{"Info": "No data"}]
    except Exception as e:
        return [{"Error": str(e)}]


@app.get("/widget/chicago_economic")
def widget_chicago_economic():
    """Chicago Fed Economic Conditions"""
    try:
        result = obb.economy.survey.economic_conditions_chicago(provider="fred")
        if result and result.results:
            return [
                {"Date": str(r.date), "Index": getattr(r, 'activity_index', None), "Diffusion": getattr(r, 'diffusion_index', None)}
                for r in result.results[-24:]
            ]
        return [{"Info": "No data"}]
    except Exception as e:
        return [{"Error": str(e)}]


@app.get("/widget/empire_state")
def widget_empire_state():
    """Empire State Manufacturing - NY Fed"""
    try:
        result = obb.economy.survey.manufacturing_outlook_ny(provider="fred")
        if result and result.results:
            return [
                {"Date": str(r.date), "Topic": getattr(r, 'topic', '-'), "Value": getattr(r, 'value', None)}
                for r in result.results[-20:]
            ]
        return [{"Info": "No data"}]
    except Exception as e:
        return [{"Error": str(e)}]


@app.get("/widget/texas_manufacturing")
def widget_texas_manufacturing():
    """Texas Manufacturing Outlook - Dallas Fed"""
    try:
        result = obb.economy.survey.manufacturing_outlook_texas(provider="fred")
        if result and result.results:
            return [
                {"Date": str(r.date), "Topic": getattr(r, 'topic', '-'), "Value": getattr(r, 'value', None)}
                for r in result.results[-20:]
            ]
        return [{"Info": "No data"}]
    except Exception as e:
        return [{"Error": str(e)}]


@app.get("/widget/sloos")
def widget_sloos():
    """SLOOS - Senior Loan Officer Opinion Survey"""
    try:
        result = obb.economy.survey.sloos(provider="fred")
        if result and result.results:
            return [
                {"Date": str(r.date), "Category": getattr(r, 'category', '-'), "Value": getattr(r, 'value', None)}
                for r in result.results[-20:]
            ]
        return [{"Info": "No data"}]
    except Exception as e:
        return [{"Error": str(e)}]


@app.get("/widget/unemployment")
def widget_unemployment():
    """Unemployment Rate"""
    try:
        result = obb.economy.unemployment(provider="fred")
        if result and result.results:
            return [
                {"Date": str(r.date), "Country": getattr(r, 'country', 'US'), "Rate %": getattr(r, 'value', None)}
                for r in result.results[-24:]
            ]
        return [{"Info": "No data"}]
    except Exception as e:
        return [{"Error": str(e)}]


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("  OpenBB Platform API (CORS Enabled)")
    print("="*60)
    print(f"\n  FRED API Key: {fred_key[:8]}..." if fred_key else "\n  WARNING: No FRED API key found!")
    print("\n  Server: http://localhost:6900")
    print("  Docs:   http://localhost:6900/docs")
    print("\n  Connect from OpenBB Workspace:")
    print("  URL: http://127.0.0.1:6900")
    print("\n" + "="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=6900)
