# Indian Stock Market Dashboard - Complete Streamlit Application
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Indian Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
        margin: 0.5rem 0;
    }
    .stMetric > label {
        font-size: 0.9rem !important;
        color: #666 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📊 Indian Stock Market Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Real-time NSE/BSE stock analysis with technical indicators and fundamental data**")

# Sidebar for controls
st.sidebar.header("🔧 Dashboard Controls")

# Exchange selection
exchange = st.sidebar.selectbox("📈 Select Exchange", ["NSE", "BSE"], index=0)

# Stock data dictionaries
@st.cache_data(ttl=3600)
def get_stock_lists():
    nse_stocks = {
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "Tata Consultancy Services",
        "HDFCBANK.NS": "HDFC Bank",
        "INFY.NS": "Infosys",
        "ICICIBANK.NS": "ICICI Bank",
        "HINDUNILVR.NS": "Hindustan Unilever",
        "SBIN.NS": "State Bank of India",
        "BAJFINANCE.NS": "Bajaj Finance",
        "BHARTIARTL.NS": "Bharti Airtel",
        "KOTAKBANK.NS": "Kotak Mahindra Bank",
        "ITC.NS": "ITC Limited",
        "ASIANPAINT.NS": "Asian Paints",
        "AXISBANK.NS": "Axis Bank",
        "MARUTI.NS": "Maruti Suzuki India",
        "HCLTECH.NS": "HCL Technologies",
        "WIPRO.NS": "Wipro",
        "ULTRACEMCO.NS": "UltraTech Cement",
        "NESTLEIND.NS": "Nestle India",
        "TATAMOTORS.NS": "Tata Motors",
        "POWERGRID.NS": "Power Grid Corporation"
    }
    
    bse_stocks = {
        "RELIANCE.BO": "Reliance Industries",
        "TCS.BO": "Tata Consultancy Services",
        "HDFCBANK.BO": "HDFC Bank",
        "INFY.BO": "Infosys",
        "ICICIBANK.BO": "ICICI Bank",
        "HINDUNILVR.BO": "Hindustan Unilever",
        "SBIN.BO": "State Bank of India",
        "BAJFINANCE.BO": "Bajaj Finance",
        "BHARTIARTL.BO": "Bharti Airtel",
        "KOTAKBANK.BO": "Kotak Mahindra Bank"
    }
    
    return nse_stocks, bse_stocks

nse_stocks, bse_stocks = get_stock_lists()
stocks_dict = nse_stocks if exchange == "NSE" else bse_stocks

# Stock selection with search
stock_options = [f"{name} ({symbol})" for symbol, name in stocks_dict.items()]
st.sidebar.markdown("---")

# Search functionality
search_query = st.sidebar.text_input("🔍 Search Stock", placeholder="Type stock name...")
if search_query:
    filtered_options = [opt for opt in stock_options if search_query.lower() in opt.lower()]
else:
    filtered_options = stock_options

# Stock dropdown
selected_option = st.sidebar.selectbox("📋 Select Stock", filtered_options, index=0)
selected_stock = selected_option.split("(")[1].split(")")[0].strip()

# Manual stock input
st.sidebar.markdown("---")
custom_stock = st.sidebar.text_input("⌨️ Manual Symbol Entry", placeholder="e.g., EICHERMOT.NS")
if custom_stock:
    selected_stock = custom_stock.upper()

# Time period and interval selection
st.sidebar.markdown("---")
time_period = st.sidebar.selectbox(
    "📅 Time Period",
    ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
    index=5
)

interval_options = {
    "1d": ["1m", "5m", "15m", "30m", "1h"],
    "5d": ["5m", "15m", "30m", "1h", "1d"],
    "1mo": ["1d", "5d", "1wk"],
    "3mo": ["1d", "5d", "1wk"],
    "6mo": ["1d", "5d", "1wk"],
    "1y": ["1d", "5d", "1wk", "1mo"],
    "2y": ["1d", "5d", "1wk", "1mo"],
    "5y": ["1d", "5d", "1wk", "1mo"],
    "max": ["1d", "5d", "1wk", "1mo"]
}

interval = st.sidebar.selectbox("⏰ Data Interval", interval_options[time_period], index=0)

# Refresh button
if st.sidebar.button("🔄 Refresh Data", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Function to get stock data with caching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(ticker, period, interval):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        info = stock.info
        return data, info
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None

# Function to calculate technical indicators
def calculate_technical_indicators(data):
    if data is None or data.empty:
        return data
    
    # Moving averages
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    # RSI calculation
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD calculation
    exp1 = data['Close'].ewm(span=12).mean()
    exp2 = data['Close'].ewm(span=26).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    
    return data

# Main content area
if selected_stock:
    # Fetch data
    with st.spinner(f"Fetching data for {selected_stock}..."):
        data, info = fetch_stock_data(selected_stock, time_period, interval)
    
    if data is not None and not data.empty:
        # Calculate technical indicators
        data = calculate_technical_indicators(data)
        
        # Display current price and basic info
        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change = current_price - prev_close
        price_change_pct = (price_change / prev_close) * 100
        
        # Header with current price
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label=f"💰 Current Price ({selected_stock})",
                value=f"₹{current_price:.2f}",
                delta=f"{price_change:.2f} ({price_change_pct:.2f}%)"
            )
        
        with col2:
            if info:
                market_cap = info.get('marketCap', 'N/A')
                if market_cap != 'N/A':
                    market_cap_cr = market_cap / 10000000  # Convert to crores
                    st.metric("📊 Market Cap", f"₹{market_cap_cr:.0f} Cr")
                else:
                    st.metric("📊 Market Cap", "N/A")
        
        with col3:
            if info:
                pe_ratio = info.get('trailingPE', 'N/A')
                st.metric("📈 P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio != 'N/A' else 'N/A')
        
        with col4:
            if info:
                div_yield = info.get('dividendYield', 'N/A')
                if div_yield != 'N/A':
                    st.metric("💵 Dividend Yield", f"{div_yield*100:.2f}%")
                else:
                    st.metric("💵 Dividend Yield", "N/A")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "🔧 Technical Analysis", "📊 Fundamentals", "📋 Data Table"])
        
        with tab1:
            st.subheader(f"Price Chart - {selected_stock}")
            
            # Candlestick chart
            fig_candlestick = go.Figure(data=go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Price"
            ))
            
            fig_candlestick.update_layout(
                title=f"{selected_stock} - Candlestick Chart",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=500,
                xaxis_rangeslider_visible=False,
                showlegend=False
            )
            
            st.plotly_chart(fig_candlestick, use_container_width=True)
            
            # Volume chart
            fig_volume = px.bar(
                x=data.index, 
                y=data['Volume'],
                title=f"{selected_stock} - Trading Volume",
                labels={'x': 'Date', 'y': 'Volume'}
            )
            fig_volume.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with tab2:
            st.subheader("Technical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Moving averages chart
                fig_ma = go.Figure()
                fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
                fig_ma.add_trace(go.Scatter(x=data.index, y=data['MA20'], mode='lines', name='MA20', line=dict(color='orange')))
                fig_ma.add_trace(go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='MA50', line=dict(color='red')))
                
                if len(data) >= 200:
                    fig_ma.add_trace(go.Scatter(x=data.index, y=data['MA200'], mode='lines', name='MA200', line=dict(color='green')))
                
                fig_ma.update_layout(title="Moving Averages", height=400)
                st.plotly_chart(fig_ma, use_container_width=True)
            
            with col2:
                # RSI chart
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                fig_rsi.update_layout(title="RSI (Relative Strength Index)", yaxis=dict(range=[0, 100]), height=400)
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            # MACD chart
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
            fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='Signal', line=dict(color='red')))
            fig_macd.add_trace(go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histogram', opacity=0.7))
            fig_macd.update_layout(title="MACD Analysis", height=400)
            st.plotly_chart(fig_macd, use_container_width=True)
            
            # Technical signals
            st.subheader("📊 Technical Signals")
            
            if not data.empty:
                current_rsi = data['RSI'].iloc[-1]
                current_macd = data['MACD'].iloc[-1]
                current_signal = data['MACD_Signal'].iloc[-1]
                
                signal_col1, signal_col2, signal_col3 = st.columns(3)
                
                with signal_col1:
                    if current_rsi > 70:
                        st.error("🔴 RSI: Overbought")
                    elif current_rsi < 30:
                        st.success("🟢 RSI: Oversold")
                    else:
                        st.info("🟡 RSI: Neutral")
                
                with signal_col2:
                    if current_macd > current_signal:
                        st.success("🟢 MACD: Bullish")
                    else:
                        st.error("🔴 MACD: Bearish")
                
                with signal_col3:
                    if not data['MA20'].isna().iloc[-1] and not data['MA50'].isna().iloc[-1]:
                        if data['MA20'].iloc[-1] > data['MA50'].iloc[-1]:
                            st.success("🟢 MA: Bullish Cross")
                        else:
                            st.error("🔴 MA: Bearish Cross")
        
        with tab3:
            st.subheader("Fundamental Analysis")
            
            if info:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🏢 Company Information")
                    company_name = info.get('longName', info.get('shortName', 'N/A'))
                    st.write(f"**Company:** {company_name}")
                    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.write(f"**Country:** {info.get('country', 'N/A')}")
                    st.write(f"**Currency:** {info.get('currency', 'N/A')}")
                    
                    # Business summary
                    if 'longBusinessSummary' in info:
                        st.markdown("### 📄 Business Summary")
                        st.write(info['longBusinessSummary'][:500] + "..." if len(info['longBusinessSummary']) > 500 else info['longBusinessSummary'])
                
                with col2:
                    st.markdown("### 📊 Key Financial Metrics")
                    
                    metrics_data = {
                        "Market Cap": f"₹{info.get('marketCap', 0)/10000000:.0f} Cr" if info.get('marketCap') else 'N/A',
                        "P/E Ratio (TTM)": f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else 'N/A',
                        "Forward P/E": f"{info.get('forwardPE', 'N/A'):.2f}" if info.get('forwardPE') else 'N/A',
                        "PEG Ratio": f"{info.get('pegRatio', 'N/A'):.2f}" if info.get('pegRatio') else 'N/A',
                        "Price to Book": f"{info.get('priceToBook', 'N/A'):.2f}" if info.get('priceToBook') else 'N/A',
                        "Dividend Yield": f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else 'N/A',
                        "Beta": f"{info.get('beta', 'N/A'):.2f}" if info.get('beta') else 'N/A',
                        "52 Week High": f"₹{info.get('fiftyTwoWeekHigh', 'N/A'):.2f}" if info.get('fiftyTwoWeekHigh') else 'N/A',
                        "52 Week Low": f"₹{info.get('fiftyTwoWeekLow', 'N/A'):.2f}" if info.get('fiftyTwoWeekLow') else 'N/A',
                        "Average Volume": f"{info.get('averageVolume', 'N/A'):,}" if info.get('averageVolume') else 'N/A'
                    }
                    
                    metrics_df = pd.DataFrame(list(metrics_data.items()), columns=["Metric", "Value"])
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                    
                    # Additional financial ratios
                    st.markdown("### 💰 Financial Ratios")
                    ratios_data = {
                        "Return on Assets": f"{info.get('returnOnAssets', 0)*100:.2f}%" if info.get('returnOnAssets') else 'N/A',
                        "Return on Equity": f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else 'N/A',
                        "Profit Margin": f"{info.get('profitMargins', 0)*100:.2f}%" if info.get('profitMargins') else 'N/A',
                        "Operating Margin": f"{info.get('operatingMargins', 0)*100:.2f}%" if info.get('operatingMargins') else 'N/A',
                        "Debt to Equity": f"{info.get('debtToEquity', 'N/A'):.2f}" if info.get('debtToEquity') else 'N/A',
                        "Current Ratio": f"{info.get('currentRatio', 'N/A'):.2f}" if info.get('currentRatio') else 'N/A'
                    }
                    
                    ratios_df = pd.DataFrame(list(ratios_data.items()), columns=["Ratio", "Value"])
                    st.dataframe(ratios_df, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ Fundamental data not available for this stock.")
        
        with tab4:
            st.subheader("📋 Historical Data")
            
            # Display options
            col1, col2 = st.columns(2)
            with col1:
                show_technical = st.checkbox("Show Technical Indicators", value=False)
            with col2:
                rows_to_show = st.selectbox("Rows to display", [10, 25, 50, 100, len(data)], index=0)
            
            # Prepare data for display
            display_data = data.copy()
            
            if not show_technical:
                # Remove technical indicator columns
                tech_cols = ['MA20', 'MA50', 'MA200', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram']
                display_data = display_data.drop(columns=[col for col in tech_cols if col in display_data.columns])
            
            # Format numerical columns
            numeric_cols = display_data.select_dtypes(include=[np.number]).columns
            display_data[numeric_cols] = display_data[numeric_cols].round(2)
            
            # Show latest data first
            display_data = display_data.tail(rows_to_show).iloc[::-1]
            
            st.dataframe(display_data, use_container_width=True)
            
            # Download option
            csv = display_data.to_csv(index=True)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"{selected_stock}_{time_period}_data.csv",
                mime="text/csv"
            )
    
    else:
        st.error(f"❌ Unable to fetch data for {selected_stock}. Please check the symbol and try again.")

else:
    st.info("👆 Please select a stock from the sidebar to begin analysis.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    📊 <strong>Indian Stock Market Dashboard</strong> | Data powered by Yahoo Finance<br>
    ⚠️ <em>Disclaimer: This tool is for educational purposes only. Not financial advice.</em>
</div>
""", unsafe_allow_html=True)