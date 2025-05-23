import streamlit as st
import pandas as pd
import plotly_express as px
from yf_utils import TickerAnalysis, SectorAnalysis, Query, MarketAnalysis
y = TickerAnalysis()
s = SectorAnalysis()
q = Query()
m = MarketAnalysis()

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .stButton button {
        background-color: lightblue;
        color: black;
        border-radius: 100px;
        font-size: 5px;
        font-color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

pages = ['Home', 'Ticker Analysis', 'Sector & Industry Analysis',
         'Market Analysis', 'Portfolio Analysis', 'Invest Divest', 'Search']

side_bar = st.sidebar.selectbox(
    'Select a Page',
    options = pages
)

if side_bar == 'Home':

    st.title('Welcome to Paladin Market Analytics')
    
    # fig1, fig2, fig3, fig4 = y.landing()

    # st.plotly_chart(fig1)
    # st.plotly_chart(fig2)
    # st.plotly_chart(fig3)
    # st.plotly_chart(fig4)

elif side_bar == 'Ticker Analysis':
    
    st.title('Ticker Analysis')
    tickers = st.text_input(label='Enter Tickers', placeholder='MSFT, AAPL, TSLA')
    history_options = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    history = st.selectbox(label='Choose Timeframe', placeholder='1d', options=history_options)

    if st.button('Run Analysis'):

        historical = y.ticker_history(symbol=tickers, timeframe=history)
        info = y.ticker_info(symbol=tickers)
        ticker_name = info.loc[info['attributes'] == 'longName', f'{tickers}'].values[0]
        candlestick = y.candlestick(symbol=tickers, timeframe=history)
        short_term = y.short_term_moving(symbol=tickers, timeframe=history)
        long_term = y.long_term_moving(symbol=tickers, timeframe=history)
        sustainability = y.ticker_sustainability(symbol=tickers)
        ticker_analyst_price_targets = y.analyst_price_targets(symbol=tickers)
        analyst_reccomendations = y.analyst_reccomendations(symbol=tickers)
        updown = y.upgrades_downgrades(symbol=tickers)
        ticker_news = y.ticker_news(symbol=tickers)
        ticker_news_list = y.ticker_news_list(symbol=tickers)
        volatility = y.volatility(symbol=tickers, timeframe=history)
        volatility_plot = y.volatility_plot(symbol=tickers, timeframe=history)
        insider_transactions = y.insider_transactions(symbol=tickers)
        institutional_holders = y.insitutional_holders(symbol=tickers)
        fund_top_holdings = y.fund_holdings(symbol=tickers)
        sec_filings = y.sec_filings(symbol=tickers)
        
        response = y.llm(prompt=f'''

        Given the following ticker {tickers}, 
        and associated company {ticker_name},
        the historical trends {historical},
        core info: {info}, 
        price_targets: {ticker_analyst_price_targets},
        analyst_reccomendations: {analyst_reccomendations}, 
        upgrades and downgrades: {updown},
        latest_news: {ticker_news_list},
        volatility_measurements: {volatility}, and
        insider_transactions: {insider_transactions}

        Please develop an incredibly detailed investment report detailing insights from all of the above, 
        especially the news, and provide a reccomendation for buying, holding, and selling.
                
        ''')

        st.plotly_chart(candlestick)
        st.plotly_chart(volatility_plot)
        st.plotly_chart(short_term)
        st.plotly_chart(long_term)
        st.write('Ticker Info')
        st.dataframe(info)
        st.write('Analyst Price Targets')
        st.dataframe(ticker_analyst_price_targets)
        st.write('Analyst Reccomendations')
        st.dataframe(analyst_reccomendations)
        st.write('Upgrades/Downgrades')
        st.dataframe(updown)
        st.write('Latest News')
        st.dataframe(ticker_news)
        st.write('Insider Transactions')
        st.dataframe(insider_transactions)
        st.write('Institutional Holders')
        st.dataframe(institutional_holders)
        st.write('Top Holdings')
        st.dataframe(fund_top_holdings)
        st.write('SEC Filings')
        st.dataframe(sec_filings)
        st.sidebar.write('Investment Report (Generated by AI)')
        st.sidebar.write_stream(response)

elif side_bar == 'Sector & Industry Analysis':

    st.title('Sector Analysis')

    sectors_industries = s.get_sectors_and_industries()
    sector_list = sectors_industries.Sector.unique().tolist()
    sector = st.selectbox(label='Choose Sector', options=sector_list)

    if sector:

        industries = sectors_industries[sectors_industries['Sector'] == sector]
        industry_list = industries.Industry.unique().tolist()
        industry = st.selectbox(label='Choose Industry', options=industry_list)

    if st.button('Run Sector Analysis'):

        #sectors
        sector_overview = s.sector_overview(sector_name=sector)
        top_s_companies = s.top_sector_companies(sector_name=sector)
        top_s_etfs = s.top_sector_etfs(sector_name=sector)
        top_s_mutuals = s.top_sector_mutual_funds(sector_name=sector)

        #industries
        industry_overview = s.industry_overview(industry_name=industry)
        top_i_companies = s.top_industry_companies(industry_name=industry)
        top_ig_companies = s.top_industry_growth_companies(industry_name=industry)
        top_ip_companies = s.top_industry_performing_companies(industry_name=industry)
        
        st.write('Sector Overview')
        st.dataframe(sector_overview)
        st.write('Top Sector Companies')
        st.dataframe(top_s_companies)
        st.write('Top Sector ETFs')
        st.dataframe(top_s_etfs)
        st.write('Top Sector Mutuals')
        st.dataframe(top_s_mutuals)

        st.write('Industry Overview')
        st.dataframe(industry_overview)
        st.write('Top Industry Companies')
        st.dataframe(top_i_companies)
        st.write('Top Industry Growth Companies')
        st.dataframe(top_ig_companies)
        st.write('Top Industry Performing Companies')
        st.dataframe(top_ip_companies)
        
elif side_bar == 'Market Analysis':

    market_options = ['US', 'GB', 'ASIA', 'EUROPE', 'RATES', 'COMMODITIES', 'CURRENCIES', 'CRYPTOCURRENCIES']
    market_selection = st.selectbox(label='Select Market', options=market_options)

    if st.button('Analyze Market'):
        
        market_summary = m.market_summary(market_name=market_selection)
        market_status = m.market_status(market_name=market_selection)
        st.write('Market Summary')
        st.dataframe(market_summary)
        st.write('Market Status')
        st.dataframe(market_status)

elif side_bar == 'Portfolio Analysis':

    st.title('Portfolio Analysis')
    file = st.file_uploader('Please Upload Porfolio (CSV)')

    if file is not None:
        if "cached_df" in st.session_state:
            del st.session_state["cached_df"] 
        st.session_state["cached_df"] = pd.read_csv(file)
        st.write('File successfully uploaded...')
        st.dataframe(st.session_state['cached_df'])

        if st.button('Analyze Portfolio for Long Term Growth'):

            response = y.llm(prompt=f'''
            Given my portfolio: {st.session_state['cached_df']}, please provide an incredibly detailed investment report on how I am doing,
            how I can improve, invest, divest, diversify. Please go into specifics with tickers and explain. My goal is long term growth.
            ''')

            st.write_stream(response)
        
        if st.button('Analyze Portfolio for Short Term Growth'):
            portfolio_str = st.session_state["cached_df"].to_string(index=False)

            response = y.llm(prompt=f'''
            Given my portfolio: {st.session_state['cached_df']}, please provide an incredibly detailed investment report on how I am doing,
            how I can improve, invest, divest, diversify. Please go into specifics with tickers and explain. My goal is short term growth.
            ''')

            st.write_stream(response)

elif side_bar == 'Invest Divest':

    st.title('Get Latest Investment / Divestment Reccomendations')

    rec_options = ['All', 'Strong Buy', 'Buy', 'Hold', 'Sell', 'Underperform']
    reccomendation = st.selectbox(label='Choose Reccomendation', options=rec_options)

    if st.button('Get Reccomendations'):

        results = s.sbh(reccomendation=reccomendation)
        st.dataframe(results)
    
elif side_bar == 'Search':

    st.title('Search for Symbols, Quotes, News, Research Reports, Etc.')
    search_query = st.text_area(label='Enter Search Query', placeholder='Donald Trump')

    if st.button('Search'):

        quotes, news, lists, research = q.search(query=search_query)
        st.write('Quotes')
        st.dataframe(quotes)
        st.write('Latest News')
        st.dataframe(news)
        st.write('Yahoo Finance Lists')
        st.dataframe(lists) 
        st.write('Research Reports')
        st.dataframe(research)

