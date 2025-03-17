import streamlit as st
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go
import ollama
from ollama import chat
import os
import yfinance as yf

class TickerAnalysis():

    def __init__(self):

        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.client = ollama.Client(host=self.OLLAMA_BASE_URL)

    def llm(self, model='llama3.2', prompt=None):

        OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        client = ollama.Client(host=OLLAMA_BASE_URL)
        stream = client.chat(
            model=model,
            messages=[
                {'role': 'user',
                'content': f'{prompt}'}],
            stream=True
        )

        for chunk in stream:
            text = chunk['message']['content']
            yield text  # Stream each chunk instead of returning only the first one

    def landing(self):

        us_tickers = ['^DJI', '^GSPC', '^IXIC', '^RUT', 'CL=F', 'GC=F']
        eu_tickers = ['^FTSE', '^FCHI', '^GDAXI', '^N100', 'EURUSD=X', 'GBP=X']
        as_tickers = ['000001.SS', '^N225', '^HSI', '^AXJO', '^ADOW', 'JPY=X']
        rates = ['^IRX', '^FVX', '^TNX', '^TYX', '2YY=F', 'ZN=F']

        us_markets = yf.download(tickers=us_tickers, period='6mo')
        eu_markets = yf.download(tickers=eu_tickers, period='6mo')
        as_markets = yf.download(tickers=as_tickers, period='6mo')
        rates = yf.download(tickers=rates, period='6mo')

        us_markets = us_markets.reset_index()
        us_markets.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in us_markets.columns]

        eu_markets = eu_markets.reset_index()
        eu_markets.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in eu_markets.columns]

        as_markets = as_markets.reset_index()
        as_markets.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in as_markets.columns]

        rates = rates.reset_index()
        rates.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in rates.columns]

        fig_1 = px.line(us_markets, x='Date_', y = ['Close_^DJI', 'Close_^GSPC', 'Close_^IXIC', 'Close_^RUT', 'Close_CL=F', 'Close_GC=F'], markers=True, title='US Indicies')
        fig_2 = px.line(eu_markets, x='Date_', y = ['Close_^FTSE', 'Close_^FCHI', 'Close_^GDAXI', 'Close_^N100', 'Close_EURUSD=X', 'Close_GBP=X'], markers=True, title='EU Indicies')
        fig_3 = px.line(as_markets, x='Date_', y = ['Close_000001.SS', 'Close_^N225', 'Close_^HSI', 'Close_^AXJO', 'Close_^ADOW', 'Close_JPY=X'], markers=True, title='Asian Indicies')
        fig_4 = px.line(rates, x='Date_', y=['Close_^IRX', 'Close_^FVX', 'Close_^TNX', 'Close_^TYX', 'Close_2YY=F', 'Close_ZN=F'], markers=True, title='Rates')

        return fig_1, fig_2, fig_3, fig_4
    
    def ticker_history(self, symbol=None, timeframe='1y'):

        data = yf.Ticker(symbol)
        history = data.history(period=timeframe).reset_index()
        history['Date'] = pd.to_datetime(history['Date'])
        history = history.sort_values(by='Date', ascending=False)
        return history
    
    def ticker_info(self, symbol=None):

        data = yf.Ticker(symbol)
        data = pd.json_normalize(data.info)
        transposed_data = data.transpose().reset_index()
        transposed_data.columns = ["attributes", f"{symbol}"]  # Rename columns
        return transposed_data
    
    def ticker_sustainability(self, symbol=None):

        data = yf.Ticker(symbol)
        esg_data = data.sustainability
        return esg_data
    
    def candlestick(self, symbol=None, timeframe=None):

        df = self.ticker_history(symbol=symbol, timeframe=timeframe)

        fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
        
        fig.update_layout(
            title=f"{symbol} Stock Price Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Stock Price",
            xaxis_rangeslider_visible=False
        )

        return fig
        
    def long_term_moving(self, symbol=None, timeframe=None):

        hist = self.ticker_history(symbol=symbol, timeframe=timeframe)

        # Compute Moving Averages
        hist['Rolling'] = hist['Close'].rolling(window=200).mean()
        hist['EWMA'] = hist['Close'].ewm(span=200, adjust=False).mean()

        # Detect Buy/Sell Signals
        hist['Buy Signal'] = (hist['Close'] > hist['Rolling']) & (hist['Close'].shift(1) <= hist['Rolling'])
        hist['Sell Signal'] = (hist['Close'] < hist['Rolling']) & (hist['Close'].shift(1) >= hist['Rolling'])

        # Plot the Moving Averages & Signals
        fig = px.line(hist, x='Date', y=['Close', 'Rolling', 'EWMA'], title="Long Term Strategy: Moving Averages")

        # Add Buy/Sell Points
        buy_points = hist.loc[hist['Buy Signal'], ['Date', 'Close']]
        sell_points = hist.loc[hist['Sell Signal'], ['Date', 'Close']]
        fig.add_scatter(x=buy_points['Date'], y=buy_points['Close'], mode='markers', marker=dict(color='green', size=10), name='Buy Signal')
        fig.add_scatter(x=sell_points['Date'], y=sell_points['Close'], mode='markers', marker=dict(color='red', size=10), name='Sell Signal')

        return fig

    def short_term_moving(self, symbol=None, timeframe=None):

        hist = self.ticker_history(symbol=symbol, timeframe=timeframe)

        # Compute Short-Term Moving Averages
        hist['Rolling'] = hist['Close'].rolling(window=15).mean()
        hist['EWMA'] = hist['Close'].ewm(span=15, adjust=False).mean()

        # Detect Buy/Sell Signals
        hist['Buy Signal'] = (hist['Close'] > hist['Rolling']) & (hist['Close'].shift(1) <= hist['Rolling'])
        hist['Sell Signal'] = (hist['Close'] < hist['Rolling']) & (hist['Close'].shift(1) >= hist['Rolling'])

        # Plot Moving Averages & Signals
        fig = px.line(hist, x='Date', y=['Close', 'Rolling', 'EWMA'], title="Short Term Strategy: Moving Averages")

        # Add Buy/Sell Points
        buy_points = hist.loc[hist['Buy Signal'], ['Date', 'Close']]
        sell_points = hist.loc[hist['Sell Signal'], ['Date', 'Close']]
        fig.add_scatter(x=buy_points['Date'], y=buy_points['Close'], mode='markers', marker=dict(color='green', size=10), name='Buy Signal')
        fig.add_scatter(x=sell_points['Date'], y=sell_points['Close'], mode='markers', marker=dict(color='red', size=10), name='Sell Signal')

        return fig

    def analyst_price_targets(self, symbol=None):

        data = yf.Ticker(symbol)
        price_targets = data.analyst_price_targets
        return price_targets

    def analyst_reccomendations(self, symbol=None):

        data = yf.Ticker(symbol)
        reccomendations = data.recommendations
        return reccomendations

    def upgrades_downgrades(self, symbol=None):

        data = yf.Ticker(symbol)
        updown = getattr(data, "upgrades_downgrades", None)
        if updown is None or updown.empty:
            print(f"No upgrade/downgrade data available for {symbol}")
            return pd.DataFrame()  # Return an empty DataFrame
        updown = updown.reset_index()
        if "GradeDate" in updown.columns:
            updown["GradeDate"] = pd.to_datetime(updown["GradeDate"])
            updown = updown.sort_values(by="GradeDate", ascending=False)
        else:
            print(f"'GradeDate' column missing for {symbol}. Returning raw data.")
        return updown

    def ticker_news(self, symbol=None):

        data = yf.Ticker(symbol)
        news_data = getattr(data, "news", None)
        if not news_data:
            print(f"No news data available for {symbol}")
            return pd.DataFrame()  # Return an empty DataFrame
        news = pd.json_normalize(news_data)
        if "content.pubDate" in news.columns:
            news["content.pubDate"] = pd.to_datetime(news["content.pubDate"], errors="coerce")
            news = news.sort_values(by="content.pubDate", ascending=False)
            news = news[['content.provider.displayName', 'content.title', 'content.contentType', 'content.summary', 'content.pubDate', 'content.canonicalUrl.url']]
            news.columns = ['Publisher', 'Title', 'Type', 'Summary', 'Publication Date', 'URL']
        else:
            print(f"'content.pubDate' column missing for {symbol}. Returning raw news data.")
        return news

    def ticker_news_list(self, symbol=None):

        content_list = []
        data = self.ticker_news(symbol=symbol)
        if not data.empty and "Summary" in data.columns:
            content_list = data["Summary"].tolist()
        else:
            print(f"No valid news summaries found for {symbol}")
        return content_list

    def volatility(self, symbol=None, timeframe=None):

        history = self.ticker_history(symbol=symbol, timeframe=timeframe)
        history['Daily Return'] = history['Close'].pct_change()
        history['Volatility'] = history['Daily Return'].rolling(window=30).std()

        return history

    def volatility_plot(self, symbol=None, timeframe=None):
        
        data = self.volatility(symbol=symbol, timeframe=timeframe)
        fig = px.line(data, x = 'Date', y = 'Volatility', title=f'{symbol} Volatility Over Time')

        return fig
        
    def insider_transactions(self, symbol=None):

        data = yf.Ticker(symbol)
        transactions = data.insider_transactions

        return transactions

    def insitutional_holders(self, symbol=None):

        data = yf.Ticker(symbol)
        institutional_holders = data.institutional_holders

        return institutional_holders

    def cash_flow(self, symbol=None):

        data = yf.Ticker(symbol)
        cf = data.cash_flow
        return cf

    def balance_sheet(self, symbol=None):

        data = yf.Ticker(symbol)
        bs = data.balance_sheet
        return bs

    def income_statements(self, symbol=None):

        data = yf.Ticker(symbol)
        incs = data.income_stmt

        return incs

    def top_holdings(self, symbol=None):

        data = yf.Ticker(symbol)
        top_holders = data.funds_data.top_holdings.reset_index()
        return top_holders

    def fund_holdings(self, symbol=None):

        try:
            dat = yf.Ticker(symbol)
            if hasattr(dat, 'funds_data') and hasattr(dat.funds_data, 'top_holdings'):
                fund_holdings = dat.funds_data.top_holdings
                if fund_holdings is not None and not fund_holdings.empty:
                    return fund_holdings.reset_index()
                else:
                    print(f"⚠️ No fund holdings data found for {symbol}.")
                    return None
            else:
                print(f"⚠️ No fund holdings attribute found for {symbol}.")
                return None
        except Exception as e:
            print(f"🚨 Error fetching fund holdings for {symbol}: {e}")
            return None
    
    def earnings_estimate(self, symbol=None):

        data = yf.Ticker(symbol)
        earnings = data.earnings_estimate.reset_index()
        return earnings

    def sec_filings(self, symbol=None):

        dat = yf.Ticker(symbol)
        filings = dat.sec_filings
        filings_df = pd.json_normalize(filings)
        return filings_df

class SectorAnalysis():

    def __init__(self):
        pass

    def get_sectors_and_industries(self):

        sectors = {
            "basic-materials": [
                "agricultural-inputs", "aluminum", "building-materials", "chemicals", "coking-coal",
                "copper", "gold", "lumber-wood-production", "other-industrial-metals-mining",
                "other-precious-metals-mining", "paper-paper-products", "silver", "specialty-chemicals",
                "steel"
            ],
            "communication-services": [
                "advertising-agencies", "broadcasting", "electronic-gaming-multimedia",
                "entertainment", "internet-content-information", "publishing", "telecom-services"
            ],
            "consumer-cyclical": [
                "apparel-manufacturing", "apparel-retail", "auto-manufacturers", "auto-parts",
                "auto-truck-dealerships", "department-stores", "footwear-accessories",
                "furnishings-fixtures-appliances", "gambling", "home-improvement-retail",
                "internet-retail", "leisure", "lodging", "luxury-goods", "packaging-containers",
                "personal-services", "recreational-vehicles", "residential-construction",
                "resorts-casinos", "restaurants", "specialty-retail", "textile-manufacturing",
                "travel-services"
            ],
            "consumer-defensive": [
                "beverages-brewers", "beverages-non-alcoholic", "beverages-wineries-distilleries",
                "confectioners", "discount-stores", "education-training-services", "farm-products",
                "food-distribution", "grocery-stores", "household-personal-products", "packaged-foods",
                "tobacco"
            ],
            "energy": [
                "oil-gas-drilling", "oil-gas-e-p", "oil-gas-equipment-services", "oil-gas-integrated",
                "oil-gas-midstream", "oil-gas-refining-marketing", "thermal-coal", "uranium"
            ],
            "financial-services": [
                "asset-management", "banks-diversified", "banks-regional", "capital-markets",
                "credit-services", "financial-conglomerates", "financial-data-stock-exchanges",
                "insurance-brokers", "insurance-diversified", "insurance-life",
                "insurance-property-casualty", "insurance-reinsurance", "insurance-specialty",
                "mortgage-finance", "shell-companies"
            ],
            "healthcare": [
                "biotechnology", "diagnostics-research", "drug-manufacturers-general",
                "drug-manufacturers-specialty-generic", "health-information-services",
                "healthcare-plans", "medical-care-facilities", "medical-devices",
                "medical-distribution", "medical-instruments-supplies", "pharmaceutical-retailers"
            ],
            "industrials": [
                "aerospace-defense", "airlines", "airports-air-services",
                "building-products-equipment", "business-equipment-supplies", "conglomerates",
                "consulting-services", "electrical-equipment-parts", "engineering-construction",
                "farm-heavy-construction-machinery", "industrial-distribution",
                "infrastructure-operations", "integrated-freight-logistics", "marine-shipping",
                "metal-fabrication", "pollution-treatment-controls", "railroads",
                "rental-leasing-services", "security-protection-services",
                "specialty-business-services", "specialty-industrial-machinery",
                "staffing-employment-services", "tools-accessories", "trucking", "waste-management"
            ],
            "real-estate": [
                "real-estate-development", "real-estate-diversified", "real-estate-services",
                "reit-diversified", "reit-healthcare-facilities", "reit-hotel-motel",
                "reit-industrial", "reit-mortgage", "reit-office", "reit-residential",
                "reit-retail", "reit-specialty"
            ],
            "technology": [
                "communication-equipment", "computer-hardware", "consumer-electronics",
                "electronic-components", "electronics-computer-distribution",
                "information-technology-services", "scientific-technical-instruments",
                "semiconductor-equipment-materials", "semiconductors", "software-application",
                "software-infrastructure", "solar"
            ],
            "utilities": [
                "utilities-diversified", "utilities-independent-power-producers",
                "utilities-regulated-electric", "utilities-regulated-gas",
                "utilities-regulated-water", "utilities-renewable"
            ]
        }

        data = [(sector, industry) for sector, industries in sectors.items() for industry in industries]
        df = pd.DataFrame(data, columns=["Sector", "Industry"])
        return df

    def sector_overview(self, sector_name=None):

        sector = yf.Sector(key=sector_name)
        overview = sector.overview
        data = pd.json_normalize(overview)
        transposed_data = data.transpose().reset_index()
        transposed_data.columns = ["attributes", f"{sector_name}"]  # Rename columns
        
        return transposed_data
    
    def top_sector_companies(self, sector_name=None):

        sector = yf.Sector(key=sector_name)
        top_companies = sector.top_companies
        return top_companies

    def top_sector_etfs(self, sector_name=None):

        sector = yf.Sector(key=sector_name)
        top_companies = sector.top_etfs
        return top_companies

    def top_sector_mutual_funds(self, sector_name=None):

        sector = yf.Sector(key=sector_name)
        top_companies = sector.top_mutual_funds
        return top_companies
    
    def sector_research_reports(self, sector_name=None):

        sector = yf.Sector(key=sector_name)
        srr = sector.research_reports
        return srr
    
    def industry_overview(self, industry_name=None):

        indsutry = yf.Industry(key=industry_name)
        tc = indsutry.overview
        data = pd.json_normalize(tc)
        transposed_data = data.transpose().reset_index()
        transposed_data.columns = ["attributes", f"{industry_name}"]  # Rename columns
        return transposed_data
    
    def industry_research_reports(self, industry_name=None):

        indsutry = yf.Industry(key=industry_name)
        rr = indsutry.research_reports
        return rr

    def top_industry_companies(self, industry_name=None):

        indsutry = yf.Industry(key=industry_name)
        tc = indsutry.top_companies
        return tc

    def top_industry_performing_companies(self, industry_name=None):

        indsutry = yf.Industry(key=industry_name)
        tpc = indsutry.top_performing_companies
        return tpc

    def top_industry_growth_companies(self, industry_name=None):

        indsutry = yf.Industry(key=industry_name)
        tgc = indsutry.top_growth_companies
        return tgc

    def sbh(self, reccomendation=None):

        companies_df = pd.DataFrame()

        d = self.get_sectors_and_industries()

        for x in d.Sector.unique():
            dat = yf.Sector(x)
            s_companies = dat.top_companies
            s_companies = s_companies.reset_index()
            companies_df = pd.concat([s_companies, companies_df])

        for y in d.Industry.unique():
            dat = yf.Industry(y)
            i_companies = dat.top_companies
            try:
                i_companies = i_companies.reset_index()
            except AttributeError as e:
                continue
            companies_df = pd.concat([i_companies, companies_df])
        companies_df = companies_df.sort_values(by='rating')

        if reccomendation == 'Strong Buy':
            companies_df = companies_df[(companies_df['rating'] == 'Strong Buy')]
        elif reccomendation == 'Buy':
            companies_df = companies_df[(companies_df['rating'] == 'Buy')]
        elif reccomendation == 'Hold':
            companies_df = companies_df[(companies_df['rating'] == 'Hold')]
        elif reccomendation == 'Sell':
            companies_df = companies_df[(companies_df['rating'] == 'Sell')]
        elif reccomendation == 'Underperform':
            companies_df = companies_df[(companies_df['rating'] == 'Underperform')]
        else:
            return companies_df
        return companies_df

class MarketAnalysis():

    def __init__(self):
        pass

    def get_markets(self):

        markets = ["US", "GB", "ASIA", "EUROPE", "RATES", "COMMODITIES", "CURRENCIES", "CRYPTOCURRENCIES"]
        return markets
    
    def market_summary(self, market_name=None):

        market = yf.Market(market=market_name)
        summary = market.summary
        market_summary = pd.json_normalize(summary)
        market_summary = market_summary.transpose()
        return market_summary

    def market_status(self, market_name=None):

        market = yf.Market(market=market_name)
        status = market.status
        market_status = pd.json_normalize(status)
        market_status = market_status.transpose()
        return market_status

class Query():

    def __init__(self):
        pass

    def search(self, query=None):

        search = yf.Search(query=query).all
        quotes = pd.json_normalize(search['quotes'])
        news = pd.json_normalize(search['news'])
        lists = pd.json_normalize(search['lists'])
        research = pd.json_normalize(search['research'])
        return quotes, news, lists, research

class LLM():

    def __init__(self):
        pass
    
    def llm(self, prompt, model='llama3.2'):
        """Run the Llama 3.2 model using Ollama and return the result."""
        """Run the gemma:7b model using Ollama and return the result."""

        stream = chat(
            model=model,
            messages=[
                {'role': 'user',
                 'content': f'{prompt}'}],
            stream=True
        )

        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)

