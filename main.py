import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import datetime as dt
import pytz
# import altair as alt
# from vega_datasets import data
# import r 
# import plotly.graph_objects as go
# import math

from streamlit_autorefresh import st_autorefresh

# update every 5 mins
# st_autorefresh(interval=5 * 60 * 1000, key="dataframerefresh")
# refresh 20 sec
refresh_time = 20 * 1000
now = dt.now()
today_8_30am = now.replace(hour=8, minute=30, second=0, microsecond=0)
today_4pm = now.replace(hour=16, minute=0, second=0, microsecond=0)

def is_refresh():
  if (today_8_30am.time() < now.time() < today_4pm.time()) and now.weekday() < 5:
    return True
  return False

if is_refresh():
  st_autorefresh(interval=refresh_time, key="dataframerefresh")

# https://en.wikipedia.org/wiki/Pivot_point_(technical_analysis)

pd.options.display.float_format = "{:,.2f}".format

# Sidebar section
ticker = '^GSPC'
# def submit_cb():
#   ticker = user_ticker
st.sidebar.write('# Enter Ticker')
user_ticker = st.sidebar.text_input('Ticker (Symbol)', value='^GSPC') # , on_change=submit_cb

if user_ticker:
  ticker = user_ticker.upper()

# print('Ticker', ticker)

# Main section
if ticker == '^GSPC':
  st.subheader('SPX (' + ticker + ') 0DT Price Range')
else:
  st.subheader(ticker + ' 0DT Price Range')

f_data = yf.Ticker(ticker)
ticker_df = f_data.history(period='1d', interval='1m')
# print('LAST TWO: ', ticker_df.tail(2))
# print(ticker_df.info())

ticker_open = ticker_df.Open[0]
ticker_high = ticker_df.High.max()
ticker_low = ticker_df.Low.min()
ticker_close = ticker_df.Close[-1]
close_price = str("{:,.2f}".format(ticker_close))
# print(ticker_open, ticker_high, ticker_low, ticker_close)

def vol_profile(df):
  agg_df = df.copy()
  agg_df = agg_df.drop(['Open', 'High', 'Low', 'Dividends', 'Stock Splits'], axis=1)
  agg_df['Close'] = agg_df['Close'].astype(int)
  agg_df = agg_df.groupby('Close').sum().reset_index()
  agg_df = agg_df.sort_values(by=['Volume'], ascending=False)
  agg_df = agg_df.set_index('Close')
  # print(agg_df.head(20))
  return agg_df.head(30)

# def get_high_vol_close(df, high_vol):
#   hv_df = df.where(df.Volume >= high_vol)
#   close = hv_df.Close.max()
#   return close

st.text('Data date: ' + str(ticker_df.index[-1]) + '. Refreshed: ' 
  + str("{:,.0f}".format(refresh_time/1000)) + ' sec! Time: ' + str(now.time()))

st.subheader('Spot Price: $' + close_price)

vol_prof_df = vol_profile(ticker_df)
poc = vol_prof_df.head(1).index.to_list()[0]
poc_vol = vol_prof_df.head(1).Volume.to_list()[0]

poc_html = '<p style="font-family:sans-serif; color:Red;">POC: ' + str("{:,.0f}".format(poc)) + ', Vol: ' + str("{:,.0f}".format(poc_vol)) + '</p>'
st.write(poc_html, unsafe_allow_html=True)
st.bar_chart(vol_prof_df)

# Select Max Vol Row
max_vol_row = ticker_df[ticker_df.Volume == ticker_df.Volume.max()]
max_vol = round(max_vol_row.Volume.to_numpy()[0],0)
max_vol_price = round(max_vol_row.Close.to_numpy()[0],2)
# print('Result: ',max_vol, max_vol_price)
# print("%s %s" % ("Hello", "World"))
st.markdown('#### Closing Price: %s, Volume: %s' % (str("{:,.2f}".format(max_vol_price)), str("{:,.0f}".format(max_vol))))
st.line_chart(data=ticker_df, x='Close', y='Volume')

# https://tradingfuel.com/pivot-point-calculator-and-strategy/
# https://www.pivotpointcalculator.com/
# Calculate PP, R, S
def calc_pp_r_s(open, high, low, close, useOpen):

  pp = None
  if useOpen:
    pp = (open + high + low + close)/4
  else:
    pp = (high + low + close)/3

  r1 = (2 * pp) - low
  s1 = (2 * pp) - high
  r2 = (pp - s1) + r1
  s2 = pp - (r1 - s1)
  r3 = high + 2 * (pp - low) # r3 = (pp - s2) + r2
  s3 = low - 2 * (high - pp) # s3 = pp - (r2 -s2)
  r4 = high + 3 * (pp - low) # r3 = (pp - s2) + r2
  s4 = low - 3 * (high - pp) # s3 = pp - (r2 -s2)
  result = {
    'useOpen': useOpen,
    'PP':  [str("{:,.2f}".format(pp))], 
    'R1': [str("{:,.2f}".format(r1))], 
    'R2': [str("{:,.2f}".format(r2))],
    'R3': [str("{:,.2f}".format(r3))],
    'R4': [str("{:,.2f}".format(r4))], 
    'S1':    [str("{:,.2f}".format(s1))], 
    'S2':    [str("{:,.2f}".format(s2))], 
    'S3':    [str("{:,.2f}".format(s3))],
    'S4':    [str("{:,.2f}".format(s4))]
  }

  return result

pp_rs = calc_pp_r_s(ticker_open, ticker_high, ticker_low, ticker_close, False)
df = pd.DataFrame(pp_rs)

pp4_rs = calc_pp_r_s(ticker_open, ticker_high, ticker_low, ticker_close, True)
df4 = pd.DataFrame(pp4_rs)

# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)
# Display a static table
st.markdown('##### Pivot Point (high, low, close)')
st.table(df)
# st.markdown('##### Pivot Point (open, high, low, close)')
# st.table(df4)
# Bar Chart
# st.bar_chart(df)

# Central Pivot Range
# Pivot point = (High + Low + Close) / 3
# Top Central Pivot Point (BC) = (Pivot – BC) + Pivot
# Bottom Central Pivot Point (TC) = (High + Low) / 2

def calc_cpr(high, low, close):
  pp = (high + low + close)/3
  # BC: Bottom Central
  # Lower boundary
  bc = (high + low) / 2
  # TC: Top Central
  # Upper boundary
  tc = (pp - bc) + pp
  
  cpr = {
    'Top PP': [str("{:,.2f}".format(tc))],
    'Middle PP': [str("{:,.2f}".format(pp))],
    'Bottom PP': [str("{:,.2f}".format(bc))]
  }
  return cpr

cpr_dict = calc_cpr(ticker_high, ticker_low, ticker_close)
cpr_df = pd.DataFrame(cpr_dict)

# Display dataframe table
st.markdown('##### Central Pivot Range (CPR)')
st.table(cpr_df)
# st.bar_chart(cpr_df)

# Line Chart
# st.line_chart(ticker_df.Close)
#------------
# line_chart = alt.Chart(ticker_df.Close).mark_line() #.encode(alt.Y(scale=alt.Scale(zero=False)))
# st.altair_chart(line_chart, use_container_width=True)

# alt.Chart(cars).mark_point().encode(
#     alt.X('Acceleration:Q',
#         scale=alt.Scale(domain=(5, 20))
#     ),
#     y='Horsepower:Q'
# )

# Long period
df_1y = f_data.history(period='1y', interval='1d')

# Candlestick Chart
# fig = go.Figure(data=[go.Candlestick(x=df_1y.index,
#                                    open=df_1y['Open'],
# high=df_1y['High'],
# low=df_1y['Low'],
# close=df_1y['Close'])])
# fig.show()

# Calculate ATR
def calc_atr(df, period):
  # print('length', len(df_1y.Close))
  working_range = period + 2
  high = df.High[-working_range:]
  low = df.Low[-working_range:]
  close = df.Close[-working_range:]
  date = df.index[-working_range:]
  # print('date', df_1y.index[-9:])
  # Calcualate previous 14 day's ATR
  tr_sum = 0
  for i in range(1, working_range-1):
    tr = max((high[i] - low[i]), abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr_sum += tr
    # print(i, date[i], tr)
  atr = tr_sum / period

  # Calculate current TR
  curr = working_range-1
  curr_tr = max((high[curr] - low[curr]), abs(high[curr] - close[curr-1]), abs(low[curr] - close[curr-1]))
  # print(curr, date[curr], curr_tr)
  smoothed_atr = round((atr * (period -1) + curr_tr) / period, 2)

  # print(round(atr, 2), smoothed_atr)
  return smoothed_atr

st.sidebar.title('Average True Range')
num = 14
user_num = st.sidebar.slider('ATR: Select days in the period', 1, 200, num, 1)
if user_num:
  num = int(user_num)
atr = calc_atr(df_1y, num)
atr_1d = calc_atr(df_1y, 1)

mid_point = round(atr/2, 2)
upper_bound = str(round((ticker_close + mid_point), 2))
lower_bound = str(round((ticker_close - mid_point), 2))
max_today = str("{:,.2f}".format(ticker_high))
min_today = str("{:,.2f}".format(ticker_low))
atr_today = str("{:,.2f}".format(atr_1d))

# print('atr', atr)
st.markdown('#### ATR (' + str(num) +'-day): ' + str(atr))
st.markdown('##### Today ATR:' + atr_today + ', High: ' + max_today  + ', Low: ' + min_today)
st.markdown('##### Dynamic Range: ' + upper_bound + '~' + lower_bound)
# st.markdown('#### Today Volume')

# st.line_chart(data=ticker_df, x='Close', y='Volume')

# print(ticker_df.index)
# print(ticker_df.tail(3))
# print(ticker_df.info())

# Print Charts
# st.line_chart(df_1y.Close)
# st.subheader('Volume Chart')
# st.line_chart(df_1y.Volume)