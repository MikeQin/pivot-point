import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import datetime as dt
import pytz
import numpy as np
import altair as alt
# import altair as alt
# from vega_datasets import data
# import r 
# import plotly.graph_objects as go
# import math

from streamlit_autorefresh import st_autorefresh

# update every 5 mins
# st_autorefresh(interval=5 * 60 * 1000, key="dataframerefresh")
# refresh 20 sec
refresh_time = 30 * 1000
now = dt.now(pytz.timezone('US/Eastern'))
today_9_30am = now.replace(hour=9, minute=30, second=0, microsecond=0)
today_4pm = now.replace(hour=16, minute=0, second=0, microsecond=0)
now_str = now.strftime("%m/%d/%Y, %H:%M:%S")

def is_refresh():
  if (today_9_30am.time() < now.time() < today_4pm.time()) and now.weekday() < 5:
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
  agg_df = agg_df.groupby('Close').sum()
  # agg_df = agg_df.groupby('Close')["Volume"].sum().reset_index(name='Volume')
  # agg_df = agg_df.sort_values(by=['Volume'], ascending=False)
  # agg_df = agg_df.set_index('Close')
  # agg_df.dropna()
  return agg_df #.head(30)

st.text('Data date: ' + str(ticker_df.index[-1]) + '. Refreshed: ' 
  + str("{:,.0f}".format(refresh_time/1000)) + ' sec! Time: ' + now_str)

st.subheader('Spot Price: $' + close_price)

def vol_profile_adj(df):
  # Convert input df to input_dict
  input_dict = {}
  for index, vol in df.iterrows():
    input_dict[index] = vol.to_numpy()[0]

  # Create index, key dict reference
  index_ref = {}
  for index, key in enumerate(input_dict):
    index_ref[index] = key

  # Create adjusted strike_vol dictionary
  index_len = len(index_ref)
  adj_dict = {}
  for i, k in index_ref.items():
    if i == 0:
      adj_dict[k] = round((input_dict[index_ref[i]] + input_dict[index_ref[i+1]])/2,0)
    elif i > 0 and i < index_len - 1:
      adj_dict[k] = round((input_dict[index_ref[i-1]] + input_dict[index_ref[i]] + input_dict[index_ref[i+1]])/3,0)
    elif i == index_len - 1:
      adj_dict[k] = round((input_dict[index_ref[i-1]] + input_dict[index_ref[i]])/2,0)

  # return a dataframe 
  key_arr = adj_dict.keys()
  val_arr = adj_dict.values()
  adj_dict = {
    'Close': key_arr,
    'Volume': val_arr
  }
  raw_df = pd.DataFrame.from_dict(adj_dict).set_index('Close')
  return raw_df

def find_poc(vol_prof_df):
  # poc = vol_prof_df.head(1).index.to_list()[0]
  # poc_vol = vol_prof_df.head(1).Volume.to_list()[0]
  poc_vol_row = vol_prof_df[vol_prof_df.Volume == vol_prof_df.Volume.max()]
  poc_vol = round(poc_vol_row.Volume.to_numpy()[0],0)
  poc = round(poc_vol_row.index.to_numpy()[0],2)
  return (poc, poc_vol)

def find_poc_range(vol_prof_df, poc):
  arr = []
  for index, vol in vol_prof_df.iterrows():
    arr.append(index)
  count = 0
  result = []
  for price in arr:
    if price == poc:
      if count >= 2:
        result.append(arr[count - 2])
        # result.append(arr[count - 1])
      if count <= len(arr) - 3:
        # result.append(arr[count + 1])
        result.append(arr[count + 2])
    count += 1
  
  return result

def vp_chart(vol_prof_df, color):
  poc_tuple = find_poc(vol_prof_df)
  poc = poc_tuple[0]
  poc_vol = poc_tuple[1]

  poc_range = find_poc_range(vol_prof_df, poc)

  str_poc_title = '<h5 style="font-family:sans-serif; color:'+color+';">POC: '
  str_poc_vol = str("{:,.0f}".format(poc)) + ', Vol: ' + str("{:,.0f}".format(poc_vol))
  str_poc_range = ', Range: ' +  str(poc_range) + '</h5>'
  poc_html = str_poc_title + str_poc_vol + str_poc_range
  st.write(poc_html, unsafe_allow_html=True)
  # Note here vol_prof_df has index: Close, 1 column: Volume
  # st.bzar_chart(vol_prof_df)
  # st.area_chart(vol_prof_df)

  # Note after reset, vol_prof_df has index: sequence, 2 columns: Close, Volume
  # Altair can't use index, and it must use columns as x, y
  vol_prof_df = vol_prof_df.reset_index()

  title = str("%s Volume Profile, POC: %s, Vol: %s" % (ticker, poc, str("{:,.0f}".format(poc_vol))))

  vp_chart = alt.Chart(vol_prof_df, title=title).mark_bar(opacity=0.8).encode(x='Close:O', y='Volume', color=alt.condition(
          alt.datum.Close == poc,  # If the year is 1810 this test returns True,
          alt.value(color),     # which sets the bar orange.
          alt.value('steelblue')   # And if it's not true it sets the bar steelblue.
      ))
  
  return vp_chart

# 1. Volume Profile Chart
# Note here vol_prof_df has index: Close, 1 column: Volume
vol_prof_df = vol_profile(ticker_df)
# Plot chart
vol_profile_basic_chart = vp_chart(vol_prof_df, '#0b9e12')
st.altair_chart(vol_profile_basic_chart, use_container_width=True)
# 2. Adjusted Volume Profile Chart
adjusted_vol_profile = vol_profile_adj(vol_prof_df)
# Plot chart
adjusted_vol_profile_chart = vp_chart(adjusted_vol_profile, '#fc5e03')
st.altair_chart(adjusted_vol_profile_chart, use_container_width=True)

# Select Max Vol Row
max_vol_row = ticker_df[ticker_df.Volume == ticker_df.Volume.max()]
max_vol = round(max_vol_row.Volume.to_numpy()[0],0)
max_vol_price = round(max_vol_row.Close.to_numpy()[0],2)
# print('Max Strike and Vol: ', max_vol, max_vol_price)
st.markdown('#### Max Vol Price: %s, Volume: %s' % (str("{:,.2f}".format(max_vol_price)), str("{:,.0f}".format(max_vol))))
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
    # 'useOpen': useOpen,
    'PP':  [str("{:,.2f}".format(pp))], 
    'R1': [str("{:,.2f}".format(r1))], 
    'R2': [str("{:,.2f}".format(r2))],
    'R3': [str("{:,.2f}".format(r3))],
    # 'R4': [str("{:,.2f}".format(r4))], 
    'S1':    [str("{:,.2f}".format(s1))], 
    'S2':    [str("{:,.2f}".format(s2))], 
    'S3':    [str("{:,.2f}".format(s3))]
    # 'S4':    [str("{:,.2f}".format(s4))]
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
    'Range': [str("{:,.2f}".format(abs(tc-bc)))],
    'Top': [str("{:,.2f}".format(tc))],
    # 'Middle PP': [str("{:,.2f}".format(pp))],
    'Bottom': [str("{:,.2f}".format(bc))]
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

# Remove Hamburger Menu, and Footer
st.markdown("""
<style>
.css-9s5bis {
  visibility: hidden;
}
.css-h5rgaw {
  visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# Footer
# .css-h5rgaw {
#   visibility: hidden;
# }
# .viewerBadge_link__1S137 {
#   visibility: hidden;
# }
# ##########################################
# def create_strikes(n1, n2):
#   n1 = int(n1 / 10)
#   n2 = int(n2 / 10)

#   arr = []
#   for x in range (n1 * 10, n2 * 10, 20):
#     arr.append(x)

#   return arr
if ticker == '^GSPC':
  ticker = '^SPX'
spx_data = yf.Ticker(ticker)
# spot_price = spx_data.info['regularMarketPrice']
spot_price = round(ticker_close, 2)

drop_rows = []
for i in range(0, 11):
  drop_rows.append(i)
title = str('%s: %s, Options for ' % (ticker, spot_price))
legend = ", VOL: GR, OI: BO, " + now_str

for i in range (5):
  expiry_date = spx_data.options[i]
  spx_option_chain = spx_data.option_chain(expiry_date)
  calls_df = spx_option_chain.calls
  puts_df = spx_option_chain.puts
  # call max vol row
  calls_max_vol_row = spx_option_chain.calls[spx_option_chain.calls.volume == spx_option_chain.calls.volume.max()]
  calls_max_vol_strike = calls_max_vol_row.strike.to_numpy()[0]
  calls_max_vol = calls_max_vol_row.volume.to_numpy()[0]
  # put max vol row
  puts_max_vol_row = spx_option_chain.puts[spx_option_chain.puts.volume == spx_option_chain.puts.volume.max()]
  puts_max_vol_strike = puts_max_vol_row.strike.to_numpy()[0]
  puts_max_vol = puts_max_vol_row.volume.to_numpy()[0]
  trend_json = {
    "strike": [calls_max_vol_strike, puts_max_vol_strike],
    "volume": [calls_max_vol, puts_max_vol]
  }
  # Create trend DF
  trend_df = pd.DataFrame(trend_json)

  # from_strike = 0.8 * spot_price # 0.8
  # to_strike = 1.2 * spot_price # 1.2
  # x_strikes = create_strikes(from_strike, to_strike)

  # for i in range(105, 115):
  #   drop_rows.append(i)

  # call_chart = alt.Chart(calls_df.drop(drop_rows)).mark_bar().encode(alt.X('strike', axis=alt.Axis(values=x_strikes)), y='volume', color=alt.value("green"))
  # put_chart = alt.Chart(puts_df.drop(drop_rows)).mark_bar().encode(alt.X('strike', axis=alt.Axis(values=x_strikes)), y='volume', color=alt.value("red"))
  
  # Option Chart
  call_vol_chart = alt.Chart(calls_df.drop(drop_rows), title=title + expiry_date + legend).mark_line(opacity=0.5).encode(x='strike', y='volume', color=alt.value("green"))
  call_oi_chart = alt.Chart(calls_df.drop(drop_rows)).mark_line(opacity=0.5).encode(x='strike', y='openInterest', color=alt.value("steelblue"))
  put_vol_chart = alt.Chart(puts_df.drop(drop_rows)).mark_line(opacity=0.5).encode(x='strike', y='volume', color=alt.value("#FF3D3A"))
  put_oi_chart = alt.Chart(puts_df.drop(drop_rows)).mark_line(opacity=0.5).encode(x='strike', y='openInterest', color=alt.value("orange"))
  max_vol_points = alt.Chart(trend_df).mark_circle(size=60).encode(x='strike', y='volume', color=alt.value("black"))
  xrule = alt.Chart(calls_df).mark_rule(color="blue", opacity=0.5).encode(x=alt.datum(spot_price))
  labels = max_vol_points.mark_text(
      align='center',
      # baseline='middle',
      dx=1, dy=-20,
      fontStyle='regular',
      fontSize=16
    ).encode(
      text=alt.Text('strike')
    )
  # st.markdown('##### Option Volume Live: ' + expiry_date)
  st.altair_chart(
    call_vol_chart + 
    put_vol_chart + 
    call_oi_chart +
    put_oi_chart +
    xrule + 
    max_vol_points + 
    labels,
    use_container_width=True
  )

  # print(trend_df)
