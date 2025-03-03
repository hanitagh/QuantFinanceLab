# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:20:47 2024

@author: manas
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Read the CSV file into a DataFrame
nfp_data = pd.read_csv("C:/Users/manas/Downloads/nfp.csv")
adp_data = pd.read_csv("C:/Users/manas/Downloads/ADP.csv")

# Print the DataFrame
print(nfp_data)
print(adp_data)

def fetch_intraday_data(ticker, start_date, end_date):
    # Fetch historical intraday data
    data = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval='60m',  # 1-minute interval data
    )
    print("data")
    print(data)
    # Filter data between 8:30 AM and 9:30 AM
    data['time'] = data.index.time
    data['Date'] = data.index.date
    filtered_data = data.between_time('08:00', '10:00')
  # Add a column to determine rise or fall status
    def calculate_status(row):
        if row['Close'] > row['Open']:
            return 'Rise'
        elif row['Close'] < row['Open']:
            return 'Fall'
        else:
            return 'Neutral'

    filtered_data['status'] = filtered_data.apply(calculate_status, axis=1)
    
    return filtered_data
# Define start and end dates
start_date = '2023-01-01'
end_date = '2024-11-03'

# Fetch data for USD/JPY (symbol: JPY=X), Gold (symbol: GC=F), and Oil (symbol: CL=F)
usdjpy_data = fetch_intraday_data('JPY=X', start_date, end_date)
gold_data = fetch_intraday_data('GC=F', start_date, end_date)
oil_data = fetch_intraday_data('CL=F', start_date, end_date)

def calculate_hypothesis(data):
    # Ensure 'Actual', 'Previous', and 'Forecast' columns are present in the data
    data['Actual-Previous'] = data['Actual'] - data['Previous']
    data['Actual-Forecast'] = data['Actual'] - data['Forecast']
    
    # Create 'Hypothesis 1' and 'Hypothesis 2' based on the difference
    data['Hypothesis_1'] = data['Actual-Previous'].apply(lambda x: 'positive' if x > 0 else 'negative')
    data['Hypothesis_2'] = data['Actual-Forecast'].apply(lambda x: 'positive' if x > 0 else 'negative')
    
    return data[['Date', 'Actual-Previous', 'Actual-Forecast', 'Hypothesis_1', 'Hypothesis_2']]

# Apply the function to both NFP and ADP datasets
nfp_hypothesis = calculate_hypothesis(nfp_data)
adp_hypothesis = calculate_hypothesis(adp_data)
print(nfp_hypothesis['Hypothesis_2'].value_counts())
print (nfp_hypothesis)
print (adp_hypothesis)
print(usdjpy_data)
nfp_hypothesis['Hypothesis_1_Status'] = nfp_hypothesis['Hypothesis_1'].map({'positive': 1, 'negative': -1, 'neutral': 0})
nfp_hypothesis['Hypothesis_2_Status'] = nfp_hypothesis['Hypothesis_2'].map({'positive': 1, 'negative': -1, 'neutral': 0})
adp_hypothesis['Hypothesis_1_Status'] = adp_hypothesis['Hypothesis_1'].map({'positive': 1, 'negative': -1, 'neutral': 0})
adp_hypothesis['Hypothesis_2_Status'] = adp_hypothesis['Hypothesis_2'].map({'positive': 1, 'negative': -1, 'neutral': 0})
nfp_hypothesis['Date'] = pd.to_datetime(nfp_hypothesis['Date'])
adp_hypothesis['Date'] = pd.to_datetime(adp_hypothesis['Date'],errors='coerce')
usdjpy_data['Date'] = pd.to_datetime(usdjpy_data['Date'])

nfp_combined_df = pd.merge(nfp_hypothesis, usdjpy_data, on='Date', how='outer')
adp_combined_df = pd.merge(adp_hypothesis, usdjpy_data, on='Date', how='outer')
usdjpy_data['Date'] = pd.to_datetime(usdjpy_data['Date'])
gold_data['Date'] = pd.to_datetime(gold_data['Date'])
oil_data['Date'] = pd.to_datetime(oil_data['Date'])
nfp_data['Date'] = pd.to_datetime(nfp_data['Date'])
adp_data['Date'] = pd.to_datetime(adp_data['Date'],errors='coerce')


mergedJPYData = pd.merge(usdjpy_data,nfp_data, on= 'Date', how= 'inner')
mergedgoldData = pd.merge(gold_data,nfp_data, on= 'Date', how= 'inner')
mergedoilData = pd.merge(oil_data,nfp_data, on= 'Date', how= 'inner')

rise = []
fall = []
grouped = mergedJPYData['status'].value_counts(normalize=True)
rise.append((grouped['Rise'] * 100))
fall.append((grouped['Fall'] * 100))
grouped = mergedgoldData['status'].value_counts(normalize=True)
rise.append((grouped['Rise'] * 100))
fall.append((grouped['Fall'] * 100))

grouped = mergedoilData['status'].value_counts(normalize=True)
rise.append((grouped['Rise'] * 100))
fall.append((grouped['Fall'] * 100))

assetList = ['USDJPY','XAGUSD','OILUSD']
print(rise)
# Bar graph
fig = go.Figure()

# Add bars for Rise probabilities
fig.add_trace(
    go.Bar(
        x= assetList,
        y= rise,
        name='Rise Probability (%)',
        marker_color='green'
    )
)

# Add bars for Fall probabilities
fig.add_trace(
    go.Bar(
        x= assetList,
        y= fall,
        name='Fall Probability (%)',
        marker_color='red'
    )
)

# Update layout
fig.update_layout(
    title="Forex Currency Rise/Fall Probabilities on NFP Release Days",
    xaxis_title="Forex Currency",
    yaxis_title="Probability (%)",
    barmode='group',
    legend_title="Status"
)

# Show the graph
fig.show()

mergedJPYData = pd.merge(usdjpy_data,adp_data, on= 'Date', how= 'inner')
mergedgoldData = pd.merge(gold_data,adp_data, on= 'Date', how= 'inner')
mergedoilData = pd.merge(oil_data,adp_data, on= 'Date', how= 'inner')

rise = []
fall = []
grouped = mergedJPYData['status'].value_counts(normalize=True)
rise.append((grouped['Rise'] * 100))
fall.append((grouped['Fall'] * 100))
grouped = mergedgoldData['status'].value_counts(normalize=True)
rise.append((grouped['Rise'] * 100))
fall.append((grouped['Fall'] * 100))

grouped = mergedoilData['status'].value_counts(normalize=True)
rise.append((grouped['Rise'] * 100))
fall.append((grouped['Fall'] * 100))

assetList = ['USDJPY','XAGUSD','OILUSD']
print(rise)
# Bar graph
fig2 = go.Figure()

# Add bars for Rise probabilities
fig2.add_trace(
    go.Bar(
        x= assetList,
        y= rise,
        name='Rise Probability (%)',
        marker_color='green'
    )
)

# Add bars for Fall probabilities
fig2.add_trace(
    go.Bar(
        x= assetList,
        y= fall,
        name='Fall Probability (%)',
        marker_color='red'
    )
)

# Update layout
fig2.update_layout(
    title="Forex Currency Rise/Fall Probabilities on ADP Release Days",
    xaxis_title="Forex Currency",
    yaxis_title="Probability (%)",
    barmode='group',
    legend_title="Status"
)
fig2.show()
# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Economic Indicators and Forex Visualization"),
    
    # Bar graph for economic indicators
    dcc.Graph(id='economic-bar', style={'margin-bottom': '50px'}),
    dcc.Graph(id='economic-bar2', style={'margin-bottom': '50px'}),
    dcc.Graph(id='adp_economic-bar', style={'margin-bottom': '50px'}),
    dcc.Graph(id='adp_economic-bar2', style={'margin-bottom': '50px'}),
    
    # Line graph for Forex currency
    dcc.Graph(id='forex-line'),
    dcc.Graph(id='Probability of profit'),
    dcc.Graph(id='Probability of adp')
])

# Callback to update graphs
@app.callback(
    [Output('economic-bar', 'figure'),Output('economic-bar2', 'figure'),Output('adp_economic-bar', 'figure'),Output('adp_economic-bar2', 'figure'), Output('forex-line', 'figure'),Output('Probability of profit', 'figure'),Output('Probability of adp', 'figure')],
    [Input('economic-bar', 'id')]  # Dummy input to initialize
)
def update_graphs(dummy_input):
    # Bar graph for economic indicators
    bar_fig = go.Figure(
        data=[
            go.Bar(
                x=nfp_combined_df['Date'],
                y=nfp_combined_df['Hypothesis_1_Status'],
                marker_color=nfp_combined_df['Hypothesis_1_Status'].apply(lambda x: 'green' if x == 1 else 'red' if x == -1 else 'grey')
            )
        ]
    )
    bar_fig.update_layout(
        title="NFP Hypothesis1",
        xaxis_title="Date",
        yaxis_title="Hypothesis Result Value Actual - Forecast",
        yaxis=dict(tickvals=[-1, 0, 1], ticktext=['Negative', 'Neutral', 'Positive'])
    )
    bar_fig2 = go.Figure(
        data=[
            go.Bar(
                x=nfp_combined_df['Date'],
                y=nfp_combined_df['Hypothesis_2_Status'],
                marker_color=nfp_combined_df['Hypothesis_2_Status'].apply(lambda x: 'green' if x == 1 else 'red' if x == -1 else 'grey')
            )
        ]
    )
    bar_fig2.update_layout(
        title="NFP Hypothesis2",
        xaxis_title="Date",
        yaxis_title="Hypothesis Result Value Actual - Previous",
        yaxis=dict(tickvals=[-1, 0, 1], ticktext=['Negative', 'Neutral', 'Positive'])
    )
    bar_adp_fig = go.Figure(
      data=[
          go.Bar(
              x=adp_combined_df['Date'],
              y=adp_combined_df['Hypothesis_1_Status'],
              marker_color=adp_combined_df['Hypothesis_1_Status'].apply(lambda x: 'green' if x == 1 else 'red' if x == -1 else 'grey')
          )
      ].
  )
    bar_adp_fig.update_layout(
      title="ADP Hypothesis1",
      xaxis_title="Date",
      yaxis_title="Hypothesis Result Value Actual - Forecast",
      yaxis=dict(tickvals=[-1, 0, 1], ticktext=['Negative', 'Neutral', 'Positive'])
  )
    print(adp_combined_df['Hypothesis_2_Status'])
    bar_adp_fig2 = go.Figure(
      data=[
          go.Bar(
              x=adp_combined_df['Date'],
              y=adp_combined_df['Hypothesis_2_Status'],
              marker_color=adp_combined_df['Hypothesis_2_Status'].apply(lambda x: 'green' if x == 1 else 'red' if x == -1 else 'grey')
          )
      ]
  )
    bar_adp_fig2.update_layout(
      title="ADP Hypothesis2",
      xaxis_title="Date",
      yaxis_title="Hypothesis Result Value Actual - Previous",
      yaxis=dict(tickvals=[-1, 0, 1], ticktext=['Negative', 'Neutral', 'Positive'])
  )
    # Line graph for Forex currency
    line_fig = go.Figure(
        data=[
            go.Scatter(
                x=nfp_combined_df['Date'],
                y=nfp_combined_df['Close'],
                mode='lines+markers',
                name='(USD/JPY)',
                line=dict(color='blue')
            )
        ]
    )
    line_fig.update_layout(
        title="Forex Currency Movement (USD/JPY)",
        xaxis_title="Time",
        yaxis_title="Close Price"
    )

    return bar_fig,bar_fig2,bar_adp_fig,bar_adp_fig2,line_fig,fig ,fig2

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)