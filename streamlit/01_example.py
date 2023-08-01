import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

st.markdown("Cheat sheet: https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/app.py")
st.markdown("API docs: https://docs.streamlit.io/library/api-reference")

st.text("TEST TEXT")
st.write(1234)
df = pd.DataFrame(
    {'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40],}
    )

st.text("Using Magic")
df

st.text("Using st.write")
st.write(df)


st.write('Below is a DataFrame:', df, 'Above is a dataframe.')


st.text("LINE PLOT USING")
from bokeh.plotting import figure

x = [1, 2, 3, 4, 5]
y = [2, 1, 4, 1, 6]

fig = figure(title="Test Plot",
            x_axis_label="x",
            y_axis_label="y")

fig.line(x, y, legend_label="Line", line_width=2)
st.bokeh_chart(fig, use_container_width=True)

st.text("LINE CHART USING ST")
chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

st.text("MAP USING ST")
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data


option = st.selectbox(
    'Which number do you like best?',
     ["mobilenet", "resnet50", ""])