import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Simulation[tm]")
st.write("One demo simulation")

st.sidebar.markdown("## Line Plot Demo")
st.sidebar.markdown("**Hyperparameters**")

x = st.sidebar.slider("Slope", min_value = 0.01, max_value = 0.1, step = 0.01)
y = st.sidebar.slider("Noise", min_value = 0.01, max_value = 0.1, step = 0.01)


st.write(f"x={x},  y={y}")

values = np.cumprod(1+np.random.normal(x,y, (100, 10)), axis=0)

for i in range(values.shape[1]):
    plt.plot(values[:, i])

st.pyplot()

