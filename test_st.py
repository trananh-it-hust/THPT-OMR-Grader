import streamlit as st
import concurrent.futures
def worker(x): return x*x
if st.button('run'):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f = executor.submit(worker, 5)
        st.write(f.result())
