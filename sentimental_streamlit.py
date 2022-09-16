import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

    
#tokenizer = AutoTokenizer.from_pretrained("C:\\Users\\Gagandeep Singh\\Documents\\amazon-review-sentiment-analysis\\")
#model = AutoModelForSequenceClassification.from_pretrained("C:\\Users\\Gagandeep Singh\\Documents\\amazon-review-sentiment-analysis\\")
def main():
    # title and description
    st.header("""
    # Sentiment Analysis
    """)
    st.text("")
    st.text("")
    st.text("Works with English, Dutch, German, French, Spanish and Italian")
    st.text("")
    # search bar
    query = st.text_input("Find Sentiment!", "")
    st.text("")

    st.write(f"{query}")


    if query != "":
        # encode the query as sentence vector

        tokenized_text = tokenizer(query, return_tensors='pt')
        output = model(**tokenized_text)


        outputs = output.logits.softmax(dim=-1).tolist()[0]
        outputs = np.argmax(outputs)+1
        outputs = "*"*outputs 

        st.text("")
        st.write("Sentiment score: "," ",f"{outputs}")

    if __name__=='__main__':
    main()
