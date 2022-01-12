import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import plotly.express as px
from datetime import datetime
#from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotionclf_table,clear_tracking_data
##loading the prediction model
pipe_lr = joblib.load(open("models/demo.pkl","rb"))

emoji = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}
def predict_emoticon(text):
    results = pipe_lr.predict([text])
    return results[0]

def get_prediction_probability(text):
    results = pipe_lr.predict_proba([text])
    return results

def main():
    st.title("Emoticon Classification Demo")
    menu_items = ["Home","About"]

    choice = st.sidebar.selectbox("Menu",menu_items)
    #create_page_visited_table()
    #create_emotionclf_table()

    if choice == "Home":
        #add_page_visited_details("Home",datetime.now())
        st.subheader("Home Emoticon in Text")

        with st.form(key = 'emotion_form'):
            raw_text = st.text_area("Type your sentence here")
            submit_text = st.form_submit_button(label = 'Submit')

        if submit_text:
            col1,col2 = st.columns(2) ##when submitted 2 columns shown Original and Results

            prediction = predict_emoticon(raw_text)
            probability = get_prediction_probability(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                emoji_icon = emoji[prediction]
                #emoji_icon = "ok"
                st.write("{}:{}".format(prediction,emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                #st.write(probability)
                proba_df = pd.DataFrame(probability,columns = pipe_lr.classes_)
                #st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emoticons","probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x="emoticons",y="probability",color="emoticons")
                st.altair_chart(fig,use_container_width = True )

    elif choice == "About":
        st.subheader("About")
        #add_page_visited_details("About",datetime.now())
        libs = {"Libraries":["streamlit","altair","plotly","pandas","numpy","joblib"]}
        lib_table = pd.DataFrame(data = libs)
        st.write(lib_table)
        link = '[Github](http://github.com)'
        st.markdown(link,unsafe_allow_html = True)







if __name__ == '__main__':
    main()
