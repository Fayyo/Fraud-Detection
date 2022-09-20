
import streamlit as st
import pandas as pd
import pickle

header = st.container()
dataset = st.container()
visualisations = st.container()
model_training = st.container()

@st.cache
def get_data(filename):
    data = pd.read_csv(filename)

    return data

with header:
    st.title(' Online Payment Fraud Prediction')
    st.text('In this project, I looked into the trends of fraudulent online transactions')


with dataset:
    st.header('Online Payments Dataset')
    st.text('I got this dataset from Kaggle.com. It contains over three hundred thousand samples and 11 features')

    online_payments = get_data('online_payments.csv')
    st.write(online_payments.head())


with visualisations:
    st.header('Exploring the dataset to show visuals')
    st.subheader('Distribution of online transactions')
    payment_type = pd.DataFrame(online_payments['type'].value_counts())
    st.bar_chart(payment_type)


with model_training:
    st.header('Training the model')
    st.text('I used a decision tree classifier to train this model')

    st.text('Here is the list of transaction types and their corresponding numbers')
    st.markdown('CASH_OUT: 1')
    st.markdown('PAYMENT: 2')
    st.markdown('CASH_IN: 3')
    st.markdown('TRANSFER: 4')
    st.markdown('DEBIT: 5')

    input_feature_1 = st.slider('What transaction are you performing?', min_value=1, max_value=5, step=1, value=1)
    input_feature_2 = st.number_input('What amount would you like to input', min_value=0, step=1000)
    input_feature_3 = st.number_input('What is your current account balance', min_value=0, step=1000)
    input_feature_4 = st.number_input('What will your account balance be after the transaction', min_value=0, step=1000)


    loaded_model = pickle.load(open('trained_lr_model.csv', 'rb'))


    submit = st.button('Predict')
    if submit:
        prediction = loaded_model.predict([[input_feature_1, input_feature_2, input_feature_3, input_feature_4]])
        if prediction == 'Fraud':
            st.write('This is a Fraud transaction')
        else:
            st.write('This is not a Fraud transaction')
