import streamlit as st
import pickle as pk
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('stopwords')
stop_list = stopwords.words('english')

ps = PorterStemmer()

model = pk.load(open("reviewsentiment.pkl",'rb'))
cv = pk.load(open('vectorizer.pkl', 'rb'))

def clean_text(text):
  text = text.lower()
  text = re.sub(r'<*.?>','', text)
  text = re.sub(r"[^a-zA-Z0-9]+", " ",text)
  text = ' '.join([i for i in text.split() if i not in stop_list])
  data = text.split()
  stemt = list(map(ps.stem,data))
  text = ' '.join(stemt)
  return text

#GUI
st.title('Review Sentiment Analysis')
st.markdown('''Enter a movie/anime/book review and the AI will predict whether it is a positive or a negitive review.
             It works on the [Bernoulli Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Bernoulli_naive_Bayes) algorithm.''')

review = st.text_area('Review',placeholder='Write a review or copy and paste a review here to analyse', height=250)

mrev = [clean_text(review.strip())]
feed = cv.transform(mrev)

def clean_text2(text):
  text = text.lower()
  text = re.sub(r"[^a-zA-Z0-9]+", " ",text)
  return text

rev_text = clean_text2(review)
def word_character_count_frequency(text):
    words = text.split()
    char_count_freq = {}

    for word in words:
        char_count = len(word)
        if char_count in char_count_freq:
            char_count_freq[char_count] += 1
        else:
            char_count_freq[char_count] = 1

    return char_count_freq

freq_dict = word_character_count_frequency(rev_text)
data = {
    'chars':list(freq_dict.keys()),
    'word count':list(freq_dict.values())
}
df = pd.DataFrame(data)




if st.button('Predict'):
    if review.strip()!='':
        
        
        pred = model.predict(feed)
        if pred[0] == 1:
            st.success('It is most likely a Positive review')
            st.balloons()
        else:
            st.error('It is most likely a Negitive review')
        st.write('---')
        st.bar_chart(df,x='chars', y='word count', use_container_width=True)

        st.write('Word frequency cloud')
        rev_text = clean_text2(review)
        wordcloud = WordCloud(background_color="white").generate(rev_text)

        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.show()
        st.pyplot()
    else:
        st.write('Write a review first to analyse')


