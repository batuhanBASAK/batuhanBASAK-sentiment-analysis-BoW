
from BoW import BoW
from preprocessing import preprocess_text
import numpy as np

print('\nLoading the model...')
model = BoW()
model.load_from_file('the_model.pkl')
print('\nThe model has been completed successfully!')




from tkinter import *
from tkinter import ttk

root = Tk()
root.geometry('300x300')

label = ttk.Label(root, text="Write a text")
label.pack()

entry = ttk.Entry(root)
entry.pack()


# Create a label for displaying the sentiment result (initially empty)
sentiment_label = ttk.Label(root, text="")
sentiment_label.pack()


def get_sentiment():
    # Get the text entered in the entry widget
    text = entry.get()
    
    # Optionally print the text in the console (for debugging purposes)
    print(f"Entered text: {text}")

    # Preprocess the text
    text_preprocessed = preprocess_text(text)
    print('\nText preprocessed:')
    print(text_preprocessed)

    # Prepare the input for the model
    X = np.asarray([text_preprocessed])

    # Predict the sentiment
    y = model.predict(X)

    sentiment = y[0]

    sentiment_label.config(text=f"Sentiment: {sentiment}")

    # Print the prediction result (example sentiment)
    print('\nSentiment prediction:', sentiment)

    

button = ttk.Button(root, text='Get the sentiment!', command=get_sentiment)
button.pack()

root.mainloop()


