from flask import Flask,render_template,session,url_for,redirect
import numpy as np 
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
import joblib 
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

			
def text_process(mess):
	
	# Check characters to see if they are in punctuation
	nopunc = [char for char in mess if char not in string.punctuation]

	# Join the characters again to form the string.
	nopunc = ''.join(nopunc)
		
	# Now just remove any stopwords
	return [stemmer.stem(word) for word in nopunc.split() if word.lower() not in stopwords.words('english')]




def return_prediction(model,mess):
    
    ans = model.predict([mess])[0]
    return ans


app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
# Load the model from the file 
model = joblib.load(open('spam_model.pkl','rb'))

class checkmessage(FlaskForm):

	mess = TextField("Message")
	submit = SubmitField("Analyze")



@app.route("/",methods=['GET','POST'])
def index():

	form = checkmessage()

	if form.validate_on_submit():

		session['mess'] = form.mess.data

		return redirect(url_for("prediction"))

	return render_template('home.html',form=form)
	



@app.route('/prediction')
def prediction():

	mess = session['mess']

	results = return_prediction(model,mess)

	return render_template('prediction.html',results=results)


if __name__ == "__main__":
    app.run(debug=True)


