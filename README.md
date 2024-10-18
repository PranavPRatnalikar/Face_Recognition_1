#Create Virtual Enviornment: 
python -m venv venv

#start venv
venv/scripts/activate

#install dependencies
pip install -r requirements.txt

#run streamlit
streamlit run app.py
