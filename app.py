from flask import Flask, redirect, render_template, request, url_for
from flask import jsonify
import pickle
import mysql.connector as mysql
import pandas as pd
app = Flask(__name__)
#app.config["UPLOAD_FOLDER"]="./uploads" 
app.secret_key="djjdjdjddj"
app.config["MAX_CONTENT_LENGTH"]=10*1024*1024

con=None
cur=None
def connect():
    global con
    global cur
    con=mysql.connect(host="localhost",user="root",password="",database="laptop")
    cur=con.cursor(dictionary=True)
def closeDB():
    global con
    con.close()

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/login')
def login():
    return render_template('login.html')

@app.route("/api/login",methods=["GET"])
def api_login():
    global con
    global cur
    username=request.args.get("uname")
    password=request.args.get("password")
    print(f"{username} and {password}")
    connect()
    cur.execute("select * from users where username=%s and password=%s",(username,password))
    data=cur.fetchone()
    if(data!=None):
        print(data)        
        return jsonify({"status":"success","data":data})
    else:
        return jsonify({"status":"error"})


@app.route('/register')
def register():
    return render_template('register.html')

@app.route("/api/customer",methods=["GET","POST"],defaults={'id':0})
@app.route("/api/customer/<int:id>",methods=["GET","PUT","DELETE"])
def customer(id=None):
    global con
    global cur
    connect()
    print(request.method)
    if(request.method=="GET"):
        if(id==0):
            cur.execute("select * from customer")
            data=cur.fetchall()
            if(len(data)>0):
                return jsonify(data) 
            else:
                return jsonify({"data":""})
        else:
            cur.execute("select * from customer where cid=%s",(id,))
            data=cur.fetchone()
            return jsonify(data)
    elif(request.method=="POST"):
        name=request.form["name"]
        username=request.form["uname"]
        password=request.form["password"]
        address=request.form["address"]
        mobileno=int(request.form["mobileno"])
        gender=request.form["gender"]
        utype='customer'
        cur.execute("insert into customer(name,address,gender,mobileno) values(%s,%s,%s,%s)",(name,address,gender,mobileno))
        cur.execute("insert into users(username,password,type) values(%s,%s,%s)",(username,password,utype))
        con.commit()
        if(cur.rowcount>0):
            lasrowid=cur.lastrowid
            data={"status":"success","rows":lasrowid}
            return jsonify(data)
        else:
            data={"status":"error","rows":0}
            return jsonify(data)

@app.route('/ahome')
def ahome():
    return render_template('ahome.html')
@app.route('/ahome_display')
def ahome1():
    connect()
    cur.execute("SELECT * FROM customer")
    data = cur.fetchall()
    closeDB()
    return jsonify(data)  # Replace with HTML render if needed



@app.route('/chome')
def chome():
    return render_template('chome.html')

with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']
    label_encoders = data['label_encoders']

def encode_label(label, encoder):
    """Encodes a label with a fallback for unseen labels."""
    try:
        return encoder.transform([label])[0]
    except ValueError:
        # Fallback: Assign an existing label with similar characteristics or the most frequent label
        print(f"Warning: Unseen label '{label}' for column. Assigning fallback value.")
        return encoder.transform([encoder.classes_[0]])[0] 

@app.route('/index')
def index():
     # Prepare the dropdown options
    data_dict = {
        'company': ['Apple', 'Dell', 'HP', 'Lenovo', 'Acer', 'Asus'],  # Example values
        'typename': ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible'],
        'ram': [4, 8, 16, 32],
        'weight': [1.0, 4.0],
        'os': ['Windows', 'macOS', 'Linux', 'No OS'],
        'touchscreen': [0, 1],
        'isips': [0, 1],
        'cpubrand': ['Intel Core i7', 'AMD'],
        'ssd': [128, 256, 512, 1024],
        'hdd': [0, 500, 1000],
        'Gpu_brand': ['Intel', 'Nvidia', 'AMD'],
        'resolution': ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'],
        'prediction': False
    }
    return render_template('index.html', data_dict=data_dict)
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    brand = request.form['brand']
    typename = request.form['typename']
    ram = int(request.form['ram'])
    weight = float(request.form['weight'])
    touchscreen = int(request.form['touchscreen'])
    ips = int(request.form['IPS'])
    screen_size = float(request.form['screen_size'])
    resolution = request.form['resolution']
    cpu = request.form['cpu']
    ssd = int(request.form['ssd'])
    hdd = int(request.form['hdd'])
    gpu = request.form['gpu']
    os = request.form['os']

    # Process resolution to get PPI
    X_resolution = int(resolution.split('x')[0])
    y_resolution = int(resolution.split('x')[1])
    ppi = ((X_resolution**2) + (y_resolution**2))**0.5 / screen_size

    # Construct Memory column
    memory = f"{ssd}GB SSD {hdd}GB HDD"

    # Sample data for prediction with unseen labels handling
    sample_data = pd.DataFrame({
    'Company': [encode_label(brand, label_encoders['Company'])],
    'TypeName': [encode_label(typename, label_encoders['TypeName'])],
    'ScreenResolution': [encode_label(resolution, label_encoders['ScreenResolution'])],
    'Cpu': [encode_label(cpu, label_encoders['Cpu'])],
    'Memory': [encode_label('8GB SSD', label_encoders['Memory'])],
    'Gpu': [encode_label(gpu, label_encoders['Gpu'])],
    'OpSys': [encode_label('Windows 10', label_encoders['OpSys'])],
    'Inches': [screen_size],
    'Ram': [ram],
    'Weight': [weight]})


    # Ensure the columns are in the same order as during training
    feature_order = ['Company', 'TypeName','Inches' ,'ScreenResolution', 'Cpu','Ram', 'Memory', 'Gpu', 'OpSys',  'Weight']
    sample_data = sample_data[feature_order]

    # Scale the numerical features
    sample_data[['Inches', 'Ram', 'Weight']] = scaler.transform(sample_data[['Inches', 'Ram', 'Weight']])

    # Predict the price using the model
    predicted_price = model.predict(sample_data)
    price=predicted_price[0]
    price = int(round(price))  

    return render_template('predict.html', price=price)


@app.route('/contact', methods=['GET'])
def contact_form():
    return render_template('contact.html')  # assuming you saved your form in contact.html

# Route to handle form submission
@app.route('/contact', methods=['POST'])
def handle_contact():
    global con
    global cur
    connect()
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']
    # Insert into MySQL
    cur.execute("INSERT INTO contact(name, email, message) VALUES (%s, %s, %s)",(name,email,message))
    con.commit()
    return "Message submitted successfully!"



# Route to show edit form with existing data
@app.route('/edit/<int:cid>')
def edit_contact(cid):
    connect()
    cur.execute("SELECT * FROM contact WHERE cid = %s", (cid,))
    contact = cur.fetchone()
    closeDB()
    return render_template('edit_form.html', contact=contact)

@app.route('/msg_list')
def msg_list():
    return render_template('msg_list.html')

# Route to update reply
@app.route('/update_reply/<int:cid>', methods=['POST'])
def update_reply(cid):
    reply = request.form['reply']
    connect()
    cur.execute("UPDATE contact SET reply = %s WHERE cid = %s", (reply, cid))
    con.commit()
    closeDB()
    return redirect(url_for('msg_list'))  # You can change this route

@app.route('/contactpageedit')
def contactpageedit():
    return render_template('contactpageedit.html')
# Route to list all contacts (for viewing after update)
@app.route('/contacts_reply')
def contacts():
    connect()
    cur.execute("SELECT * FROM contact")
    data = cur.fetchall()
    closeDB()
    return jsonify(data)  # Replace with HTML render if needed




@app.route('/edit_form')
def edit_form():
    cid = request.args.get('cid')  # Get cid from URL query parameter
    connect()
    cur.execute("SELECT * FROM contact WHERE cid = %s", (cid,))
    contact = cur.fetchone()
    closeDB()
    return render_template('edit_form.html', contact=contact)

@app.route('/messagelist')
def display_reply():
    return render_template('display.html')

if __name__ == '__main__':
    app.run(debug=True)
