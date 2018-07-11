from flask import Flask,request,make_response,Response
import face_recognition
from flask_cors import CORS
import json
import os
app=Flask(__name__)
CORS(app)
@app.route('/getFeature',methods=['POST'])
def getFeature():
    name = request.args['name']
    path = request.args['filePath']
    print(request.args)
    print(name,path)
    if(not os.path.isfile(path)):
        return Response('File Not Found')
    img = face_recognition.load_image_file(path)
    allencodings = face_recognition.face_encodings(img)
    if(len(allencodings)!=1):
        return Response('Invalid Image')
    
    encoding = allencodings[0]
    data = {}
    data['name']=name
    data['face-data']=encoding.tolist()

    str_content = json.dumps(data)
    return Response(str_content)



if __name__=='__main__':
    app.run(host='0.0.0.0',debug = True)