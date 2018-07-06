import os
import os.path
import face_recognition
import pickle

class face_reco:
    def __init__(self):
        self.knownFaces=[]
        self.knownNames=[]

    def init_with_images(self,folderpath):
        if(os.path.isfile(folderpath+"/bak_face.json") and os.path.isfile(folderpath+"/bak_name.json")):
                print("predifined model found, skipping loading images...")
                facef = open(folderpath+"/bak_face.json","rb")
                namef = open(folderpath+"/bak_name.json","rb")
                self.knownFaces = pickle.loads(facef.read())
                self.knownNames = pickle.loads(namef.read())
                print(len(self.knownNames))
                return
        for parent,dirnames,filenames in os.walk(folderpath):          
            print("Loading known images")
            for filename in filenames:
                file_path = os.path.join(parent,filename)
                if(file_path.endswith(".jpg")or file_path.endswith(".png")):
                    img = face_recognition.load_image_file(file_path)
                    encoding = face_recognition.face_encodings(img)[0]
                    name = file_path.split('\\')[-1][0:-4]
                    print (name)
                    
                    self.knownFaces.append(encoding)
                    self.knownNames.append(name)
        facef = open(folderpath+"/bak_face.json","wb")
        namef = open(folderpath+"/bak_name.json","wb")
        facef.write(pickle.dumps(self.knownFaces))
        namef.write(pickle.dumps(self.knownNames))
        print("Ready to go!")

    def process_one_pic(self,frame):
        isMatch=False
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.knownFaces, face_encoding,tolerance=0.4)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = self.knownNames[first_match_index]
                isMatch=True
            face_names.append(name)
        return face_locations,face_names,isMatch

    def reset(self):
        self.knownFaces=[]
        self.knownNames=[]      