import os
import os.path
import face_recognition
import pickle

class face_reco:
    def __init__(self):
        self.knownFaces=[]
        self.knownNames=[]
        self.blackListNames=[]
        self.folderpath=''

    def init_with_images(self,folderpath):
        self.folderpath=folderpath
        if(os.path.isfile(folderpath+"/bak_face.json") and os.path.isfile(folderpath+"/bak_name.json") and os.path.isfile(folderpath+"/bak_blname.json")):
                print("predifined model found, skipping loading images...")
                facef = open(folderpath+"/bak_face.json","rb")
                namef = open(folderpath+"/bak_name.json","rb")
                blnamef = open(folderpath+"/bak_blname.json","rb")
                self.knownFaces = pickle.loads(facef.read())
                self.knownNames = pickle.loads(namef.read())
                self.blackListNames = pickle.loads(blnamef.read())

                print(len(self.knownNames))
                print(len(self.blackListNames))
                return
        for parent,dirnames,filenames in os.walk(folderpath+"/whiteList"):          
            print("Loading known images")
            for filename in filenames:
                file_path = os.path.join(parent,filename)
                if(file_path.endswith(".jpg")or file_path.endswith(".png")):
                    img = face_recognition.load_image_file(file_path)
                    encoding = face_recognition.face_encodings(img)[0]
                    name = file_path.split('/')[-1][0:-4]
                    print (name)
                    
                    self.knownFaces.append(encoding)
                    self.knownNames.append(name)
        for parent,dirnames,filenames in os.walk(folderpath+"/blackList"):          
            print("Loading known images")
            for filename in filenames:
                file_path = os.path.join(parent,filename)
                if(file_path.endswith(".jpg")or file_path.endswith(".png")):
                    img = face_recognition.load_image_file(file_path)
                    encoding = face_recognition.face_encodings(img)[0]
                    name = file_path.split('/')[-1][0:-4]
                    print (name)
                    
                    self.knownFaces.append(encoding)
                    self.knownNames.append(name)
                    self.blackListNames.append(name)
        self.Save()
        print("Ready to go!")
    
    def Save(self):
        facef = open(self.folderpath+"/bak_face.json","wb")
        namef = open(self.folderpath+"/bak_name.json","wb")
        blnamef = open(self.folderpath+"/bak_blname.json","wb")
        facef.write(pickle.dumps(self.knownFaces))
        namef.write(pickle.dumps(self.knownNames))
        blnamef.write(pickle.dumps(self.blackListNames))
    def process_one_pic(self,frame):
        isBlackList=False
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
                if name in self.blackListNames:
                    isBlackList=True
            face_names.append(name)
        return face_locations,face_names,isBlackList

    def reset(self):
        self.knownFaces=[]
        self.knownNames=[] 
        self.blackListNames=[]     
