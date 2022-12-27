from fileinput import filename
from multiprocessing.resource_sharer import stop
from multiprocessing.util import Finalize
from tkinter.tix import TEXT
from typing import Text
from PySimpleGUI import Image
import PySimpleGUI as sg
import cv2
import face_recognition
import pickle
import numpy as np
import glob
import time
import csv
import pandas as pd
from sys import argv
import json



i = 0
array = []

def janela_principal():
    sg.theme('LightGrey')
    layout = [
        #[sg.Image(filename='logo.png')]
        [sg.Button('Cadastrar Usuário',size=(21, 3)), sg.Button('Reconhecer Rostos',size=(21, 3))],
    ]
    return sg.Window('MENU RECONHECIMENTO FACIAL', layout, finalize=True, icon='/home/julio/Área de Trabalho/Telas P.I/logoo.png')
def janela_cadastro():
    sg.theme('LightGrey')
    layout = [
        [sg.Text('Usuário'), sg.Input(key='-NAME-',size=(20,1))],
        [sg.Text('ID'), sg.Input(key='-ID-',size=(20,1))],
        [sg.Text('Acesso'), sg.Combo(['1', '2', '3', '4'], key='-ACESSO-')],
        [sg.Button('CADASTRAR')]
    ]
    return sg.Window('CADASTRO', layout,  finalize=True)

def responde_id(idk):
    idk = idk
    
    lista = {
        'id': idk
    }
    
    json_object = json.dumps(lista,indent=3)
    with open("ultimoid.json","w") as outfile:
        outfile.write(json_object)

def populalista(li):
    
    li = li      
          
    json_object = json.dumps(li,indent=3)
    with open("Lista.json","w") as outfile:
        outfile.write(json_object)
        
    
    
        
def escreve_lista(idk,nome,aces):
    idk = idk
    nome = nome
    acesso = aces
    
    lista = {
        'id': idk,
        'Nome': nome,
        'Nivel': acesso    
        } 

   # json_object = json.dumps(lista,indent=3)
    #with open("Lista.json","w") as outfile:
   #     outfile.write(json_object)
    
    return lista


def atualiza_lista():

    # Load the pickle format file
    input_file = open("ref_name.pkl", 'rb')
    new_dict = pickle.load(input_file)

    # Create a Pandas DataFrame
    df = pd.DataFrame.from_dict(new_dict, orient='index')

    # Copy DataFrame index as a column
    df['index1'] = df.index

    # Move the new index column to the front of the DataFrame
    index1 = df['index1']
    df.drop(labels=['index1'], axis=1, inplace=True)
    df.insert(0, 'index1', index1)

    # Convert to json values
    json_df = df.to_json(orient='values', date_format='iso', date_unit='s')

    # Create and record the JSON data in a new .JSON file
    with open('data.json', 'w') as js_file:
        js_file.write(json_df)


def reconhecer():

    f=open("ref_name.pkl","rb")
    ref_dictt=pickle.load(f)         #ref_dict=ref vs name
    f.close()

    f=open("ref_embed.pkl","rb")
    embed_dictt=pickle.load(f)      #embed_dict- ref  vs embedding 
    f.close()

    ############################################################################  encodings and ref_ids 
    known_face_encodings = []  #encodingd of faces
    known_face_names = []	   #ref_id of faces
    known_face_nivel = []



    for ref_id , embed_list in embed_dictt.items():
        for embed in embed_list:
            #print(embed)
            known_face_encodings +=[embed]
            known_face_names += [ref_id]

            
                                                
    #############################################################frame capturing from camera and face recognition
    video_capture = cv2.VideoCapture(2)
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True  :
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown" 

                # # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    unk = True
                else:
                    unk = False 
                    print("DESCONHECIDO")
                    responde_id(0)

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    responde_id(name)
                #else: name = 'Unknown'
                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

                        #updating in database

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            if unk:
                cv2.putText(frame, ref_dictt[name], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                font = cv2.FONT_HERSHEY_DUPLEX
            else:
                cv2.putText(frame, "Desconhecido", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                font = cv2.FONT_HERSHEY_DUPLEX
            
        #cv2.putText(frame, last_rec[0], (6,20), font, 1.0, (0,0 ,0), 1)

        # Display the resulting imagecv2.putText(frame, ref_dictt[name], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # t.cancel()
            break

            # break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

def cadastrar_rosto(idk,nome,aces):
    
    name = nome
    
    ref_id = idk
    
    acesso = aces

    try:
        f=open("ref_name.pkl","rb")

        ref_dictt=pickle.load(f)
        f.close()
    except:
        ref_dictt={}
    ref_dictt[ref_id]=name


    f=open("ref_name.pkl","wb")
    pickle.dump(ref_dictt,f)
    f.close()

    try:
        f=open("ref_embed.pkl","rb")

        embed_dictt=pickle.load(f)
        f.close()
    except:
        embed_dictt={}





    for i in range(5):
        key = cv2. waitKey(1)
        webcam = cv2.VideoCapture(2)
        while True:
            
            check, frame = webcam.read()
            # print(check) #prints true as long as the webcam is running
            # print(frame) #prints matrix values of each framecd 
            cv2.imshow("Capturando rostos", frame)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            key = cv2.waitKey(1)

            if key == ord('s') : 
                face_locations = face_recognition.face_locations(rgb_small_frame)
                if face_locations != []:

                    # filename="photo.jpg"
                    # cv2.imwrite(filename=filename, img=frame)
                    # image = face_recognition.load_image_file(filename)
                    # image = Image.fromarray(frame)
                    # image = image.convert('RGB')
                    face_encoding = face_recognition.face_encodings(frame)[0]
                    if ref_id in embed_dictt:
                        embed_dictt[ref_id]+=[face_encoding]
                    else:
                        embed_dictt[ref_id]=[face_encoding]
                    webcam.release()
                    # img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
                    # img_new = cv2.imshow("Captured Image", img_new)
                    cv2.waitKey(1)
                    cv2.destroyAllWindows()     
                    break
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Programa encerrado.")
                cv2.destroyAllWindows()
                break
    f=open("ref_embed.pkl","wb")
    pickle.dump(embed_dictt,f)
    f.close()            


janela1,janela2 = janela_principal(), None



while True:
    window,event,values = sg.read_all_windows()
    
    if window == janela1 and event == sg.WIN_CLOSED:
        break
    if window == janela1 and event == 'Cadastrar Usuário':
        janela2 = janela_cadastro()
    if window == janela2 and event == sg.WIN_CLOSED:
        janela2.hide()
    if window == janela2 and event == 'CADASTRAR':
        idk = int(values['-ID-'])
        nome = str(values['-NAME-'])
        acesso = int(values['-ACESSO-'])
        print(idk)
        print(nome)
        print(acesso)
        cadastrar_rosto(idk,nome,acesso)
        
        for numero in range(1):
            array.append(escreve_lista(idk,nome,acesso))
            i+=1
            print(i)
        
        print(array)
        populalista(array)

        janela2.hide()
        
    if window == janela1 and event == 'Reconhecer Rostos':
        reconhecer()
        

