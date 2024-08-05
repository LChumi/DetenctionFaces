#-----------------------Importamos las librerias ---------------
import cv2
import mediapipe as np

#---------------------Declaramos el detector-------------------------
detector = np.solutions.face_detection  #Detector
dibujo = np.solutions.drawing_utils  #Dibujo

#---------------------Realizar la videoCaptura-----------------------
cap = cv2.VideoCapture(0)

#---------------------Inicializacion de parametros-------------------
with detector.FaceDetection(min_detection_confidence=0.75) as rostros:

    while True:
        #Realizamos la lectura de la videocamara
        ret, frame = cap.read()

        #Eliminar el error de movimiento
        frame = cv2.flip(frame, 1)

        #Correcion de color
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Deteccion de rostros
        resultado = rostros.process(rgb)

        #Filtro de seguridad
        if resultado.detections is not None:
            for rostro in resultado.detections:
                dibujo.draw_detection(frame, rostro,dibujo.DrawingSpec(color=(0,255,0)))

                for id, coordenadas in enumerate(resultado.detections):
                    #Mostramos las coordenadas
                    #print("Cordenadas: ", coordenadas)

                    #Extraccion de dimensiones de nuestra imagen
                    al, an, c = frame.shape

                    #Extraer X inicial e Y inicial
                    x= coordenadas.location_data.relative_bounding_box.xmin
                    y= coordenadas.location_data.relative_bounding_box.ymin

                    #Extraer ancho y alto
                    ancho = coordenadas.location_data.relative_bounding_box.width
                    alto = coordenadas.location_data.relative_bounding_box.height

                    #Conversion a pixeles
                    xi,yi = int(x * an), int(y * al)
                    xf,yf = int(ancho * an), int(alto * al)

                    #Estraer el punto central de nuestro rostro
                    cx = (xi + (xi + xf)) // 2
                    cy = (yi + (yi + yf)) // 2

                    #Mostrar esta coordenada
                    cv2.circle(frame, (cx, cy), 5, (255,0,255), cv2.FILLED)


        #Mostramos los fotogramas
        cv2.imshow('Camera', frame)

        #Leemos el teclado metodo ASCI 27=ESC
        t = cv2.waitKey(1)
        if t == 27:
            break
cap.release()
cv2.destroyAllWindows()
