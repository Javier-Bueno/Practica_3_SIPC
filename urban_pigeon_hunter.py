# Universidad de La Laguna
# Grado en Ingeniería Informática
# Sistemas de Interacción Persona-Computador
# Práctica 3: Interfaces gestuales
# Componentes del grupo:
# - Javier Bueno Calzadilla - alu0101627922
# - Adriel Reyes Suárez
# - Carlos Pérez Gómez
# Versión: 1.0.0
# Historial de modificaciones con versiones:
# Versión 0.0.1
# - Creación inicial del código
# Versión 0.0.2
# - Modificaciones: Aparición de cazador, proyectil, proyectil de la paloma


import sys, random
random.seed(1) # Hace que la simulación sea igual cada vez, más fácil de depurar
import pygame
import pymunk
import pymunk.pygame_util
import cv2  # Para captura de video desde la cámara
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp  # Para utilidades de dibujo
import numpy as np  # Para operaciones con arrays numéricos

# Tamaño de la pantalla
display_h = 800
display_w = 800

# Tamaños del cazador y el arma (en píxeles)
HUNTER_W = 20
HUNTER_H = 60
GUN_W = 10
GUN_H = 30

# Configuraciones de la simulación
FPS = 50
TICKS_MIN = 1 * FPS  # 50 ticks (1 segundo)
TICKS_MAX = 2 * FPS  # 100 ticks (2 segundos)

# Configuraciones de las imágenes
scale_width = 29 * 1.5
scale_height = 29 * 1.5

# Animaciones de las palomas:
FRAME_INTERVAL = FPS * 1.5 # Las palomas cambian de imagen cada 1.5 segundos

# ========== INICIALIZAR MEDIAPIPE TASKS PARA DETECCIÓN DE MANOS ==========
Base_options = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Crear opciones para HandLandmarker
options = HandLandmarkerOptions(
    base_options=Base_options(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

# Crear el detector de manos
landmarker = HandLandmarker.create_from_options(options)
mp_drawing = mp.solutions.drawing_utils  # Utilidades para dibujar los puntos de la mano

# Captura de video
cap = cv2.VideoCapture(0)  # Abrir cámara web (índice 0 = cámara por defecto)
hand_x = display_w // 2  # Posición inicial de la mano en el centro

# Función para detectar si la mano está abierta (compatible con mediapipe.solutions)
def is_hand_open(hand_landmarks):
    """Detecta si la mano está abierta calculando la distancia entre los dedos"""
    # Puntos de referencia clave (landmarks)
    wrist = hand_landmarks.landmark[0]  # Muñeca
    middle_finger_tip = hand_landmarks.landmark[12]  # Punta del dedo medio
    
    # Calcular distancia vertical entre muñeca y punta del dedo medio
    # Si la distancia es grande, la mano está abierta
    distance = middle_finger_tip.y - wrist.y
    
    # Si la distancia es negativa y el valor absoluto es mayor a 0.15, la mano está abierta
    return distance < -0.15

# Función para detectar si la mano está abierta (compatible con MediaPipe Tasks)
def is_hand_open_tasks(hand_landmarks):
    """Detecta si la mano está abierta usando landmarks de MediaPipe Tasks (lista de NormalizedLandmark)"""
    # En MediaPipe Tasks, hand_landmarks es una lista de NormalizedLandmark
    # Índice 0 = muñeca, Índice 12 = punta del dedo medio
    wrist = hand_landmarks[0]
    middle_finger_tip = hand_landmarks[12]
    
    # Calcular distancia vertical entre muñeca y punta del dedo medio
    distance = middle_finger_tip.y - wrist.y
    
    # Si la distancia es negativa y el valor absoluto es mayor a 0.15, la mano está abierta
    return distance < -0.15

def get_hand_position():
    """Detecta la posición de la mano abierta usando MediaPipe Tasks HandLandmarker"""
    global hand_x, landmarker
    import time
    
    # Capturar frame de la cámara
    ret, frame = cap.read()
    if not ret:
        return hand_x
    
    # Voltear la imagen horizontalmente para efecto espejo
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Convertir BGR a RGB (OpenCV usa BGR, MediaPipe usa RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convertir a formato de imagen de MediaPipe Tasks
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detectar manos con timestamp
    timestamp_ms = int(time.time() * 1000)
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    # Si se detecta una mano abierta, obtener su posición
    if results.hand_landmarks:
        hand_landmarks = results.hand_landmarks[0]  # Primera (única) mano
        
        # Verificar si la mano está abierta
        if is_hand_open_tasks(hand_landmarks):
            # Usar el punto de la muñeca/palma (punto 0) para controlar el juego
            palm_center = hand_landmarks[0]
            # Convertir coordenadas normalizadas a píxeles del juego
            hand_x = int(palm_center.x * display_w)
            # Limitar dentro de los bordes
            hand_x = max(30, min(display_w - 30, hand_x))
        
        # Dibujar puntos de la mano en el frame (opcional, para debugging)
        if results.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    # Mostrar frame con detección
    cv2.imshow('Hand Detection', frame)
    cv2.waitKey(1)
    
    return hand_x

# Fucniones para añadir los objetos al espacio de pymunk
def add_lr_pidgeon(space): # Paloma que aparece por la izquierda y desaparece por la derecha
    mass = 3
    # radius = 20
    width = scale_width
    height = scale_height
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC) # Creamos el cuerpo de la paloma cinemático
    x = random.choice([0, display_w])  # Seleccionamos el rango en x en que se va a crear (bordes de la pantalla)
    y = random.randint(375, display_h - 50)  # Seleccionamos el rango en y en que se va a crear
    body.position = x, y # Fijamos su posición
    if x == 0:
        body.velocity = 120, 0 # Asignamos una velocidad aleatoria en x
        direction = "RIGHT" # La paloma surge del origen (para diferenciar sprites)
    else:
        body.velocity = -120, 0 # Asignamos una velocidad aleatoria en x
        direction = "LEFT" # La paloma surge de la derecha (para diferenciar sprites)
    shape = pymunk.Poly.create_box(body, (width, height)) # Creamos una forma rectangular para que el cuerpo pueda colisionar
    shape.mass = mass # Asignamos la masa a la forma 
    shape.width = width
    shape.height = height
    shape.direction = direction
    shape.animation_frame_index = 0 # Índice inicial para el conjunto con las imágenes de la paloma
    shape.animation_timer  = FRAME_INTERVAL # Temporizador para saber cuando cambiar la imagen

    space.add(body, shape) # Añadimos el cuerpo con su forma al espacio

    return shape


def add_proyectile(space, position): # Proyectil de la paloma
    mass = 1
    radius = 5
    body = pymunk.Body(body_type=pymunk.Body.DYNAMIC) # Creamos el cuerpo del proyectil dinámico
    body.position = position # Fijamos su posición a la posición de la paloma
    shape = pymunk.Circle(body, radius)
    shape.mass = mass
    space.add(body, shape)
    return shape

def add_hunter(space): # Cazador de palomas
    mass = 10
    width = HUNTER_W
    height = HUNTER_H
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC) # Creamos el cuerpo del cazador cinemático
    body.position = display_w // 2, 50 # Fijamos su posición en el centro abajo de la pantalla
    shape = pymunk.Poly.create_box(body, (width, height)) # Creamos una forma rectangular para que el cuerpo pueda colisionar
    shape.mass = mass
    # Guardar dimensiones para dibujo (ancho y alto en píxeles del cuerpo)
    shape.width = width
    shape.height = height
    shape.draw_radius = width // 2  # atributo auxiliar (semántico)
    space.add(body, shape)
    return shape

def add_gun(space, hunter):
    mass = 2
    width = GUN_W
    height = GUN_H
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC) # Creamos el cuerpo del arma cinemático
    body.position = hunter.body.position.x, hunter.body.position.y + 40 # Fijamos su posición encima del cazador
    shape = pymunk.Poly.create_box(body, (width, height)) # Creamos una forma rectangular para que el cuerpo pueda colisionar
    shape.mass = mass
    # Guardar dimensiones para dibujo
    shape.width = width
    shape.height = height
    shape.draw_radius = width // 2
    space.add(body, shape)
    return shape


# Funciones para dibujar los objetos en pantalla (HAY QUE REVISARLAS)
def draw_pidgeon(screen, pidgeon):
    p = int(pidgeon.body.position.x), display_h - int(pidgeon.body.position.y)
    #pygame.draw.circle(screen, (0,0,255), p, int(pidgeon.radius), 2)
    pygame.draw.rect(screen, (255,0,0), (p[0] - pidgeon.width / 2, p[1] - pidgeon.height / 2, pidgeon.width, pidgeon.height), 2)

def draw_pidgeon_with_image(screen,pidgeon,lr_frame, rl_frame):
    current_index = pidgeon.animation_frame_index
    if pidgeon.direction == "LEFT":
        image_to_draw = rl_frame[current_index]
    else:
        image_to_draw = lr_frame[current_index]
        
    p = int(pidgeon.body.position.x - pidgeon.width / 2), (display_h - int(pidgeon.body.position.y + (pidgeon.height / 2)))
    screen.blit(image_to_draw, p)

def draw_proyectile(screen, proyectile):
    p = int(proyectile.body.position.x), display_h - int(proyectile.body.position.y)
    pygame.draw.circle(screen, (255,0,0), p, int(proyectile.radius), 2)

def draw_proyectile_with_image(screen,proyectile,image):
    p = int(proyectile.body.position.x - proyectile.radius), (display_h - int(proyectile.body.position.y))- proyectile.radius
    screen.blit(image, p)

def draw_hunter(screen, hunter):
    # Dibujar rectángulo del cazador usando sus dimensiones guardadas
    cx = int(hunter.body.position.x)
    cy = display_h - int(hunter.body.position.y)
    w = int(getattr(hunter, 'width', 20))
    h = int(getattr(hunter, 'height', 60))
    rect = (cx - w // 2, cy - h // 2, w, h)
    pygame.draw.rect(screen, (0,255,0), rect, 2)

def draw_hunter_with_image(screen,hunter,image):
    # Dibujar rectángulo del cazador y superponer la imagen centrada
    cx = int(hunter.body.position.x)
    cy = display_h - int(hunter.body.position.y)
    w = int(getattr(hunter, 'width', 20))
    h = int(getattr(hunter, 'height', 60))
    rect = (cx - w // 2, cy - h // 2, w, h)
    # Dibujar rectángulo (borde)
    pygame.draw.rect(screen, (0,255,0), rect, 2)
    # Superponer imagen centrada en el rectángulo
    if image is not None:
        ix, iy = image.get_width(), image.get_height()
        img_pos = (cx - ix // 2, cy - iy // 2)
        screen.blit(image, img_pos)

def draw_gun(screen, gun):
    # Dibujar rectángulo del arma usando sus dimensiones guardadas
    cx = int(gun.body.position.x)
    cy = display_h - int(gun.body.position.y)
    w = int(getattr(gun, 'width', 10))
    h = int(getattr(gun, 'height', 30))
    rect = (cx - w // 2, cy - h // 2, w, h)
    pygame.draw.rect(screen, (255,255,0), rect, 2)

def draw_gun_with_image(screen,gun,image):
    # Dibujar rectángulo del arma y superponer la imagen centrada
    cx = int(gun.body.position.x)
    cy = display_h - int(gun.body.position.y)
    w = int(getattr(gun, 'width', 10))
    h = int(getattr(gun, 'height', 30))
    rect = (cx - w // 2, cy - h // 2, w, h)
    pygame.draw.rect(screen, (255,255,0), rect, 2)
    if image is not None:
        ix, iy = image.get_width(), image.get_height()
        img_pos = (cx - ix // 2, cy - iy // 2)
        screen.blit(image, img_pos)

def update_pidgeon_animation(pidgeon):
    pidgeon.animation_timer -= 1
    if pidgeon.animation_timer <= 0:
        # reiniciamos el temporizador
        pidgeon.animation_timer = FRAME_INTERVAL

        # cambiamos a la siguiente imagen
        pidgeon.animation_frame_index = (pidgeon.animation_frame_index + 1) % 2 # que no supere el tamaño del vector de imágenes

# Pasar de pymunk a pygame la posición
def to_pygame(p):
    """Small helper to convert Pymunk vec2d to Pygame integers"""
    return round(p.x), round(display_h - p.y) # Modificamos la conversión de la coordenada y



def main():

    
    #Cargamos y escalamos las imagenes
    image_lr_pidgeon_open = pygame.image.load("ID-PALOMA-2.png")
    image_lr_pidgeon_open = pygame.transform.scale(image_lr_pidgeon_open,(scale_width,scale_height ))

    image_lr_pidgeon_close = pygame.image.load("ID-PALOMA-1.png")
    image_lr_pidgeon_close = pygame.transform.scale(image_lr_pidgeon_close,(scale_width,scale_height ))

    # Agrupamos las imágenes que componen el movimiento de la paloma que surge de la izquierda
    left_pidgeon_frame = [image_lr_pidgeon_open, image_lr_pidgeon_close]

    image_rl_pidgeon_open = pygame.image.load("DI-PALOMA-1.png")
    image_rl_pidgeon_open = pygame.transform.scale(image_rl_pidgeon_open,(scale_width,scale_height ))

    image_rl_pidgeon_close = pygame.image.load("DI-PALOMA-2.png")
    image_rl_pidgeon_close = pygame.transform.scale(image_rl_pidgeon_close,(scale_width,scale_height ))

    # Agrupamos las imágenes que componen el movimiento de la paloma que surge de la derecha
    right_pidgeon_frame = [image_rl_pidgeon_open, image_rl_pidgeon_close]

    image_proyectile = pygame.image.load("basketball.png")
    image_proyectile = pygame.transform.scale(image_proyectile, (5*2, 5*2))

    # Escalar las imágenes del cazador y del arma al cargarlas usando los tamaños definidos
    image_hunter = pygame.image.load("basketball.png")
    image_hunter = pygame.transform.scale(image_hunter, (HUNTER_W, HUNTER_H))

    image_gun = pygame.image.load("basketball.png")
    image_gun = pygame.transform.scale(image_gun, (GUN_W, GUN_H))
    
    # Inicializamos PyGame
    pygame.init()
    # Definimos el tamaño (en píxeles) de la ventana de visualización en PyGame
    screen = pygame.display.set_mode((display_w, display_h))
    # Añadimos un título a la ventana
    pygame.display.set_caption("El destino de la humanidad depende de ti. ¡Acaba con las palomas urbanas!")
    clock = pygame.time.Clock()
    # Inicializamos el espacio en pymunk
    space = pymunk.Space()
    # Ajustamos la gravedad a un valor adecuado. 
    # Recuerda que lo importante es cómo se ve en pantalla, no su valor en el mundo real. 
    # 900 dará como resultado una simulación visualmente atractiva
    space.gravity = (0.0, -900.0) # Modificamos la gravedad 

    

    pidgeons = []
    #draw_options = pymunk.pygame_util.DrawOptions(screen)

    # Crear el cazador y el arma
    hunter_shape = add_hunter(space)
    gun_shape = add_gun(space, hunter_shape)

    ticks_to_next_pidgeon = 10
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit(0)
        
        # Obtener la posición de la mano abierta y controlar el cazador
        hand_x = get_hand_position()
        hunter_shape.body.position = hand_x, hunter_shape.body.position.y
        gun_shape.body.position = hand_x, gun_shape.body.position.y
 
        ticks_to_next_pidgeon -= 1   # Decrementamos el contador de ticks para la siguiente paloma
        if ticks_to_next_pidgeon <= 0:  # Si ha llegado a cero, añadimos una nueva paloma
            ticks_to_next_pidgeon = random.randint(TICKS_MIN, TICKS_MAX)  # Reiniciamos el contador de ticks para que tarde de 1 a 2 s en cargar una paloma
            pidgeon_shape = add_lr_pidgeon(space)     # Añadimos la paloma al espacio
            pidgeons.append(pidgeon_shape)     # Y la añadimos a la lista de palomas
        
        # La función 'step' hace avanzar la simulación un paso (en seg.) en el tiempo cada vez que se la llama.
        # Es mejor usar un paso constante y no ajustarlo en función de lo que tarde cada iteración del bucle. 
        # Debe estar coordinado con lo FPS definidos en PyGame.
        space.step(1/FPS)

        # Rellenamos la ventana con un fondo blanco
        screen.fill((255,255,255))

        pidgeons_to_remove = []  # Lista para almacenar las palomas que deben ser eliminadas
         # Dibujamos las palomas
        for pidgeon in pidgeons:
            update_pidgeon_animation(pidgeon)
            draw_pidgeon(screen,pidgeon)
            draw_pidgeon_with_image(screen,pidgeon, left_pidgeon_frame, right_pidgeon_frame)
            if pidgeon.body.position.x > 850 or pidgeon.body.position.x < -50:   # Si la paloma sale de la pantalla, la marcamos para eliminarla
                pidgeons_to_remove.append(pidgeon)
        # Eliminamos las palomas que han salido de la pantalla        
        for pidgeon in pidgeons_to_remove:
            space.remove(pidgeon, pidgeon.body)
            pidgeons.remove(pidgeon)

        # Dibujamos el cazador y el arma
        draw_hunter(screen, hunter_shape)
        draw_hunter_with_image(screen, hunter_shape, image_hunter)
        draw_gun(screen, gun_shape)
        draw_gun_with_image(screen, gun_shape, image_gun)

        #space.debug_draw(draw_options)

        # Actualiza la ventana de visualización. Muestra todo lo que se dibujó en este frame.
        pygame.display.flip()

        # Limita la velocidad del juego a 50 FPS
        clock.tick(FPS)

if __name__ == '__main__':
    sys.exit(main())