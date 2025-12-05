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


import sys, random, time
random.seed(1) # Hace que la simulación sea igual cada vez, más fácil de depurar
import pygame
import pymunk
import pymunk.pygame_util
import cv2  # Para captura de video desde la cámara
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
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
model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Variable global para almacenar el resultado de la detección
detection_result = None
hand_x = display_w // 2  # Posición inicial de la mano en el centro
gun_rotation_angle = 0  # Ángulo de rotación del arma
is_gun_charged = False  # Estado de carga del arma
was_pinch_pressed = False  # Estado anterior del pellizco

def get_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """Callback para procesar los resultados de la detección de manos"""
    global detection_result
    detection_result = result


def draw_landmarks_on_image(rgb_image, detection_result):
    """Dibuja los landmarks de las manos en la imagen"""
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    # Iterar sobre las manos detectadas
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        
        # Dibujar los landmarks de la mano
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

    return annotated_image


def is_hand_open(hand_landmarks):
    """Detecta si la mano está abierta calculando la distancia entre la muñeca y el dedo medio"""
    # En MediaPipe Tasks, hand_landmarks es una lista de NormalizedLandmark
    # Índice 0 = muñeca, Índice 12 = punta del dedo medio
    wrist = hand_landmarks[0]
    middle_finger_tip = hand_landmarks[12]
    
    # Calcular distancia vertical entre muñeca y punta del dedo medio
    distance = middle_finger_tip.y - wrist.y
    
    # Si la distancia es negativa y el valor absoluto es mayor a 0.15, la mano está abierta
    return distance < -0.15


def get_hand_rotation(hand_landmarks):
    """Calcula el ángulo de rotación de la mano basado en la orientación de los dedos"""
    import math
    # Usar la muñeca (0) y el dedo índice (6) para calcular la rotación
    wrist = hand_landmarks[0]
    index_finger_pip = hand_landmarks[6]  # Articulación PIP del índice
    
    # Calcular el vector desde la muñeca al dedo índice
    dx = index_finger_pip.x - wrist.x
    dy = index_finger_pip.y - wrist.y
    
    # Calcular el ángulo en radianes y convertir a grados
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    # Ajustar el ángulo para Pygame (para visualización)
    angle_pygame = -angle_deg - 90
    
    return angle_pygame


def get_hand_rotation_raw(hand_landmarks):
    """Calcula el ángulo de rotación de la mano en radianes (sin ajustes) para cálculos de física"""
    import math
    # Usar la muñeca (0) y el dedo índice (6) para calcular la rotación
    wrist = hand_landmarks[0]
    index_finger_pip = hand_landmarks[6]  # Articulación PIP del índice
    
    # Calcular el vector desde la muñeca al dedo índice
    dx = index_finger_pip.x - wrist.x
    dy = index_finger_pip.y - wrist.y
    
    # Calcular el ángulo en radianes
    # En coordenadas de cámara (MediaPipe): X va a la derecha, Y va hacia abajo
    # atan2(dy, dx) nos da: 0° a la derecha, 90° hacia abajo, -90° hacia arriba
    angle_rad = math.atan2(-dy, dx)  # Negamos dy porque en pymunk Y va hacia arriba
    
    return angle_rad


def is_pinch_gesture(hand_landmarks):
    """Detecta si la mano está haciendo un gesto de pellizco (pulgar e índice juntos)"""
    # Índice 4 = punta del pulgar, Índice 8 = punta del índice
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    
    # Calcular distancia euclidiana entre pulgar e índice
    distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    
    # Si la distancia es pequeña (< 0.05), el gesto es un pellizco
    return distance < 0.05


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

def add_projectile_from_gun(space, gun_position, hand_landmarks):
    """Crear un proyectil disparado desde el arma según su ángulo"""
    import math
    mass = 1
    radius = 5
    body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
    body.position = gun_position
    
    # Obtener el ángulo en radianes para cálculos de física
    angle_rad = get_hand_rotation_raw(hand_landmarks)
    
    # Calcular velocidad del proyectil
    projectile_speed = 400
    
    # Calcular componentes de velocidad
    vx = projectile_speed * math.cos(angle_rad)
    vy = projectile_speed * math.sin(angle_rad)
    
    body.velocity = (vx, vy)
    
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

def draw_gun(screen, gun, angle=0, is_charged=False):
    # Dibujar rectángulo del arma usando sus dimensiones guardadas
    cx = int(gun.body.position.x)
    cy = display_h - int(gun.body.position.y)
    w = int(getattr(gun, 'width', 10))
    h = int(getattr(gun, 'height', 30))
    rect = (cx - w // 2, cy - h // 2, w, h)
    # Cambiar color según si está cargado
    gun_color = (255, 100, 0) if is_charged else (255, 255, 0)
    pygame.draw.rect(screen, gun_color, rect, 2)

def draw_gun_with_image(screen, gun, image, angle=0, is_charged=False):
    # Dibujar rectángulo del arma y superponer la imagen centrada
    cx = int(gun.body.position.x)
    cy = display_h - int(gun.body.position.y)
    w = int(getattr(gun, 'width', 10))
    h = int(getattr(gun, 'height', 30))
    rect = (cx - w // 2, cy - h // 2, w, h)
    # Cambiar color según si está cargado
    gun_color = (255, 100, 0) if is_charged else (255, 255, 0)
    pygame.draw.rect(screen, gun_color, rect, 2)
    if image is not None:
        # Rotar la imagen según el ángulo de la mano
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rect = rotated_image.get_rect(center=(cx, cy))
        screen.blit(rotated_image, rotated_rect)

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
    projectiles = []  # Lista para almacenar los proyectiles del cazador

    # Crear el cazador y el arma
    hunter_shape = add_hunter(space)
    gun_shape = add_gun(space, hunter_shape)

    ticks_to_next_pidgeon = 10
    
    # Variables de control para disparos
    last_pinch_frame = -100  # Frame del último pellizco detectado
    pinch_cooldown = 10  # Frames de espera entre disparos
    
    # Configurar MediaPipe Tasks
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=get_result)

    cap = cv2.VideoCapture(0)
    
    # Variables locales para el loop
    current_hand_x = hand_x
    current_gun_rotation = 0
    is_gun_charged = False  # Estado de carga del arma (local)
    was_pinch_pressed = False  # Estado anterior del pellizco (local)
    
    with HandLandmarker.create_from_options(options) as landmarker:
        running = True
        frame_count = 0
        current_frame = 0  # Contador de frames
        
        while running and cap.isOpened():
            current_frame += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            
            # Capturar frame de la cámara
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            image = cv2.flip(image, 1)
            h, w, c = image.shape
            
            # Convertir a formato de MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Detectar manos
            frame_timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, frame_timestamp_ms)
            
            # Procesar resultados de detección
            if detection_result is not None:
                image_annotated = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
                
                if len(detection_result.hand_landmarks) > 0:
                    landmarks = detection_result.hand_landmarks[0]
                    
                    # Verificar si la mano está abierta
                    if is_hand_open(landmarks):
                        # Usar la muñeca para controlar el cazador
                        wrist = landmarks[0]
                        # Convertir coordenadas normalizadas a píxeles del juego
                        current_hand_x = int(wrist.x * display_w)
                        # Limitar dentro de los bordes
                        current_hand_x = max(30, min(display_w - 30, current_hand_x))
                        
                        # Calcular el ángulo de rotación del arma basado en la orientación de la mano
                        current_gun_rotation = get_hand_rotation(landmarks)
                    
                    # Detectar gesto de pellizco (pulgar e índice juntos)
                    current_pinch = is_pinch_gesture(landmarks)
                    
                    # Debug: mostrar estado del pellizco
                    if current_pinch:
                        print(f"Frame {current_frame}: Pellizco detectado - Arma cargada")
                    
                    # Si está haciendo pellizco, cargar el arma
                    if current_pinch:
                        is_gun_charged = True
                    
                    # Si estaba haciendo pellizco pero ahora soltó, disparar
                    if was_pinch_pressed and not current_pinch and is_gun_charged:
                        print(f"Frame {current_frame}: Pellizco soltado - DISPARANDO")
                        gun_pos = (int(gun_shape.body.position.x), int(gun_shape.body.position.y))
                        projectile = add_projectile_from_gun(space, gun_pos, landmarks)
                        projectiles.append(projectile)
                        is_gun_charged = False
                        last_pinch_frame = current_frame
                    
                    # Actualizar el estado anterior del pellizco
                    was_pinch_pressed = current_pinch
            
            # Actualizar posición del cazador y el arma
            hunter_shape.body.position = current_hand_x, hunter_shape.body.position.y
            gun_shape.body.position = current_hand_x, gun_shape.body.position.y
            
            ticks_to_next_pidgeon -= 1   # Decrementamos el contador de ticks para la siguiente paloma
            if ticks_to_next_pidgeon <= 0:  # Si ha llegado a cero, añadimos una nueva paloma
                ticks_to_next_pidgeon = random.randint(TICKS_MIN, TICKS_MAX)  # Reiniciamos el contador de ticks
                pidgeon_shape = add_lr_pidgeon(space)     # Añadimos la paloma al espacio
                pidgeons.append(pidgeon_shape)     # Y la añadimos a la lista de palomas
            
            # La función 'step' hace avanzar la simulación un paso (en seg.) en el tiempo cada vez que se la llama.
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

            # Dibujar y actualizar proyectiles del cazador
            projectiles_to_remove = []
            for projectile in projectiles:
                # Dibujar proyectil como círculo
                p = int(projectile.body.position.x), display_h - int(projectile.body.position.y)
                pygame.draw.circle(screen, (0, 0, 0), p, int(projectile.radius), 2)
                
                # Remover proyectiles que salieron de la pantalla
                if projectile.body.position.x > 850 or projectile.body.position.x < -50 or projectile.body.position.y > 850 or projectile.body.position.y < -50:
                    projectiles_to_remove.append(projectile)
            
            # Eliminar proyectiles fuera de pantalla
            for projectile in projectiles_to_remove:
                space.remove(projectile, projectile.body)
                projectiles.remove(projectile)

            # Dibujamos el cazador y el arma
            draw_hunter(screen, hunter_shape)
            draw_hunter_with_image(screen, hunter_shape, image_hunter)
            draw_gun(screen, gun_shape, current_gun_rotation, is_gun_charged)
            draw_gun_with_image(screen, gun_shape, image_gun, current_gun_rotation, is_gun_charged)

            # Actualiza la ventana de visualización
            pygame.display.flip()

            # Mostrar frame de OpenCV con landmarks
            cv2.imshow('Hand Detection', image_annotated if detection_result is not None else image)
            if cv2.waitKey(5) & 0xFF == 27:
                running = False

            # Limita la velocidad del juego a 50 FPS
            clock.tick(FPS)
    
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == '__main__':
    sys.exit(main())