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
import mediapipe as mp  # Para detección de manos
import numpy as np  # Para operaciones con arrays numéricos

# Tamaño de la pantalla
display_h = 800
display_w = 800

# Configuraciones de la simulación
FPS = 50
TICKS_MIN = 1 * FPS  # 50 ticks (1 segundo)
TICKS_MAX = 2 * FPS  # 100 ticks (2 segundos)

# Configuraciones de las imágenes
scale_width = 29 * 1.5
scale_height = 29 * 1.5

# Animaciones de las palomas:
FRAME_INTERVAL = FPS * 1.5 # Las palomas cambian de imagen cada 1.5 segundos

# ========== INICIALIZAR MEDIAPIPE PARA DETECCIÓN DE MANOS ==========
mp_hands = mp.solutions.hands  # Solución de MediaPipe para detección de manos
hands = mp_hands.Hands(  # Crear detector de manos con parámetros específicos
    static_image_mode=False,  # Modo dinámico para video en tiempo real
    max_num_hands=1,  # Detectar solo 1 mano
    min_detection_confidence=0.7,  # Confianza mínima del 70% para detección
    min_tracking_confidence=0.7  # Confianza mínima del 70% para seguimiento
)
mp_drawing = mp.solutions.drawing_utils  # Utilidades para dibujar los puntos de la mano

# Captura de video
cap = cv2.VideoCapture(0)  # Abrir cámara web (índice 0 = cámara por defecto)
hand_x = display_w // 2  # Posición inicial de la mano en el centro

# Función para detectar si la mano está abierta
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

def get_hand_position():
    """Detecta la posición de la mano abierta usando MediaPipe"""
    global hand_x
    
    # Capturar frame de la cámara
    ret, frame = cap.read()
    if not ret:
        return hand_x
    
    # Voltear la imagen horizontalmente para efecto espejo
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Convertir BGR a RGB (OpenCV usa BGR, MediaPipe usa RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detectar manos
    results = hands.process(rgb_frame)
    
    # Si se detecta una mano abierta, obtener su posición
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Verificar si la mano está abierta
        if is_hand_open(hand_landmarks):
            # Usar el punto de la muñeca/palma (punto 0) para controlar el juego
            palm_center = hand_landmarks.landmark[0]
            # Convertir coordenadas normalizadas a píxeles del juego
            hand_x = int(palm_center.x * display_w)
            # Limitar dentro de los bordes
            hand_x = max(30, min(display_w - 30, hand_x))
    
    # Mostrar el video con la detección de mano
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
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
    width = 20
    height = 60
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC) # Creamos el cuerpo del cazador cinemático
    body.position = display_w // 2, 50 # Fijamos su posición en el centro abajo de la pantalla
    shape = pymunk.Poly.create_box(body, (width, height)) # Creamos una forma rectangular para que el cuerpo pueda colisionar
    shape.mass = mass
    shape.radius = width // 2  # Agregar atributo radius para las funciones de dibujo
    space.add(body, shape)
    return shape

def add_gun(space, hunter):
    mass = 2
    width = 10
    height = 30
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC) # Creamos el cuerpo del arma cinemático
    body.position = hunter.body.position.x, hunter.body.position.y + 40 # Fijamos su posición encima del cazador
    shape = pymunk.Poly.create_box(body, (width, height)) # Creamos una forma rectangular para que el cuerpo pueda colisionar
    shape.mass = mass
    shape.radius = width // 2  # Agregar atributo radius para las funciones de dibujo
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
    p = int(hunter.body.position.x), display_h - int(hunter.body.position.y)
    pygame.draw.rect(screen, (0,255,0), (p[0] - hunter.radius, p[1] - hunter.radius, hunter.radius * 2, hunter.radius * 4), 2)

def draw_hunter_with_image(screen,hunter,image):
    p = int(hunter.body.position.x - hunter.radius), (display_h - int(hunter.body.position.y))- hunter.radius
    screen.blit(image, p)

def draw_gun(screen, gun):
    p = int(gun.body.position.x), display_h - int(gun.body.position.y)
    pygame.draw.rect(screen, (255,255,0), (p[0] - gun.radius, p[1] - gun.radius, gun.radius * 2, gun.radius * 3), 2)

def draw_gun_with_image(screen,gun,image):
    p = int(gun.body.position.x - gun.radius), (display_h - int(gun.body.position.y))- gun.radius
    screen.blit(image, p)

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

    image_proyectile= pygame.image.load("basketball.png")
    image_proyectile = pygame.transform.scale(image_proyectile,(5*2,5*2 ))
    
    image_hunter= pygame.image.load("basketball.png")
    image_hunter = pygame.transform.scale(image_hunter,(20*2,60*2 ))
    
    image_gun= pygame.image.load("basketball.png")
    image_gun = pygame.transform.scale(image_gun,(10*2,30*2 ))
    
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