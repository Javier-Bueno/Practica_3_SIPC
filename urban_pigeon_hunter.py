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
# Versión 0.1.0
# - Creación inicial del código
# Versión 0.2.0
# - Modificaciones: Movimiento de las palomas correcto
# Versión 0.3.0
# - Modificaciones: Creación del cazador y movimiento y rotación de la mano.
# Versión 0.3.1
# - Documentación y estructuración correcta del código


# Importaciones:
import sys, random
import time
import math


import pygame              # Elementos del juego
import pymunk              # Motor de físicas 2D
import pymunk.pygame_util  # Dibujar objetos de pymunk usando pygame


import cv2                                              # Capturar la cámara          
import mediapipe as mp                                  # Detección de mano
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np




random.seed(1) # Hace que la simulación sea igual cada vez, más fácil de depurar




# Tamaño de la pantalla
display_h = 800
display_w = 800


# Configuraciones de la simulación
FPS = 50
TICKS_MIN = 1 * FPS  # 50 ticks (1 segundo)
TICKS_MAX = 2 * FPS  # 100 ticks (2 segundos)


# Configuraciones de las imágenes
scale_width = 43.5
scale_height = 43.5


# Animaciones de las palomas:
FRAME_INTERVAL = FPS * 0.25 # Las palomas cambian de imagen cada 1.5 segundos


# Tamaños (en píxeles) del cazador y del arma, usados para dibujado
HUNTER_W = 70
HUNTER_H = 80
GUN_W = 75
GUN_H = 100


# Variables colisiones
COLLISION_TYPE_PIGEON = 1
COLLISION_TYPE_BULLET = 2


SHAPES_TO_REMOVE = []    # Aquí el handler meterá lo que hay que borrar
EXPLOSIONS = []          # Aquí guardaremos dónde dibujar explosiones
EXPLOSION_DURATION = 10  # Cuántos frames dura la explosión


# =================================================
#     Apartado de la configuración de la mano
# =================================================
# -----------------------------------------------------------------
# Configuración para MediaPipe Tasks Hand Landmarker
# -----------------------------------------------------------------
# Ruta al archivo del modelo Tasks (asset) que se va a cargar
model_path = 'hand_landmarker.task'


# Alias a clases útiles de MediaPipe Tasks para crear opciones y el detector
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Variable global donde guardaremos el último resultado de la detección
# Es escrita por el callback `get_result` y leída en el bucle principal.
detection_result = None


# Posición inicial prevista de la mano en el eje X (centro de la pantalla)
hand_x = display_w // 2
# Ángulo de rotación del arma (valor en grados, según convención usada)
gun_rotation_angle = 0


# -----------------------------------------------------------------
# Callback y helpers para trabajar con los resultados de MediaPipe
# -----------------------------------------------------------------


def get_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
   """Callback que MediaPipe llamará cuando haya un nuevo resultado.


   Firma: (result, output_image, timestamp_ms)
   - `result`: objeto `HandLandmarkerResult` con `hand_landmarks` y otros datos
   - `output_image`: imagen opcional con anotaciones que MediaPipe prepara
   - `timestamp_ms`: marca de tiempo en ms del frame


   La función simplemente guarda el `result` en la variable global
   `detection_result` para que el bucle principal pueda consultarlo.
   """
   # Declaramos global para sobrescribir la variable definida fuera
   global detection_result
   # Almacenar el resultado para procesarlo en el bucle principal
   detection_result = result




# ----------------- Funciones de procesamiento de landmarks -----------------


def draw_landmarks_on_image(rgb_image, detection_result):
   """Dibuja los landmarks detectados sobre una copia de la imagen RGB.


   Recibe la imagen RGB en formato numpy y el `detection_result` de
   MediaPipe Tasks. Devuelve una copia anotada para mostrar en una ventana
   de OpenCV si se desea.
   """
   # Extraer la lista de landmarks de cada mano detectada
   hand_landmarks_list = detection_result.hand_landmarks
   # Hacemos una copia de la imagen para no modificar la original
   annotated_image = np.copy(rgb_image)


   # Recorremos cada mano detectada (si hay varias)
   for idx in range(len(hand_landmarks_list)):
       # `hand_landmarks` es una lista de objetos NormalizedLandmark
       hand_landmarks = hand_landmarks_list[idx]


       # Convertir la lista de landmarks a la estructura proto que esperan
       # las utilidades de dibujo de `solutions.drawing_utils`.
       hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
       hand_landmarks_proto.landmark.extend([
           # Por cada landmark en la lista creamos el proto correspondiente
           landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
           for landmark in hand_landmarks
       ])


       # Usamos utilidades de MediaPipe para dibujar landmarks y conexiones
       solutions.drawing_utils.draw_landmarks(
           annotated_image,
           hand_landmarks_proto,
           solutions.hands.HAND_CONNECTIONS,
           solutions.drawing_styles.get_default_hand_landmarks_style(),
           solutions.drawing_styles.get_default_hand_connections_style())


   # Devolvemos la imagen anotada (numpy array)
   return annotated_image




def is_hand_open(hand_landmarks):
   """Heurística simple para detectar mano abierta.


   En este proyecto la heurística compara la posición Y de la muñeca
   (landmark 0) y la punta del dedo medio (landmark 12). Si la punta del
   dedo medio está mucho más arriba (valor Y menor en coordenadas normalizadas)
   se considera la mano abierta.


   `hand_landmarks` es una lista de objetos con atributos `.x`, `.y`, `.z`.
   """
   # Landmark 0 = muñeca
   wrist = hand_landmarks[0]
   # Landmark 12 = punta del dedo medio
   middle_finger_tip = hand_landmarks[12]


   # Distancia vertical (normalizada) entre punta del dedo medio y muñeca
   distance = middle_finger_tip.y - wrist.y


   # Si la diferencia es negativa (el dedo está por encima en la imagen) y
   # su magnitud es mayor que un umbral, consideramos la mano abierta.
   # Umbral empírico: 0.15 (ajustable dependiendo de la cámara y la pose)
   return distance < -0.15




def get_hand_rotation(hand_landmarks):
   """Calcula un ángulo de rotación estimado de la mano.


   Estrategia:
   - Tomamos la muñeca (landmark 0) y una articulación del índice (landmark 6)
   - Calculamos el vector entre ambas y usamos atan2(dy, dx) para obtener
     el ángulo en radianes.
   - Convertimos a grados y transformamos la convención para que funcione
     razonablemente con `pygame.transform.rotate`.


   Devolvemos el ángulo en grados ya preparado para PyGame (signo/offset
   aplicados).
   """


   # Landmark 0 = muñeca
   wrist = hand_landmarks[0]
   # Landmark 6 = articulación PIP del dedo índice (una referencia intermedia)
   index_finger_pip = hand_landmarks[6]


   # Vector desde muñeca hasta la articulación del índice (componentes normalizadas)
   dx = index_finger_pip.x - wrist.x
   dy = index_finger_pip.y - wrist.y


   # Ángulo en radianes: atan2(dy, dx)
   angle_rad = math.atan2(dy, dx)
   # Convertimos a grados para usar en PyGame
   angle_deg = math.degrees(angle_rad)


   # Ajuste de convención: dependiendo de cómo queramos que la imagen rote
   # respecto a la mano puede ser necesario invertir signo y sumar/sustraer 90º.
   # Este ajuste convierte el ángulo a una convención que funciona visualmente
   # con las imágenes usadas en el proyecto.
   angle_pygame = -angle_deg - 90


   return angle_pygame


# Función para disparar (Detecta la distancia entre el pulgar y el índice, si es corta,dispara)
def is_shooting_gesture(hand_landmarks):
   """
   Detecta si el usuario está haciendo el gesto de pinza (índice y pulgar juntos).
   Retorna True si la distancia es menor a un umbral.
   """
   thumb_tip = hand_landmarks[4]
   index_tip = hand_landmarks[5]


   # Calculamos la distancia euclidiana entre los dos puntos
   # D = raiz((x2 - x1)^2 + (y2 - y1)^2)
   distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)


   # Umbral empírico: 0.04 suele funcionar bien para coordenadas normalizadas.
   # Si ves que cuesta disparar, sube a 0.05 o 0.06.
   return distance < 0.05


# =======================================================
# Funciones para añadir los objetos al espacio de pymunk
# =======================================================
def add_pigeon(space): # Paloma
   mass = 3
   width = scale_width
   height = scale_height


   # La paloma es un cuerpo cinemático porque queremos que siga una trayectoria definida.
   body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
   x = random.choice([0, display_w])  # Seleccionamos el rango en x en que se va a crear (bordes de la pantalla)
   y = random.randint(375, display_h - 50)  # Seleccionamos el rango en y en que se va a crear
   body.position = x, y # Fijamos su posición


   if x == 0:
       body.velocity = 120, 0 # Asignamos la velocidad positiva (hacia la derecha)
       direction = "RIGHT" # La paloma surge del origen (para diferenciar sprites)
   else:
       body.velocity = -120, 0 # Asignamos la velocidad negativa (hacia la izquierda)
       direction = "LEFT" # La paloma surge de la derecha (para diferenciar sprites)
  
   shape = pymunk.Poly.create_box(body, (width, height)) # Creamos una forma rectangular para que el cuerpo pueda colisionar
   shape.mass = mass # Asignamos la masa a la forma
   shape.width = width
   shape.height = height
   shape.direction = direction
   shape.animation_frame_index = 0 # Índice inicial para el conjunto con las imágenes de la paloma
   shape.animation_timer  = FRAME_INTERVAL # Temporizador para saber cuando cambiar la imagen


   # COLISIONES
   shape.collision_type = COLLISION_TYPE_PIGEON


   space.add(body, shape) # Añadimos el cuerpo con su forma al espacio


   return shape




def add_proyectile(space, position):
   """Crea un proyectil (ej. disparo de una paloma) en posición dada.
   - `position`: tupla (x, y) en coordenadas Pymunk
   - Devuelve la `shape` del círculo creado.
   """
   mass = 1
   radius = 5


   # Cuerpo dinámico para que la física lo afecte (gravedad, colisiones)
   body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
   body.position = position


   shape = pymunk.Circle(body, radius)
   shape.mass = mass
   space.add(body, shape)
   return shape




def add_hunter(space):
   """Crea el cuerpo del cazador (jugador) como un body cinemático.
   El cazador se representa por un rectángulo y se posiciona en la parte
   inferior de la pantalla. Es cinemático porque su posición la fijamos
   directamente en el bucle principal (no queremos que la física la mueva).
   """
   mass = 10
   width = HUNTER_W
   height = HUNTER_H


   body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
   # Posición inicial: centro en X, 50 px en Y (altura sobre la base)
   body.position = display_w // 2, 100


   shape = pymunk.Poly.create_box(body, (width, height))
   shape.mass = mass
   # Atributos auxiliares para dibujado
   shape.width = width
   shape.height = height
   shape.draw_radius = width // 2
   space.add(body, shape)


   return shape


def add_gun(space, hunter):
   """Crea el cuerpo del arma asociado al cazador.
   La posición inicial del arma se fija en relación al cuerpo `hunter`.
   El arma también es cinemática y se moverá junto al cazador en X.
   """
   mass = 2
   width = GUN_W
   height = GUN_H


   body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
   # Posicionamos el arma por encima del cazador (offset +40 en Y)
   body.position = hunter.body.position.x, hunter.body.position.y + 40
  
   shape = pymunk.Poly.create_box(body, (width, height))
   shape.mass = mass
   shape.width = width
   shape.height = height
   shape.draw_radius = width // 2
   space.add(body, shape)


   return shape


def add_bullet(space, position, angle_degrees):
   """
   Crea una bala dinámica en la posición dada y con la velocidad angular correcta.
   """
   mass = 1
   radius = 5
   speed = 1000  # Velocidad de la bala (ajustable)


   body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
   #body.gravity_scale = 0
   # IMPORTANTE: Revertimos la transformación que hiciste en 'get_hand_rotation'
   # para recuperar el ángulo matemático real y calcular la trayectoria.
   # Tu fórmula era: angle_pygame = -angle_deg - 90
   # Despejando: angle_deg = -angle_pygame - 90
   math_angle_degrees = -angle_degrees - 90
   math_angle_radians = math.radians(math_angle_degrees)
  
   # Calculamos los componentes del vector velocidad (Trigonometría básica)
   # Cos = eje X, Sin = eje Y
   vx = speed * math.cos(math_angle_radians)
   vy = -(speed * math.sin(math_angle_radians))


   body.position = position
   body.velocity = vx, vy # Asignamos el impulso inicial
  
   shape = pymunk.Circle(body, radius)
   shape.mass = mass
   shape.elasticity = 0.0 # Las balas no suelen rebotar mucho
   shape.friction = 0.5
  
   # Asignamos el tipo de colisión que definiste arriba
   shape.collision_type = COLLISION_TYPE_BULLET
  
   # Opcional: Filtro para que la bala no colisione con el propio cazador (si fuera necesario)
   # shape.filter = pymunk.ShapeFilter(group=1)




   space.add(body, shape)
   return shape




# ================================================================
# Funciones para dibujar los objetos en pantalla (HAY QUE REVISARLAS)
# ================================================================


# Dibuja la paloma como un rectángulo rojo
def draw_pigeon(screen, pigeon):
   p = int(pigeon.body.position.x), display_h - int(pigeon.body.position.y)
   #pygame.draw.circle(screen, (0,0,255), p, int(pigeon.radius), 2)
   pygame.draw.rect(screen, (255,0,0), (p[0] - pigeon.width / 2, p[1] - pigeon.height / 2, pigeon.width, pigeon.height), 2)




# Dibuja la imagen de la paloma
# La función recibe, la pantalla, la paloma, el conjunto de imágenes i-d y el de imágenes d-i
def draw_pigeon_with_image(screen, pigeon, lr_frame, rl_frame):
   current_index = pigeon.animation_frame_index
   if pigeon.direction == "LEFT":
       image_to_draw = rl_frame[current_index]
   else:
       image_to_draw = lr_frame[current_index]
      
   p = int(pigeon.body.position.x - pigeon.width / 2), (display_h - int(pigeon.body.position.y + (pigeon.height / 2)))
   screen.blit(image_to_draw, p)




# Dibujo del proyectil como un círculo rojo
def draw_proyectile(screen, proyectile):
   p = int(proyectile.body.position.x), display_h - int(proyectile.body.position.y)
   pygame.draw.circle(screen, (160,200,0), p, int(proyectile.radius), 2)




# Dibujo de la imagen del proyectil
def draw_proyectile_with_image(screen,proyectile,image):
   p = int(proyectile.body.position.x - proyectile.radius), (display_h - int(proyectile.body.position.y))- proyectile.radius
   screen.blit(image, p)




def draw_hunter(screen, hunter):
   """Dibuja el cazador como rectángulo verde usando sus dimensiones."""
   cx = int(hunter.body.position.x)
   cy = display_h - int(hunter.body.position.y)
   w = int(getattr(hunter, 'width', 20))
   h = int(getattr(hunter, 'height', 60))
   rect = (cx - w // 2, cy - h // 2, w, h)
   pygame.draw.rect(screen, (0,255,0), rect, 2)




def draw_hunter_with_image(screen, hunter, image):
   """Dibuja el rectángulo del cazador y superpone la imagen centrada si existe."""
   cx = int(hunter.body.position.x)
   cy = display_h - int(hunter.body.position.y)
   w = int(getattr(hunter, 'width', 20))
   h = int(getattr(hunter, 'height', 60)).Body(body_type=pymunk.
   if image is not None:
       ix, iy = image.get_width(), image.get_height()
       img_pos = (cx - ix // 2, cy - iy // 2)
       screen.blit(image, img_pos)




def draw_gun(screen, gun, angle=0):
   """Dibuja el arma como rectángulo amarillo (fallback visual)."""
   cx = int(gun.body.position.x)
   cy = display_h - int(gun.body.position.y)
   w = int(getattr(gun, 'width', 10))
   h = int(getattr(gun, 'height', 30))
   rect = (cx - w // 2, cy - h // 2, w, h)
   pygame.draw.rect(screen, (255,255,0), rect, 2)




def draw_gun_with_image(screen, gun, image, angle=0):
   """Dibuja el arma y rota la imagen según `angle` (grados).
   - `angle` se pasa a `pygame.transform.rotate` que rota la imagen en sentido horario
     por defecto respecto de la imagen original.
   """
   cx = int(gun.body.position.x)
   cy = display_h - int(gun.body.position.y)
   w = int(getattr(gun, 'width', 10))
   h = int(getattr(gun, 'height', 30))
   rect = (cx - w // 2, cy - h // 2, w, h)
   pygame.draw.rect(screen, (255,255,0), rect, 2)
   if image is not None:
       # Rotar la imagen y obtener un rect centrado en (cx, cy)
       rotated_image = pygame.transform.rotate(image, angle)
       rotated_rect = rotated_image.get_rect(center=(cx, cy))
       screen.blit(rotated_image, rotated_rect)


def draw_bullet(screen, bullet):
   """
   Dibuja la bala como un círculo sólido gris oscuro/negro.
   """
   # Conversión de coordenadas Pymunk (y hacia arriba) a Pygame (y hacia abajo)
   p = int(bullet.body.position.x), display_h - int(bullet.body.position.y)
  
   # Dibujamos un círculo relleno
   # Color (50, 50, 50) es un gris muy oscuro, casi negro
   pygame.draw.circle(screen, (50, 50, 50), p, int(bullet.radius))




def draw_explosion(screen, explosion_data):
   """Dibuja un círculo temporal para simular una explosión."""
   # El color y el radio cambian según el tiempo restante
   timer = explosion_data['timer']
  
   # Calculamos el radio y la intensidad del color (decrece con el tiempo)
   # radio maximo de 20 cuando el timer es EXPLOSION_DURATION
   max_radius = 20
   radius = int(max_radius * (timer / EXPLOSION_DURATION))
  
   # Color de amarillo a rojo (o simplemente un color que se desvanece)
   # Por ejemplo, rojo si está reciente, amarillo si está terminando
   # Usamos un color fijo para simplicidad: Naranja (255, 165, 0)
   color = (255, 100 + int(100 * (timer / EXPLOSION_DURATION)), 0)
  
   pos = explosion_data['pos']
  
   # Dibujamos un círculo relleno
   pygame.draw.circle(screen, color, pos, radius)




# =======================================================
#    Funciones de Manejo de Colisiones Pymunk
# =======================================================


def bullet_hits_pigeon(arbiter, space, data):
   """
   Callback que se ejecuta cuando una bala (BULLET) colisiona con una paloma (PIGEON).
   """
   # Obtener las formas que colisionaron. Pymunk las pasa en el orden definido en add_collision_handler.
   pigeon_shape, bullet_shape = arbiter.shapes
  
   # 1. Registrar la explosión
   # Registramos la posición donde debe dibujarse la explosión
   explosion_pos = to_pygame(pigeon_shape.body.position) # Convertir a coordenadas Pygame (y invertida)
  
   # Almacenamos la posición y el temporizador/duración
   EXPLOSIONS.append({'pos': explosion_pos, 'timer': EXPLOSION_DURATION})
  
   # 2. Marcar para eliminación (no eliminamos directamente aquí)
   SHAPES_TO_REMOVE.append(pigeon_shape)
   SHAPES_TO_REMOVE.append(bullet_shape)
  
   # Devolver False para indicar a Pymunk que detenga el procesamiento físico de esta colisión
   # (ya que vamos a eliminar los objetos).
   return False




def setup_collision_handler(space):
   """
   Configura el manejador de colisiones (Sintaxis antigua ULL).
   """
   # Usamos 'on_collision' que es la función disponible en tu versión de laboratorio.
   # Usamos el parámetro 'begin' para detectar el momento exacto del impacto.
   space.on_collision(COLLISION_TYPE_PIGEON, COLLISION_TYPE_BULLET, begin=bullet_hits_pigeon)






# ====================================
#        Funciones auxiliares
# ====================================
def update_pigeon_animation(pigeon):
   """Actualiza temporizador e índice de animación de la paloma."""
   pigeon.animation_timer -= 1
   if pigeon.animation_timer <= 0:
       # Reiniciar temporizador al intervalo configurado
       pigeon.animation_timer = FRAME_INTERVAL
       # Avanzar índice circularmente (aquí hay 2 frames)
       pigeon.animation_frame_index = (pigeon.animation_frame_index + 1) % 2




# Pasar de pymunk a pygame la posición
def to_pygame(p):
   """Small helper to convert Pymunk vec2d to Pygame integers"""
   return round(p.x), round(display_h - p.y) # Modificamos la conversión de la coordenada y




# ------------------------------------------------------------
# Función principal: inicializa recursos, bucle principal y limpieza
# ------------------------------------------------------------
def main():
   #Cargamos y escalamos las imagenes
   image_lr_pigeon_open = pygame.image.load("Imagenes/Izquierda-Derecha-Paloma/ID-PALOMA-2.png")
   image_lr_pigeon_open = pygame.transform.scale(image_lr_pigeon_open,(scale_width, scale_height ))


   image_lr_pigeon_close = pygame.image.load("Imagenes/Izquierda-Derecha-Paloma/ID-PALOMA-1.png")
   image_lr_pigeon_close = pygame.transform.scale(image_lr_pigeon_close,(scale_width, scale_height ))


   # Agrupamos las imágenes que componen el movimiento de la paloma que surge de la izquierda
   left_pigeon_frame = [image_lr_pigeon_open, image_lr_pigeon_close]


   image_rl_pigeon_open = pygame.image.load("Imagenes/Derecha-Izquierda-Paloma/DI-PALOMA-1.png")
   image_rl_pigeon_open = pygame.transform.scale(image_rl_pigeon_open,(scale_width, scale_height ))


   image_rl_pigeon_close = pygame.image.load("Imagenes/Derecha-Izquierda-Paloma/DI-PALOMA-2.png")
   image_rl_pigeon_close = pygame.transform.scale(image_rl_pigeon_close,(scale_width,scale_height ))


   # Agrupamos las imágenes que componen el movimiento de la paloma que surge de la derecha
   right_pigeon_frame = [image_rl_pigeon_open, image_rl_pigeon_close]


   # Imagen y escala para proyectil (ejemplo: basketball.png)
   image_proyectile = pygame.image.load("basketball.png")
   image_proyectile = pygame.transform.scale(image_proyectile, (5*2, 5*2))


   # Imagen para el cazador (se utiliza el mismo asset como placeholder)
   image_hunter = pygame.image.load("Imagenes/Cazador/Cazador.png")
   image_hunter = pygame.transform.scale(image_hunter, (HUNTER_W, HUNTER_H))


   # Imagen para el arma (placeholder)
   image_gun = pygame.image.load("Imagenes/Rifle/Rifle.png")
   image_gun = pygame.transform.scale(image_gun, (GUN_W, GUN_H))
  


   # ------------------ Inicialización de PyGame y Pymunk ------------------
   pygame.init()
   screen = pygame.display.set_mode((display_w, display_h))
   pygame.display.set_caption("El destino de la humanidad depende de ti. ¡Acaba con las palomas urbanas!")
   clock = pygame.time.Clock()


   # Creamos el espacio físico donde se simulará la física
   space = pymunk.Space()
   # Ajuste de la gravedad (Pymunk usa coordenadas Y positivas hacia arriba;
   # en pantalla invertimos al dibujar). Aquí una gravedad alta negativa para ver caída.
   space.gravity = (0.0, -900.0)


   setup_collision_handler(space)


   # Lista que contendrá las shapes de las palomas activas
   pigeons = []




   # Creamos el cazador y el arma (shapes con bodies cinemáticos)
   hunter_shape = add_hunter(space)
   gun_shape = add_gun(space, hunter_shape)




   # Temporizador para la siguiente paloma (en ticks)
   ticks_to_next_pigeon = 10


   # ------------------ Configuración de MediaPipe HandLandmarker ------------------
   # Creamos las opciones para ejecutar el modelo en modo LIVE_STREAM con callback
   options = HandLandmarkerOptions(
       base_options=BaseOptions(model_asset_path=model_path),
       running_mode=VisionRunningMode.LIVE_STREAM,
       result_callback=get_result)


   # Abrimos la cámara por defecto (indice 0). Si no existe se fallará.
   cap = cv2.VideoCapture(0)


   # Variables locales que mantienen el estado entre frames
   current_hand_x = hand_x
   current_gun_rotation = 0


   # Crear el landmarker con las opciones definidas. Usamos el context manager
   # `with` para garantizar la liberación de recursos al salir.
   with HandLandmarker.create_from_options(options) as landmarker:
       running = True
       frame_count = 0
      
       bullets = []          # Lista para almacenar las balas vivas
       shot_cooldown = 0     # Contador para limitar la cadencia de tiro
       SHOT_COOLDOWN_MAX = FPS / 5 # Cantidad de ticks entre disparos (aprox 0.3 segundos a 50 FPS)


       # ------------------ Bucle principal del juego ------------------
       while running and cap.isOpened():
           # Gestionar eventos de PyGame (ventana, teclado)
           for event in pygame.event.get():
               if event.type == pygame.QUIT:
                   running = False
               elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                   running = False


           # Capturar un frame desde la cámara
           success, image = cap.read()
           if not success:
               # Si no se pudo capturar, ignoramos el frame y seguimos
               print("Ignoring empty camera frame.")
               continue


           # Flip horizontal para que el espejo sea más natural para el usuario
           image = cv2.flip(image, 1)
           h, w, c = image.shape


           # Convertir BGR (OpenCV) a RGB y crear un `mp.Image` para MediaPipe
           mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


           # Enviamos el frame a MediaPipe en modo asíncrono. Pasamos timestamp
           frame_timestamp_ms = int(time.time() * 1000)
           landmarker.detect_async(mp_image, frame_timestamp_ms)


           # Procesar el resultado guardado por el callback (si existe)
           if detection_result is not None:
               # Generar imagen anotada para mostrar (opcional)
               image_annotated = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)


               # Si se detectó al menos una mano, tomamos la primera
               if len(detection_result.hand_landmarks) > 0:
                   landmarks = detection_result.hand_landmarks[0]


                   # Si la mano está abierta usamos la muñeca para mover al cazador
                   #if is_hand_open(landmarks):
                   # Landmark 0 (muñeca) tiene coords normalizadas [0..1]
                   wrist = landmarks[0]
                   # Convertir a pixeles de pantalla
                   current_hand_x = int(wrist.x * display_w)
                   # Limitar posición dentro de la pantalla (margen 30 px)
                   current_hand_x = max(30, min(display_w - 30, current_hand_x))


                   # Calcular el ángulo del arma según la orientación de la mano
                   current_gun_rotation = get_hand_rotation(landmarks)


                   # --- LÓGICA DE DISPARO (NUEVO) ---
                   # Solo disparamos si detectamos el gesto Y el arma está "fría"
                   if is_shooting_gesture(landmarks) and shot_cooldown <= 0:
                      
                       # 1. Calcular dónde está la punta del cañón
                       # El arma está en (gun_shape.body.position.x, gun_shape.body.position.y)
                       # La altura del arma es GUN_H (aprox 100px). La punta está "arriba".
                      
                       # Matemáticamente, rotamos el punto (0, altura/2) según el ángulo del arma
                       # Pero para simplificar, usaremos la posición del cuerpo del arma + un offset
                       # Como tu rotación es compleja, una aproximación decente es:
                       spawn_pos = (gun_shape.body.position.x, gun_shape.body.position.y + 50)
                      
                       # 2. Crear la bala
                       new_bullet = add_bullet(space, spawn_pos, current_gun_rotation)
                       bullets.append(new_bullet)
                      
                       # 3. Reiniciar el cooldown
                       shot_cooldown = SHOT_COOLDOWN_MAX
                   shot_cooldown -= 1
           # Actualizar posiciones del cazador y el arma para que sigan la mano
           hunter_shape.body.position = current_hand_x, hunter_shape.body.position.y
           gun_shape.body.position = current_hand_x, gun_shape.body.position.y




           # Controlar aparición de nuevas palomas según el temporizador
           ticks_to_next_pigeon -= 1
           if ticks_to_next_pigeon <= 0:
               ticks_to_next_pigeon = random.randint(TICKS_MIN, TICKS_MAX)
               pigeon_shape = add_pigeon(space)
               pigeons.append(pigeon_shape)


           # Avanzar la simulación física un paso (delta = 1/FPS segundos)
           space.step(1 / FPS)


           for shape in SHAPES_TO_REMOVE:
               if shape in pigeons:
                   pigeons.remove(shape)
               if shape in bullets:
                   bullets.remove(shape)
               space.remove(shape, shape.body)
          
           SHAPES_TO_REMOVE.clear()


           # Actualizar y eliminar explosiones
           explosions_to_remove = []
           for exp in EXPLOSIONS:
               exp['timer'] -= 1
               if exp['timer'] <= 0:
                   explosions_to_remove.append(exp)
          
           for exp in explosions_to_remove:
               EXPLOSIONS.remove(exp)




                  


           # Dibujado: limpiar la pantalla y dibujar todos los elementos
           screen.fill((255,255,255))


           # Dibujar y actualizar animación de palomas
           pigeons_to_remove = []
           for pigeon in pigeons:
               update_pigeon_animation(pigeon)
               draw_pigeon(screen, pigeon)
               draw_pigeon_with_image(screen, pigeon, left_pigeon_frame, right_pigeon_frame)
               # Si la paloma sale de la pantalla la marcamos para eliminarla
               if pigeon.body.position.x > 850 or pigeon.body.position.x < -50:
                   pigeons_to_remove.append(pigeon)


           # --- GESTIÓN DE BALAS ---
           bullets_to_remove = []
           for bullet in bullets:
               draw_bullet(screen, bullet) # Usamos la función que te di antes
              
               # Eliminar balas que salen de la pantalla para liberar memoria
               # (Coordenadas Pymunk: Y aumenta hacia arriba, X igual que pantalla)
               bx, by = bullet.body.position
               if bx < 0 or bx > display_w or by < 0 or by > display_h:
                   bullets_to_remove.append(bullet)


           for b in bullets_to_remove:
               space.remove(b.body, b) # Importante: quitar del espacio se
               bullets.remove(b)       # Quitar de la lista lógica
      


           # Eliminamos palomas fuera de la pantalla del espacio y la lista
           for pigeon in pigeons_to_remove:
               space.remove(pigeon, pigeon.body)
               pigeons.remove(pigeon)


           for exp in EXPLOSIONS:
               draw_explosion(screen, exp)


           # Dibujar jugador y arma (con la rotación calculada)
           draw_gun(screen, gun_shape, current_gun_rotation)
           draw_gun_with_image(screen, gun_shape, image_gun, current_gun_rotation)
           draw_hunter(screen, hunter_shape)
           draw_hunter_with_image(screen, hunter_shape, image_hunter)


          


           # Actualizar la pantalla de PyGame con todo lo dibujado
           pygame.display.flip()


           # Mostrar la ventana de OpenCV con la imagen anotada (landmarks)
           cv2.imshow('Hand Detection', image_annotated if detection_result is not None else image)
           if cv2.waitKey(5) & 0xFF == 27:
               running = False


           # Mantener la velocidad del bucle en el framerate objetivo
           clock.tick(FPS)
  
   # ------------------ Limpieza final ------------------
   cap.release()
   cv2.destroyAllWindows()
   pygame.quit()






# Lanzamiento del programa cuando se ejecuta como script principal
if __name__ == '__main__':
   # `sys.exit(main())` permite devolver el código de salida si fuese necesario
   sys.exit(main())
