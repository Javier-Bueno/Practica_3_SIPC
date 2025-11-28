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
# Versión 1.0.0
# - Creación inicial del código

import sys, random
random.seed(1) # Hace que la simulación sea igual cada vez, más fácil de depurar
import pygame
import pymunk
import pymunk.pygame_util

# Tamaño de la pantalla
display_h = 600
display_w = 600

# Fucniones para añadir los objetos al espacio de pymunk
def add_pidgeon(space):
    mass = 3
    radius = 20
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC) # Creamos el cuerpo de la paloma cinemático
    x = random.choice([0, 600])  # Seleccionamos el rango en x en que se va a crear (bordes de la pantalla)
    y = random.randint(120, 480)  # Seleccionamos el rango en y en que se va a crear
    body.position = x, y # Fijamos su posición
    if x == 0:
        body.velocity = 100, 0 # Asignamos una velocidad aleatoria en x
    else:
        body.velocity = -100, 0 # Asignamos una velocidad aleatoria en x
    
    shape = pymunk.Circle(body, radius) # Creamos una forma circular para que el cuerpo pueda colisionar
    shape.mass = mass # Asignamos la masa a la forma 
    space.add(body, shape) # Añadimos el cuerpo con su forma al espacio
    return shape

def add_proyectile(space, position):
    mass = 1
    radius = 5
    body = pymunk.Body(body_type=pymunk.Body.DYNAMIC) # Creamos el cuerpo del proyectil dinámico
    body.position = position # Fijamos su posición a la posición de la paloma
    shape = pymunk.Circle(body, radius)
    shape.mass = mass
    space.add(body, shape)
    return shape

def add_hunter(space):
    mass = 10
    width = 20
    height = 60
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC) # Creamos el cuerpo del cazador cinemático
    body.position = display_w // 2, 50 # Fijamos su posición en el centro abajo de la pantalla
    shape = pymunk.Poly.create_box(body, (width, height)) # Creamos una forma rectangular para que el cuerpo pueda colisionar
    shape.mass = mass
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
    space.add(body, shape)
    return shape


# Funciones para dibujar los objetos en pantalla (HAY QUE REVISARLAS)
def draw_pidgeon(screen, pidgeon):
    p = int(pidgeon.body.position.x), display_h - int(pidgeon.body.position.y)
    pygame.draw.circle(screen, (0,0,255), p, int(pidgeon.radius), 2)

def draw_pidgeon_with_image(screen,pidgeon,image):
    p = int(pidgeon.body.position.x - pidgeon.radius), (display_h - int(pidgeon.body.position.y))- pidgeon.radius
    screen.blit(image, p)

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

# Pasar de pymunk a pygame la posición
def to_pygame(p):
    """Small helper to convert Pymunk vec2d to Pygame integers"""
    return round(p.x), round(display_h - p.y) # Modificamos la conversión de la coordenada y



def main():
    #Cargamos y escalamos las imagenes
    image_pidgeon= pygame.image.load("basketball.png")
    image_pidgeon = pygame.transform.scale(image_pidgeon,(20*2,20*2 ))
    
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
    pygame.display.set_caption("Acaba con las palomas urbanas!")
    clock = pygame.time.Clock()
    # Inicializamos el espacio en pymunk
    space = pymunk.Space()
    # Ajustamos la gravedad a un valor adecuado. 
    # Recuerda que lo importante es cómo se ve en pantalla, no su valor en el mundo real. 
    # 900 dará como resultado una simulación visualmente atractiva
    space.gravity = (0.0, -900.0) # Modificamos la gravedad 

    

    pidgeons = []
    #draw_options = pymunk.pygame_util.DrawOptions(screen)


    ticks_to_next_pidgeon = 10
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit(0)
 
        ticks_to_next_pidgeon -= 1   # Decrementamos el contador de ticks para la siguiente paloma
        if ticks_to_next_pidgeon <= 0:  # Si ha llegado a cero, añadimos una nueva paloma
            ticks_to_next_pidgeon = 20   # Reiniciamos el contador de ticks
            pidgeon_shape = add_pidgeon(space)     # Añadimos la paloma al espacio
            pidgeons.append(pidgeon_shape)     # Y la añadimos a la lista de palomas
        
        # La función 'step' hace avanzar la simulación un paso (en seg.) en el tiempo cada vez que se la llama.
        # Es mejor usar un paso constante y no ajustarlo en función de lo que tarde cada iteración del bucle. 
        # Debe estar coordinado con lo FPS definidos en PyGame.
        space.step(1/50.0)

        # Rellenamos la ventana con un fondo blanco
        screen.fill((255,255,255))

        pidgeons_to_remove = []  # Lista para almacenar las palomas que deben ser eliminadas
         # Dibujamos las palomas
        for pidgeon in pidgeons:
            draw_pidgeon(screen,pidgeon)
            draw_pidgeon_with_image(screen,pidgeon, image_pidgeon)
            if pidgeon.body.position.x > 550 or pidgeon.body.position.x < 50:   # Si la paloma sale de la pantalla, la marcamos para eliminarla
                pidgeons_to_remove.append(pidgeon)
        # Eliminamos las palomas que han salido de la pantalla        
        for pidgeon in pidgeons_to_remove:
            space.remove(pidgeon, pidgeon.body)
            pidgeons.remove(pidgeon)

        #space.debug_draw(draw_options)

        # Actualiza la ventana de visualización. Muestra todo lo que se dibujó en este frame.
        pygame.display.flip()

        # Limita la velocidad del juego a 50 FPS
        clock.tick(50)

if __name__ == '__main__':
    sys.exit(main())