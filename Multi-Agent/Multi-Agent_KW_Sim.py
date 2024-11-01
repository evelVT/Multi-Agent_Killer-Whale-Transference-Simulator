import numpy as np
from killer_whale import KillerWhale
from utils import draw_whales, draw_whale
from environment import Environment       
import random
import cProfile
import sys
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from copy import deepcopy
sys.stdout.reconfigure(line_buffering=False)
environment_dimensions = (1280, 720)


# rc
dialect_size = 500
environment = Environment(environment_dimensions, [], [])    # (1280 x 720)
agentlist = []
for i in range(5):
    agentlist.append(KillerWhale((random.randint(0, environment_dimensions[0]-1), 
                                  random.randint(0, environment_dimensions[1]-1)), 
                                  np.random.randint(0, 2, dialect_size, dtype=np.uint8), 
                                  abs(np.random.normal(12, 3))))

environment.add_agents(agentlist)



# simulation functions:
#   reset
#   pause/unpause

# simulation params:
#   speed (ticks/s)
#   fertility






import ctypes
import pygame
import sys
import os
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox

# Increase Dots Per inch so it looks sharper
ctypes.windll.shcore.SetProcessDpiAwareness(True)

# Initialize Pygame
pygame.init()

# Pygame config
fps = 1
fpsClock = pygame.time.Clock()

# Window dimensions
canvas_width, canvas_height = environment_dimensions
side_panel_width = 400
button_panel_height = 100
window_width = canvas_width + side_panel_width
window_height = canvas_height + button_panel_height

# Colors
background_color = (255, 255, 255)
side_panel_color = (40, 40, 40)
white = (255, 255, 255)

# Screen setup
screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE) #, pygame.RESIZABLE might lead to unexpected results?
pygame.display.set_caption('Killer Whale Dialect Transference Simulator')

# Surfaces
canvas_surface = pygame.Surface((canvas_width, canvas_height))
canvas_surface.fill((0, 0, 0))
draw_surface = pygame.Surface((canvas_width, canvas_height))
draw_surface.fill((0, 0, 0))
food_surface = pygame.Surface((canvas_width, canvas_height))
food_surface.fill((0, 0, 0))
button_surface = pygame.Surface((canvas_width, button_panel_height))
button_surface.fill(side_panel_color)
print(pygame.surfarray.pixels3d(canvas_surface).shape) # pygame.surfarray.make_surface(Z)

# Font
font = pygame.font.SysFont(None, 36)

# Create sliders using pygame-widgets
sliders = []
text_boxes = []
label_boxes = []
env_parameters = {"Surface_size": (canvas_width, canvas_height)}

# Slider properties
slider_configs = [
    {"name": "Size", "min": 1, "max": 50, "start": agentlist[0].size, "y_pos": 100},
    {"name": "Dialect size", "min": 5, "max": 50000, "start": 5000, "y_pos": 180},
    {"name": "Transference", "min": 1, "max": 10000, "start": agentlist[0].transference*10000, "y_pos": 260},
    {"name": "Mutation", "min": 0, "max": 10000, "start": agentlist[0].mutation*10000, "y_pos": 340},
    {"name": "Grouping Dialect", "min": 1, "max": 10000, "start": agentlist[0].grouping_dialect*10000, "y_pos": 420},
    {"name": "Communication range", "min": 0, "max": 500, "start": agentlist[0].comm_range, "y_pos": 500},
    {"name": "Max comm difference", "min": 0, "max": 10000, "start": agentlist[0].comm_dialect_difference*10000, "y_pos": 580},
    {"name": "Follow range", "min": 0, "max": 500, "start": agentlist[0].follow_range, "y_pos": 660},
    {"name": "Movement distance", "min": 1, "max": 25, "start": agentlist[0].max_distance, "y_pos": 740},
]

for config in slider_configs:
    # Create a label for each slider
    label_box = TextBox(
        screen, canvas_width + 20, config['y_pos'] - 30, 200, 30,
        fontSize=24, borderThickness=0, textColour=white, colour=side_panel_color
    )
    label_box.setText(config['name'])
    label_boxes.append(label_box)

    # Create the slider itself
    slider = Slider(
        screen, canvas_width + 20, config['y_pos'], 200, 20,
        min=config['min'], max=config['max'], step=1, initial=config['start']
    )
    sliders.append(slider)

    # Create a textbox to display the current value of the slider
    text_box = TextBox(
        screen, canvas_width + 240, config['y_pos'] - 5, 80, 30,
        fontSize=24, borderThickness=1
    )
    if config['name'] == 'Mutation': text_box.setText(str(config['start']/10000))
    if config['name'] == 'Grouping Dialect': text_box.setText(str(config['start']/10000))
    else: text_box.setText(str(config['start']))
    text_boxes.append(text_box)

# Variables
buttons = []

# Initial color
draw_color = [255, 255, 255]

# Initial brush size
brush_size = 30
brush_size_steps = 5

# Button Class
class Button():
    def __init__(self, x, y, width, height, button_text='Button', on_click_function=None, hold_press=False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.on_click_function = on_click_function
        self.hold_press = hold_press
        self.pressed = False

        self.fillcolors = {
            'normal': '#ffffff',
            'hover': '#666666',
            'pressed': '#333333',
        }

        self.buttonsurface = pygame.Surface((self.width, self.height))
        self.buttonrectangle = pygame.Rect(self.x, self.y, self.width, self.height)
        self.button_text = font.render(button_text, True, (20, 20, 20))
        self.already_pressed = False
        buttons.append(self)

    def process(self):
        mouse_pos = pygame.mouse.get_pos()
        if not self.pressed:
            self.buttonsurface.fill(self.fillcolors['normal'])
        else:
            self.buttonsurface.fill(self.fillcolors['pressed'])
        
        if self.buttonrectangle.collidepoint(mouse_pos):
            self.buttonsurface.fill(self.fillcolors['hover'])

            if pygame.mouse.get_pressed(num_buttons=3)[0]:
                self.buttonsurface.fill(self.fillcolors['pressed'])
                if self.hold_press and not self.already_pressed:
                    self.on_click_function()
                    self.pressed = not self.pressed
                    self.already_pressed = True
                elif not self.already_pressed:
                    self.on_click_function()
                    self.already_pressed = True
            else:
                self.already_pressed = False
        self.buttonsurface.blit(self.button_text, [
            self.buttonrectangle.width/2 - self.button_text.get_rect().width/2,
            self.buttonrectangle.height/2 - self.button_text.get_rect().height/2
        ])
        button_surface.blit(self.buttonsurface, self.buttonrectangle)


# Handler functions
# Changing the color
def change_color(color):
    global draw_color, drawing
    if not drawing: draw_color = color

# Changing the brush size
def change_brush_size(dir):
    global brush_size, drawing
    if not drawing:
        if dir == 'greater':
            brush_size += brush_size_steps
        else:
            brush_size -= brush_size_steps

def invert_brush():
    global draw_color, drawing
    if not drawing:
        draw_color = ( 255 - np.array(draw_color) ).tolist()

# Clear canvas
def clear_canvas():
    global draw_surface, drawing
    if not drawing:
        draw_surface.fill((0, 0, 0))

# Save canvas
def save():
    global drawing
    if not drawing:
        pygame.image.save(draw_surface, "environment.png")

# loads environment.png into the drawing surface
def load():
    global draw_surface, drawing
    if not drawing:
        temp_surface = pygame.image.load(os.path.join(os.getcwd(), "environment.png"))
        draw_surface = pygame.transform.smoothscale(temp_surface, (canvas_width, canvas_height))
        environment.env_layer = np.clip(np.asmatrix(pygame.surfarray.array2d(draw_surface)), 0, 1, dtype=np.int32).astype(np.uint8)
        environment.update_edges()
        reset()

def screenshot(name="canvas.png"):
    global drawing
    if not drawing:
        pygame.image.save(canvas_surface, os.path.join(os.getcwd(), "Results", name))

def reset():
    global drawing, environment, agentlist, canvas_surface, counter, stats
    agent_spawn_amount = 5

    if not drawing:
        canvas_surface = pygame.Surface.copy(draw_surface)
        environment.reset_agents()
        environment.reset_food()
        possible_positions = np.array(np.where(environment.env_layer==False)).T

        if agent_spawn_amount > len(possible_positions):
            print(f"[ERROR] not enough space for {agent_spawn_amount} agents in environment")
            return
        
        for i in range(agent_spawn_amount):
            rand_index = np.random.randint(possible_positions.shape[0])
            rand_position = possible_positions[rand_index]
            possible_positions = np.delete(possible_positions, (rand_index), axis=0)
            whale = KillerWhale(rand_position, np.random.randint(0, 2, dialect_size, dtype=np.uint8), abs(np.random.normal(12, 3)))
            environment.add_agent(whale)
            draw_whale(canvas_surface, whale)
            screen.blit(canvas_surface, (0, button_panel_height))
            pygame.display.flip()
        stats = []
        counter = -1
        
def pause():
    global drawing, pause
    if not drawing:
        pause = not pause

def output_stats():
    global drawing, stats
    if not drawing:
        np.savez(os.path.join(os.getcwd(), "Results", "sim_data.npz"), data_stack=np.asarray(stats, dtype="object"))
        


# Button variables
button_width = 95
button_height = 50

# Button initialization
button_list = [
    ['Brush +', lambda: change_brush_size('greater')],
    ['Brush -', lambda: change_brush_size('smaller')],
    ['Brush !', lambda: invert_brush()],
    ['Clear', lambda: clear_canvas()],
    ['E Save', save],
    ['E Load', load],
    ['Capture', screenshot],
    ['Reset', reset],
    ['Pause', pause],
    ['S Save', output_stats],
]

# Button creation
for index, button_object in enumerate(button_list):
    if button_object[0] != 'Pause':
        Button(index * (button_width + 20) + 110, button_height/2, button_width, button_height, button_object[0], button_object[1])
    else:
        Button(index * (button_width + 20) + 110, button_height/2, button_width, button_height, button_object[0], button_object[1], hold_press=True)


# set drawing
drawing = False
mouse_down = False
pause = False

# Draw the side panel
screen.fill(side_panel_color, (canvas_width, 0, side_panel_width, window_height))

# Main loop (update loop)
running = True
counter = -1
stats = []
grouping_dialect = agentlist[0].grouping_dialect

while running:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Reset surfaces
    screen.fill(side_panel_color)
    button_surface.fill(side_panel_color)


    #===== Killer Whale Sim functions =====

    # get the drawn inaccessible area (e.g. land) into the environment layer
    environment.env_layer = np.clip(np.asmatrix(pygame.surfarray.array2d(draw_surface)), 0, 1, dtype=np.int32).astype(np.uint8)

    # update environment parameters
    environment.set_parameters(env_parameters)

    # update the environment with its agents
    # pr = cProfile.Profile()
    # pr.enable()
    if not pause:
        agent_list, a_positions = environment.update()
        food_matrix = environment.update_food()

        # Cluster agents based on dialect:
        if len(agent_list) > 0:
            # get dialects
            a_dialects = [a.dialect for a in agent_list]
            #print(a_dialects)

            # Compute Hamming distance matrix
            hamming_distances = pdist(a_dialects, metric='hamming')
            distance_matrix = squareform(hamming_distances)

            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=grouping_dialect, min_samples=2, metric='precomputed')
            clusters = dbscan.fit_predict(distance_matrix)

            # Output cluster assignments (-1 represents noise)
            #print(clusters)
            
            if counter % 10:
                data = [deepcopy(agent_list), deepcopy(clusters)]
                stats.append(data)

    # pr.disable()
    # pr.print_stats(sort='time')
    
    # get the invisible drawing surface (draw_surface) on the visible surface (canvas_surface)
    canvas_surface = pygame.Surface.copy(draw_surface)

    canvas_matrix = pygame.surfarray.array3d(canvas_surface)
    canvas_matrix[:,:,0] = np.clip((canvas_matrix[:,:,0] + food_matrix), 0, 255).astype(np.uint8)

    # Convert the modified array back to a Pygame surface
    canvas_surface = pygame.surfarray.make_surface(canvas_matrix)

    # draw the agents in the environment onto the visible surface (canvas_surface)
    draw_whales(canvas_surface, agent_list, clusters) # draw the agents on the surface
    counter += 1
    #===== END Killer Whale Sim functions =====


    # Draw the game title
    title_surface = font.render('KW Dialect Simulation', True, white)
    screen.blit(title_surface, (canvas_width + 20, 20))

    # Update and display slider values
    for i, (slider, config) in enumerate(zip(sliders, slider_configs)):
        if config['name'] == 'Mutation': text_boxes[i].setText(str(slider.getValue()/10000))
        elif config['name'] == 'Grouping Dialect': text_boxes[i].setText(str(slider.getValue()/10000))
        elif config['name'] == 'Transference': text_boxes[i].setText(str(slider.getValue()/10000))
        elif config['name'] == 'Max comm difference': text_boxes[i].setText(str(slider.getValue()/10000))
        else: text_boxes[i].setText(str(slider.getValue()))
        if config['name'] == 'Dialect size': dialect_size = slider.getValue()
        elif config['name'] == 'Mutation': mutation = slider.getValue() /10000
        elif config['name'] == 'Grouping Dialect': grouping_dialect = slider.getValue() /10000
        elif config['name'] == 'Transference': transference = slider.getValue() /10000
        elif config['name'] == 'Max comm difference': comm_dialect_difference = slider.getValue() /10000
        env_parameters[config['name']] = slider.getValue()
    
    if pygame.mouse.get_pressed()[0] and not mouse_down:
        mouse_down = True
        if canvas_surface.get_rect(topleft=(0, button_panel_height)).collidepoint(pygame.mouse.get_pos()):
            drawing = True
    elif not pygame.mouse.get_pressed()[0] and mouse_down:
        mouse_down = False
        drawing = False
    
    # Draw with mouse
    if drawing:
        mx, my = pygame.mouse.get_pos()

        # Calculate position on canvas
        dx = mx
        dy = my - button_panel_height

        pygame.draw.circle(
            draw_surface,   #surface
            draw_color,     #color
            [dx, dy],       #center coords
            brush_size,     #radius
        )
    
    # Draw the buttons
    for button in buttons:
        button.process()
    
    # Reference Dot
    pygame.draw.circle(
        button_surface,                 #surface
        draw_color,                     #color
        [50, button_panel_height/2],    #center coords
        brush_size,                     #radius
    )

    # Blit (copy) canvas to main screen
    screen.blit(canvas_surface, (0, button_panel_height))
    screen.blit(button_surface, (0, 0))

    # Update all widgets
    pygame_widgets.update(events)

    # Update the display
    pygame.display.flip()

    if counter % 100 == 0 and not pause:
        screenshot(f"Sim-at-step-{counter}.png")
print("finished")
