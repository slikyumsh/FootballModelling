import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import numpy as np


class PathsInputOutput:
    def __init__ (self, background_path, output_path):
        self.background_image_path = background_path
        self.output_path = output_path
    
    def get_paths(self):
        return (self.background_image_path, self.output_path)

    def show_paths(self):
        print("Input image: ", self.background_image_path)
        print("Output image: ", self.output_path)
        

class Field:

    def __init__ (self, pathsInputOutput, num_cols, num_rows):
        self.paths = pathsInputOutput
        self.num_cols = num_cols
        self.num_rows = num_rows
    
    def show_input_field(self):
        cv_image = cv2.imread(self.paths.background_image_path)
        if cv_image is not None:
            plt.imshow(cv_image)
            plt.show()
        else:
            print("Failed to open the input image")
    
    def show_output_field(self, color = 'red'):
        self.create_net(color)
        cv_image = cv2.imread(self.paths.output_path)
        if cv_image is not None:
            plt.imshow(cv_image)
            plt.show()
        else:
            print("Failed to open the output image")

    def create_net(self, color = 'red'):
        image = Image.open(self.paths.background_image_path)
        draw = ImageDraw.Draw(image)

        width, height = image.size

        rect_width = width // self.num_cols
        rect_height = height // self.num_rows

        cv_image = np.array(image)
        rectangles_coordinates = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                left = j * rect_width
                upper = i * rect_height
                right = left + rect_width
                lower = upper + rect_height
                draw.rectangle([left, upper, right, lower], outline=color)
                rectangles_coordinates.append([(left, upper), (right, upper), (right, lower), (left, lower)])

        image.save(self.paths.output_path)

        return rectangles_coordinates, rect_width, rect_height

    def show_paths(self):
        self.paths.show_paths()

class Position():
    def __init__(self, pos_x, pos_y):
        self.pos_x = pos_x
        self.pos_y = pos_y

class Player():
    def __init__(self, id, team_id):
        self.id = id
        self.team_id = team_id

class PlayerOnField:
    def __init__(self, player, position):
        self.palyer = player,
        self.position = position


class FootballField:
    def __init__ (self, field):
        self.field = field
        self.players = []
        self.square_centers = []
        self.output_pic_path = 'with_players.jpg'

    def calculate_squares_centers (self):
        square_coordinates = self.field.create_net()[0]
        for square_coordinate in square_coordinates:
            x = (square_coordinate[0][0] + square_coordinate[1][0] + square_coordinate[2][0] + square_coordinate[3][0]) // 4
            y = (square_coordinate[0][1] + square_coordinate[1][1] + square_coordinate[2][1] + square_coordinate[3][1]) // 4
            self.square_centers.append((x, y))

    def generate_player_id(self, player_id, team_id):
        return player_id + team_id * 100

    def add_players(self, player_square_center, team):
 
        if len(player_square_center) != 11:
           raise Exception("Invalid size of players array: x != 11")

        cnt = 0
        for player_number in player_square_center:
            cnt += 1
            player = Player(self.generate_player_id(cnt, team), team)
            position = Position(self.square_centers[player_number][0], self.square_centers[player_number][1])
            playerOnField = PlayerOnField(player, position)
            self.players.append(playerOnField)

    def add_two_teams(self, centers1, centers2):
        self.calculate_squares_centers()
        self.add_players(centers1, 1)
        self.add_players(centers2, 2)

    def show_pic_with_players(self):
        image = Image.open( self.field.paths.output_path)
        draw = ImageDraw.Draw(image)

        team_colors = {1: 'blue', 2: 'red'}

        for player in self.players:
            x = player.position.pos_x
            y = player.position.pos_y
            team_id = player.palyer[0].team_id  
            color = team_colors.get(team_id, 'gray')

            draw.ellipse([(x - 15, y - 15), (x + 15, y + 15)], fill=color, outline=None)

        new_pic_path = self.output_pic_path
        image.save(new_pic_path)

        plt.imshow(image)
        plt.show()



class Model:
    def __init__ (self, footballField):
        self.footballField = footballField

    def run(self):
        self.footballField.show_pic_with_players()



        
