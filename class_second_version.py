import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import json
import multiprocessing as mp
import random
import numpy as np
import copy
import os
import csv
from datetime import datetime

class PathsInputOutput:
    def __init__(self, background_path, output_path):
        self.background_image_path = background_path
        self.output_path = output_path

    def get_paths(self):
        return (self.background_image_path, self.output_path)

    def show_paths(self):
        print("Input image: ", self.background_image_path)
        print("Output image: ", self.output_path)
    
    def to_dict(self):
        return {
            'background_image_path': self.background_image_path,
            'output_path': self.output_path
        }

class Position:
    def __init__(self, pos_x, pos_y):
        self.pos_x = pos_x
        self.pos_y = pos_y
    
    def to_dict(self):
        return {
            'pos_x': self.pos_x,
            'pos_y': self.pos_y
        }

class Player:
    def __init__(self, team_id, v, is_gk):
        self.id = 0
        self.team_id = team_id
        self.v = v
        self.gk = is_gk
    
    def to_dict(self):
        return {
            'id': self.id,
            'team_id': self.team_id,
            'v': self.v,
            'gk': self.gk
        }

class PlayerOnField:
    def __init__(self, player, position):
        self.player = player
        self.position = position
    
    def to_dict(self):
        return {
            'player': self.player.to_dict(),
            'position': self.position.to_dict()
        }

class Field:
    def __init__(self, pathsInputOutput):
        self.paths = pathsInputOutput
    
    def show_input_field(self):
        cv_image = cv2.imread(self.paths.background_image_path)
        if cv_image is not None:
            plt.imshow(cv_image)
            plt.show()
        else:
            print("Failed to open the input image")
    
    def to_dict(self):
        return {
            'paths': self.paths.to_dict()
        }



class FootballField:
    def __init__ (self, field):
        self.field = field
        self.players = []

   
    def generate_player_id(self, player_id, team_id):
        return player_id + team_id * 100
    
    def add_players(self, players):
        if len(players) != 11:
           raise Exception("Invalid size of players array: x != 11")

        is_gk_counter = 0
        for player in players:
            if player.player.gk == 1:
                is_gk_counter += 1

        if is_gk_counter != 1:
            raise Exception("Gk error")
        
        cnt = 0
        for player in players:
            cnt += 1
            player.player.id = self.generate_player_id(cnt, player.player.team_id)
            self.players.append(player)


    def add_two_teams(self, players1, players2):
        self.add_players(players1)
        self.add_players(players2)

    
    def show_pic_with_players(self):
        image = Image.open( self.field.paths.background_image_path)
        draw = ImageDraw.Draw(image)

        team_colors = {1: 'blue', 2: 'red'}

        for player in self.players:
            x = player.position.pos_x
            y = player.position.pos_y
            team_id = player.player.team_id  
            color = team_colors.get(team_id, 'gray')

            draw.ellipse([(x - 15, y - 15), (x + 15, y + 15)], fill=color, outline=None)

        new_pic_path = self.field.paths.output_path
        image.save(new_pic_path)

        plt.imshow(image)
        plt.show()


    def to_dict(self):
        return {
            'field': self.field.to_dict(),
            'players': [player.to_dict() for player in self.players]
        }



class Model:

    def __init__ (self, field, player, epsilon, beta, max_n, max_xg):
        self.field = field
        self.current_player = player
        self.epsilon = epsilon
        self.beta = beta
        self.max_n = max_n
        self.dynamic_vector = self.init_dynamic_vector()
        self.xg_early_stopping = max_xg
        self.active_players = self.choose_players()
        self.current_xg = self.init_xg()
        self.unique_players = []
        self.unique_players.append(self.current_player.player.id)
        self.all_positions_by_iteration = []
        self.all_positions_by_iteration.append(self.init_start_positions())
        self.cnt_passes = 0

    def calculate_xg(self, player_coords, post_center):
        x1 = player_coords[0] / 1500
        x2 = post_center[0] / 1500
        y1 = player_coords[1] / 1000
        y2 = post_center[1] / 1000
        if y1 - y2 == 0:
            return np.exp(- 3. * ((x1 - x2)**2 + (y1 - y2)**2)) * np.sin(np.arctan(abs((x1 - x2)/0.0001)))
        return np.exp(- 3. * ((x1 - x2)**2 + (y1 - y2)**2)) * np.sin(np.arctan(abs((x1 - x2)/(y1 - y2))))
    
    def init_xg (self):
        return self.calculate_xg((self.current_player.position.pos_x, self.current_player.position.pos_y),
                                 (1500, 500))
    
    def init_start_positions (self):
        return copy.deepcopy(self.field.players)
    
    def choose_players(self):
        result = []
        all_players = self.field.players
        for player in all_players:
            if player.player.team_id == 1 and player.player.gk == False:
                result.append(player)
        return result
    
    def init_dynamic_vector(self):
        result = []
        for i in range(0, 10):
            result.append([0., 0.])  # Use list instead of tuple
        return result
    
    def calculate_dynamic_vector(self):
        all_players = self.field.players
        players = []
        for player in all_players:
            if player.player.team_id == 1 and player.player.gk == False:
                players.append(player)

        cnt = 0
        for player in players:
            shift_x = np.random.uniform(-player.player.v / 1.5, player.player.v)
            self.dynamic_vector[cnt][0] = shift_x
            diff = np.sqrt(player.player.v ** 2 - shift_x ** 2)
            shift_y = np.random.uniform(-diff, diff)
            self.dynamic_vector[cnt][1] = shift_y
            cnt += 1  # Ensure that cnt is incremented within the loop


    def players_moving(self):
        players = self.active_players
        cnt = 0
        for player in players:
            player.position.pos_x += self.dynamic_vector[cnt][0]
            player.position.pos_y += self.dynamic_vector[cnt][1]
            cnt += 1
            player.position.pos_x = max(player.position.pos_x, 0)
            player.position.pos_y = max(player.position.pos_y, 0)
            player.position.pos_x = min(player.position.pos_x, 1500)
            player.position.pos_y = min(player.position.pos_y, 1000)


    def find_line_equation(self, x1, y1, x2, y2):
        if x2 - x1 == 0:
            m = (y2 - y1) / 0.0001
        else:
            m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m, b
    
    def distance_from_point_to_line(self, x1, y1, m, b):
        distance = np.abs(y1 - m * x1 - b) / np.sqrt(m**2 + 1)
        return distance

    def choose_players_for_passing (self):
        
        result = []

        for player in self.active_players:
            if player.player.id != self.current_player.player.id:
                x1 = player.position.pos_x
                y1 = player.position.pos_y
                x2 = self.current_player.position.pos_x
                y2 = self.current_player.position.pos_y
                m, b = self.find_line_equation(x1, y1, x2, y2)
                intercepted = False
                for opponent_player in self.field.players:
                    if opponent_player not in self.active_players:
                        opponent_player_x = opponent_player.position.pos_x
                        opponent_player_y = opponent_player.position.pos_y
                        distance = self.distance_from_point_to_line(opponent_player_x, opponent_player_y, m, b)
                        if (distance < self.epsilon):
                            intercepted = True
                
                if intercepted == False:
                    result.append(player)

        return result
    
    
    def mutation(self):
        partners = self.choose_players_for_passing()
        l = len(partners)
        print("Cur player: ", self.current_player.player.id)
        print("Partners number: " , l)
        for _ in range(0, 2 * l):
            id = np.random.randint(0, l)
            chosen_player = partners[id]
            x = chosen_player.position.pos_x
            y = chosen_player.position.pos_y
            xg = self.calculate_xg((x, y), (1500, 500))
            if xg > self.current_xg:
                self.current_player = chosen_player
                self.current_xg = xg
                self.cnt_passes += 1
                print("Accepted full")
                return
            else:
                random_number = np.random.uniform(0., 1.)
                if np.exp((xg - self.current_xg) * self.beta) > random_number:
                    self.current_player = chosen_player
                    self.current_xg = xg
                    print("Accepted part")
                    self.cnt_passes += 1
                    return

    def attack (self):
        for player in self.active_players:
            player.position.pos_x += 200

    def metropolis(self):
        self.unique_players.append(self.current_player.player.id)
        self.attack()
        for i in range(0, self.max_n):
            self.unique_players.append(self.current_player.player.id)
            self.mutation()
            if self.current_xg >= self.xg_early_stopping:
                print("Ended : ", i)
                break
            self.calculate_dynamic_vector()
            self.players_moving()
            self.all_positions_by_iteration.append(copy.deepcopy(self.field.players))
        self.unique_players.append(self.current_player.player.id)
        
    def show_and_save_positions_gif(self):
        if not os.path.exists('dynamic_photos'):
            os.makedirs('dynamic_photos')
        
        images_for_gif = []
    
        for idx, players_state in enumerate(self.all_positions_by_iteration):
            image = Image.open(self.field.field.paths.background_image_path)  # Убедитесь, что путь указан правильно
            draw = ImageDraw.Draw(image)
            team_colors = {1: 'blue', 2: 'red'}
            font = ImageFont.truetype("arial.ttf", 15)  # Укажите путь к шрифту, если файл не в стандартной папке

            # Идентификация игрока с мячом
            ball_holder_id = self.current_player.player.id if self.current_player else None

            for player in players_state:
                x = player.position.pos_x
                y = player.position.pos_y
                team_id = player.player.team_id  
                player_id = player.player.id
                color = 'yellow' if player_id == ball_holder_id else team_colors.get(team_id, 'gray')
                draw.ellipse([(x - 15, y - 15), (x + 15, y + 15)], fill=color, outline='black')
                draw.text((x - 10, y - 10), str(player_id), fill='white', font=font)
    
            frame_filename = f'dynamic_photos/frame_{idx}.png'
            image.save(frame_filename)
            images_for_gif.append(image)
        
        gif_path = 'dynamic_photos/players_animation.gif'
        images_for_gif[0].save(gif_path, save_all=True, append_images=images_for_gif[1:], optimize=False, duration=2000, loop=0)
        print(f"GIF saved at {gif_path}")

import copy
import json
import multiprocessing as mp
import os
from datetime import datetime
import csv

class Experiment:
    def __init__(self, models, experiment_id):
        """
        Initializes the Experiment with a list of models and an experiment ID.
        
        Args:
        models (List[Model]): The list of models to be used in the experiment.
        experiment_id (str): The unique identifier for the experiment.
        """
        self.models = models
        self.experiment_id = experiment_id

    def create_model_copies(self, n):
        """
        Creates n independent copies of each model in self.models.
        
        Args:
        n (int): The number of copies to create for each model.
        
        Returns:
        None: Updates self.models with the original models and their copies.
        """
        all_models = []
        for model in self.models:
            for _ in range(n):
                copied_model = Model(
                    copy.deepcopy(model.field),
                    copy.deepcopy(model.current_player),
                    model.epsilon,
                    model.beta,
                    model.max_n,
                    model.xg_early_stopping
                )
                all_models.append(copied_model)
        self.models = all_models

    def create_param_variations(self, param_name, step, num_steps):
        """
        Creates copies of the current models with variations in the specified parameter.
        
        Args:
        param_name (str): The name of the parameter to vary.
        step (float): The step size for each variation.
        num_steps (int): The number of variations to create.
        
        Returns:
        None: Updates self.models with the original models and their variations.
        """
        all_models = []
        for model in self.models:
            for i in range(num_steps):
                new_param_value = getattr(model, param_name) + i * step
                copied_model = copy.deepcopy(model)
                setattr(copied_model, param_name, new_param_value)
                all_models.append(copied_model)
        self.models = all_models

    def run_simulation(self, model):
        """
        Runs the simulation for a given model using the metropolis algorithm.
        
        Args:
        model (Model): The model to run the simulation on.
        
        Returns:
        dict: A dictionary containing the results of the simulation.
        """
        model.metropolis()
        return {
            'input_information': {
                'field': model.field.to_dict(),
                'current_player': model.current_player.to_dict(),
                'epsilon': model.epsilon,
                'beta': model.beta,
                'max_n': model.max_n,
                'xg_early_stopping': model.xg_early_stopping
            },
            'unique_players': len(set(model.unique_players)),
            'passes': model.cnt_passes,
            'iterations': len(model.all_positions_by_iteration),
            'final_xg': model.current_xg
        }

    def run(self):
        """
        Runs the experiment by executing simulations for all models in parallel.
        
        Returns:
        None: Saves the results to a CSV file.
        """
        with mp.Pool(mp.cpu_count() // 2) as pool:
            results = pool.map(self.run_simulation, self.models)
        
        self.save_results_to_csv(results)
    
    def save_results_to_csv(self, results):
        """
        Saves the results of the experiment to a CSV file.
        
        Args:
        results (List[dict]): The list of dictionaries containing the results of the simulations.
        
        Returns:
        None: Writes the results to a CSV file in the experiment_results directory.
        """
        now = datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
        directory = 'experiment_results'
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.join(directory, f'football_{self.experiment_id}_{now}.csv')
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['input_information', 'unique_players', 'passes', 'iterations', 'final_xg']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                result['input_information'] = json.dumps(result['input_information'])
                writer.writerow(result)

        print(f"Results saved to {filename}")

