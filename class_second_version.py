import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import json
import multiprocessing as mp
import random
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

    def to_dict(self):
        return {
            'field': self.field.to_dict(),
            'players': [player.to_dict() for player in self.players]
        }


import uuid
import matplotlib.colors as mcolors
class Model:
    def __init__(self, field, player, epsilon, beta, max_n, max_xg):
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
        self.ball_holders = []  # Новый атрибут для хранения игроков с мячом

    def calculate_xg(self, player_coords, post_center):
        x1 = player_coords[0] / 1500
        x2 = post_center[0] / 1500
        y1 = player_coords[1] / 1000
        y2 = post_center[1] / 1000
        if y1 - y2 == 0:
            return np.exp(- 3. * ((x1 - x2)**2 + (y1 - y2)**2)) * np.sin(np.arctan(abs((x1 - x2)/0.0001)))
        return np.exp(- 3. * ((x1 - x2)**2 + (y1 - y2)**2)) * np.sin(np.arctan(abs((x1 - x2)/(y1 - y2))))
    
    def init_xg(self):
        return self.calculate_xg((self.current_player.position.pos_x, self.current_player.position.pos_y), (1500, 500))

    def init_start_positions(self):
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
            result.append([0., 0.])
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
            cnt += 1

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

    def is_offside(self, receiving_player, players, current_player):
        if receiving_player.player.team_id != 1:
            return False
        x_positions = [player.position.pos_x for player in players if player.player.team_id == 2 and not player.player.gk]
        second_last_defender_x = sorted(x_positions)[-1]

        if receiving_player.position.pos_x > second_last_defender_x and receiving_player.position.pos_x > current_player.position.pos_x:
            return True
        return False



    def choose_players_for_passing(self):
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
                        if distance < self.epsilon:
                            intercepted = True
                
                if not intercepted and not self.is_offside(player, self.field.players, self.current_player):
                    result.append(player)

        return result

    def mutation(self):
        partners = self.choose_players_for_passing()
        l = len(partners)
        print("Cur player: ", self.current_player.player.id)
        print("Partners number: " , l)
        accepted_combination = False  # Флаг для отслеживания принятой комбинации
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
                accepted_combination = True  # Обновляем флаг
                print("Accepted full")
                return
            else:
                random_number = np.random.uniform(0., 1.)
                if np.exp((xg - self.current_xg) * self.beta) > random_number:
                    self.current_player = chosen_player
                    self.current_xg = xg
                    self.cnt_passes += 1
                    accepted_combination = True  # Обновляем флаг
                    print("Accepted part")
                    return
        
        if not accepted_combination:
            print("No accepted combination")

    def attack (self):
        for player in self.active_players:
            player.position.pos_x += 200


    def defense_pressing(self):
        for defender in self.field.players:
            if defender.player.team_id == 2 and defender.player.gk == False:
                min_distance = float('inf')
                closest_attacker = None
                for attacker in self.active_players:
                    distance = np.sqrt((defender.position.pos_x - attacker.position.pos_x) ** 2 + 
                                       (defender.position.pos_y - attacker.position.pos_y) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_attacker = attacker

                if closest_attacker:
                    shift_x = np.random.uniform(0, defender.player.v)
                    shift_y = np.random.uniform(0, defender.player.v)
                    defender.position.pos_x += shift_x * np.sign(closest_attacker.position.pos_x - defender.position.pos_x)
                    defender.position.pos_y += shift_y * np.sign(closest_attacker.position.pos_y - defender.position.pos_y)
                    defender.position.pos_x = max(defender.position.pos_x, 0)
                    defender.position.pos_y = max(defender.position.pos_y, 0)
                    defender.position.pos_x = min(defender.position.pos_x, 1500)
                    defender.position.pos_y = min(defender.position.pos_y, 1000)

    def metropolis(self):
        self.unique_players.append(self.current_player.player.id)
        self.ball_holders.append(self.current_player.player.id)  # Сохраняем текущего игрока с мячом
        self.attack()
        for i in range(0, self.max_n):
            self.unique_players.append(self.current_player.player.id)
            self.ball_holders.append(self.current_player.player.id)  # Сохраняем текущего игрока с мячом
            self.mutation()
            self.defense_pressing()
            if self.current_xg >= self.xg_early_stopping:
                print("Ended : ", i)
                break
            self.calculate_dynamic_vector()
            self.players_moving()
            self.all_positions_by_iteration.append(copy.deepcopy(self.field))
            print(f"Iteration {i}: Positions recorded")
        self.unique_players.append(self.current_player.player.id)
        self.ball_holders.append(self.current_player.player.id) 



    def create_gif(self, frame_duration=500):
        frames = []
        font = ImageFont.truetype("arial.ttf", 15)  # Используем меньший размер шрифта

        # Загружаем фоновое изображение
        background_path = self.field.field.paths.background_image_path
        if not os.path.exists(background_path):
            print(f"Фоновое изображение не найдено по пути {background_path}")
            return
        
        background_image = Image.open(background_path)

        for i, football_field in enumerate(self.all_positions_by_iteration):
            players_state = football_field.players if isinstance(football_field, FootballField) else football_field
            frame = background_image.copy()  # Создаем копию фонового изображения для каждого кадра
            draw = ImageDraw.Draw(frame)
            ball_holder_id = self.ball_holders[i] if i < len(self.ball_holders) else None
            current_player = next((player for player in players_state if player.player.id == ball_holder_id), None)

            for player in players_state:
                x = player.position.pos_x
                y = player.position.pos_y
                team_id = player.player.team_id
                player_id = player.player.id

                if team_id == 2:
                    color = 'red'
                elif player_id == ball_holder_id:
                    color = 'black'
                elif self.is_offside(player, players_state, current_player):
                    color = 'green'
                else:
                    color = 'blue'

                radius = 17  # Увеличиваем радиус для лучшей видимости
                draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=color, outline='black')

                text = str(player_id % 100)
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = x - text_width / 2
                text_y = y - text_height / 2
                draw.text((text_x, text_y), text, fill='white', font=font)

            frames.append(frame)

        output_dir = "dynamics"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        gif_filename = f"{output_dir}/player_positions_{uuid.uuid4()}.gif"
        frames[0].save(gif_filename, save_all=True, append_images=frames[1:], 
                    duration=frame_duration, loop=0)
        print(f"GIF сохранен по пути {gif_filename}")



    def create_heatmap(self, radius=20, output_path="heatmap_with_legend.png"):
        # Убедитесь, что фон загружен
        background_path = self.field.field.paths.background_image_path
        if not os.path.exists(background_path):
            print(f"Фоновое изображение не найдено по пути {background_path}")
            return
        
        # Загрузите фоновое изображение
        background_image = Image.open(background_path)
        heatmap = np.zeros((background_image.height, background_image.width))
        
        # Соберите данные перемещений атакующих игроков
        for football_field in self.all_positions_by_iteration:
            players_state = football_field.players if isinstance(football_field, FootballField) else football_field
            for player in players_state:
                if player.player.team_id == 1:  # Предполагаем, что команда 1 атакующая
                    x = int(player.position.pos_x)
                    y = int(player.position.pos_y)
                    cv2.circle(heatmap, (x, y), radius, 1, thickness=-1)
        
        # Примените размытие к тепловой карте
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=radius, sigmaY=radius)
        
        # Нормализуйте тепловую карту
        heatmap = np.clip(heatmap / heatmap.max(), 0, 1)
        
        # Преобразуйте тепловую карту в цветовую карту
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        
        # Преобразуйте тепловую карту в формат PIL и наложите на фоновое изображение
        heatmap_image = Image.fromarray(heatmap_color)
        combined_image = Image.alpha_composite(background_image.convert("RGBA"), heatmap_image.convert("RGBA"))
        
        # Сохраните итоговое изображение с тепловой картой
        combined_image_path = "combined_" + output_path
        combined_image.save(combined_image_path)
        





import copy
import json
import multiprocessing as mp
import os
from datetime import datetime
import csv
class Experiment:
    def __init__(self, models, experiment_id):
        self.models = models
        self.experiment_id = experiment_id

    def create_model_copies(self, n):
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
        all_models = []
        for model in self.models:
            for i in range(num_steps):
                new_param_value = getattr(model, param_name) + i * step
                copied_model = copy.deepcopy(model)
                setattr(copied_model, param_name, new_param_value)
                all_models.append(copied_model)
        self.models = all_models

    def run_simulation(self, model):
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
        with mp.Pool(mp.cpu_count() // 2) as pool:
            results = pool.map(self.run_simulation, self.models)
        self.save_results_to_csv(results)
    
    def save_results_to_csv(self, results):
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

    