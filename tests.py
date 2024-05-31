import unittest
from classes import *

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_paths(self):
        paths = PathsInputOutput("lol.png", "kek.png")
        self.assertEqual(paths.get_paths()[0], "lol.png")
        self.assertEqual(paths.get_paths()[1], "kek.png")

    def test_field_create_net_same_sizes(self, n = 5, m = 5):
        paths = PathsInputOutput("backgrounds/simple_field.jpg", "backgrounds/simple_field_output.jpg")
        field = Field(paths, n, m)
        rectangles_coordinates, rect_width, rect_height = field.create_net()
        self.assertTrue(len(rectangles_coordinates) == n * m)
        self.assertTrue(rect_width >= n)
        self.assertTrue(rect_height >= m)

    def test_field_create_net_different_sizes_of_field(self, n = 5, m = 3):
        paths = PathsInputOutput("backgrounds/simple_field.jpg", "backgrounds/simple_field_output.jpg")
        field = Field(paths, n, m)
        rectangles_coordinates, rect_width, rect_height = field.create_net()
        self.assertTrue(len(rectangles_coordinates) == n * m)
        self.assertTrue(rect_width >= n)
        self.assertTrue(rect_height >= m)
    
    def test_calculating_square_centers(self, n = 5, m = 3):
        paths = PathsInputOutput("backgrounds/simple_field.jpg", "backgrounds/simple_field_output.jpg")
        field = Field(paths, n, m)
        rectangles_coordinates = field.create_net()[0]
        football_field = FootballField(field)
        football_field.calculate_squares_centers()
        self.assertTrue(len(football_field.square_centers) == len(rectangles_coordinates))

    def test_adding_players(self, n = 10, m = 10):
        paths = PathsInputOutput("backgrounds/simple_field.jpg", "backgrounds/simple_field_output.jpg")
        field = Field(paths, n, m)
        rectangles_coordinates = field.create_net()[0]
        football_field = FootballField(field)
        team1 = [40, 21, 41, 61, 81, 52, 43, 63, 24, 84, 54]
        team2 = [49, 28, 48, 68, 88, 57, 46, 66, 25, 85, 55]
        football_field.add_two_teams(team1, team2)
        self.assertTrue(len(football_field.players) == 22)
    

unittest.main()