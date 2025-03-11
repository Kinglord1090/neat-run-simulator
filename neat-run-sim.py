import neat
import numpy
import os
import pygame
import random

MAX_WIDTH = 1280
MAX_HEIGHT = 720
WIN = pygame.display.set_mode((MAX_WIDTH, MAX_HEIGHT))
pygame.display.set_caption("AI Run")
GAME_SPEED = 10
GEN = 0
HIGHSCORE = 0

class Player(pygame.sprite.Sprite):
	def __init__(self):
		super().__init__()
		self.x_pos = 50
		self.y_pos = 400
		self.height = 100
		self.width = 50
		self.altitude = 0
		self.gravity = 1.3
		self.ducking = False

		self.standing_rect = pygame.Rect(self.x_pos, self.y_pos, self.width, self.height)
		self.ducking_rect = pygame.Rect(self.x_pos, self.y_pos + self.height // 2, self.width, self.height // 2)

		self.rect = self.standing_rect

	def jump(self):
		if self.rect.y >= 400 and not self.ducking:
			self.altitude = -20

	def start_duck(self):
		if not self.ducking:
			self.ducking = True
			self.rect = self.ducking_rect

	def stop_duck(self):
		if self.ducking:
			self.ducking = False
			self.rect = self.standing_rect

	def apply_gravity(self):
		if not self.ducking:
			self.altitude += self.gravity
			self.rect.y = min(self.rect.y + self.altitude, 400)
			if self.rect.y >= 400:
				self.altitude = 0

class Obstacle(pygame.sprite.Sprite):
	def __init__(self, type=0):
		super().__init__()
		self.height = 100
		self.width = 100
		self.type = type
		self.x_pos = 1300
		self.y_pos = 340 if type else 400
		self.rect = pygame.Rect(self.x_pos, self.y_pos, self.width, self.height)
	
	def update(self):
		self.rect.x -= GAME_SPEED

def draw_text(text, font, color, x, y):
	text_surface = font.render(text, True, color)
	WIN.blit(text_surface, (x, y))

def eval_genomes(genomes, config):
	global WIN, GEN, HIGHSCORE
	GEN += 1
	SCORE = 0
	OBSTACLE_SPAWN_TIMER = 0

	pygame.init()
	text = pygame.font.SysFont('comicsans', 50)
	clock = pygame.time.Clock()

	players = []
	nets = []
	ge = []
	obstacle_group = pygame.sprite.Group()

	for genome_id, genome in genomes:
		genome.fitness = 0
		net = neat.nn.FeedForwardNetwork.create(genome, config)
		nets.append(net)
		players.append(Player())
		ge.append(genome)

	run = True
	while run and len(players) > 0:
		SCORE += 0.1

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				pygame.quit()
				quit()

		OBSTACLE_SPAWN_TIMER += clock.get_time()
		if OBSTACLE_SPAWN_TIMER >= random.randint(1000,5000):
			OBSTACLE_SPAWN_TIMER = 0
			obstacle = Obstacle(random.choices([0, 1], weights=[0.8, 0.2], k=1)[0])
			obstacle_group.add(obstacle)
		obstacle_group.update()

		for i, player in enumerate(players):
			nearest_obstacle = min(
				(obstacle for obstacle in obstacle_group if obstacle.rect.x > player.rect.x),
				default=None,
				key=lambda obs: obs.rect.x - player.rect.x
			)

			obstacle_distance = (nearest_obstacle.rect.x - player.rect.x) if nearest_obstacle else MAX_WIDTH
			obstacle_type = nearest_obstacle.type if nearest_obstacle else 0

			inputs = (
				player.altitude,
				player.ducking,
				obstacle_distance,
				obstacle_type
			)

			output = nets[i].activate(inputs)
			action = numpy.argmax(output)

			if action == 0:
				player.jump()
			elif action == 1:
				player.start_duck()
			else:
				player.stop_duck()

			ge[i].fitness += 0.1

		for i in reversed(range(len(players))):
			if pygame.sprite.spritecollideany(players[i], obstacle_group):
				players.pop(i)
				nets.pop(i)
				ge.pop(i)

		WIN.fill((31, 31, 31))
		for i, player in enumerate(players):
			player.apply_gravity()
			pygame.draw.rect(WIN, (0, 200, 0), player.rect)

		for obstacle in obstacle_group:
			if obstacle.type:
				pygame.draw.rect(WIN, (200, 0, 0), obstacle.rect)
			else:
				pygame.draw.rect(WIN, (0, 0, 200), obstacle.rect)

		draw_text(f"Gen: {GEN}", text, (255, 255, 255), 10, 10)
		draw_text(f"Score: {int(SCORE)}", text, (255, 255, 255), 10, 60)

		HIGHSCORE = max(SCORE, HIGHSCORE)
		draw_text(f"Highscore: {int(HIGHSCORE)}", text, (255, 255, 255), 10, 620)

		pygame.display.update()
		clock.tick(60)

def run(config_file):
	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
	p = neat.Population(config)
	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)

	winner = p.run(eval_genomes, 100)
	print(f"\nBest genome:\n{winner}")

if __name__ == "__main__":
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, 'config-feedforward')
	run(config_path)
