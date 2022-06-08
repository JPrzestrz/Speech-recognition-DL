import arcade 
import random

# Screen
SCREEN_HEIGHT = 700
SCREEN_WIDTH = 600
# Coin 
COIN_COUNT = 2
COIN_SCALE = 0.075
# Start Menu 
BUTTON_SCALE = 0.5
# Character, directions, constants for storing facing 
CHARACTER_SCALE = 1.25
RIGHT_DIR = 1
LEFT_DIR = 0
# How fast to move, and how fast to run the animation
MOVEMENT_SPEED = 5
UPDATES_PER_FRAME = 5
# Obstacle const 
OBST_SCALE = 0.2
OBST_COUNT = 2
OBST_SPEED = 2

title = 'GameTest'
exit = False

def load_texture_pair(filename):
    """
    Load a texture pair, with the second being a mirror image for direction distinguishing.
    """
    return [
        arcade.load_texture(filename),
        arcade.load_texture(filename, flipped_horizontally=True)
    ]

class PlayerCharacter(arcade.Sprite):
    def __init__(self):

        # Set up parent class
        super().__init__()

        # Default to face-right
        self.character_face_direction = RIGHT_DIR

        # Used for flipping between image sequences
        self.cur_texture = 0
        self.scale = CHARACTER_SCALE

        # Adjust the collision box. Default includes too much empty space
        # side-to-side. Box is centered at sprite center, (0, 0)
        # self.points = [[-22, -64], [22, -64], [22, 28], [-22, 28]]

        # --- Load Textures ---
        # Image path
        player_path = "images/player"

        # Load textures for idle standing
        self.idle_texture_pair = load_texture_pair(f"{player_path}0.png")

        # Load textures for walking
        self.walk_textures = []
        for i in range(6):
            texture = load_texture_pair(f"{player_path}{i}.png")
            self.walk_textures.append(texture)

    def update_animation(self, delta_time: float = 1 / 60):

        # Figure out if we need to flip face left or right
        if self.change_x < 0 and self.character_face_direction == RIGHT_DIR:
            self.character_face_direction = LEFT_DIR
        elif self.change_x > 0 and self.character_face_direction == LEFT_DIR:
            self.character_face_direction = RIGHT_DIR

        # Idle animation
        if self.change_x == 0 and self.change_y == 0:
            self.texture = self.idle_texture_pair[self.character_face_direction]
            return

        # Walking animation
        self.cur_texture += 1
        if self.cur_texture > 4 * UPDATES_PER_FRAME:
            self.cur_texture = 0
        frame = self.cur_texture // UPDATES_PER_FRAME
        direction = self.character_face_direction
        self.texture = self.walk_textures[frame+1][direction]

class MyGameWindow(arcade.Window):
    def __init__(self,width,height,title):
        super().__init__(SCREEN_WIDTH,SCREEN_HEIGHT,title) 
        self.set_location(400,50)
        arcade.set_background_color(arcade.color.RED_DEVIL)

    def setup(self):
        '''
        Initial settings of game window
        '''
        # Generate boxes 
        self.background = arcade.load_texture('images/menu_bg.png')
        # Starting Menu Graphics
        self.start_button = arcade.Sprite("images/start_button.png", scale=COIN_SCALE)
        self.start_button.center_x = SCREEN_WIDTH // 2
        self.start_button.center_y = SCREEN_HEIGHT // 2
        self.start_button.scale = BUTTON_SCALE
        # Lists setup
        self.player_list = arcade.SpriteList()
        self.obst_list = arcade.SpriteList()
        self.coin_list = arcade.SpriteList()
        # Player setup 
        self.player = PlayerCharacter()
        self.player.center_x = SCREEN_WIDTH // 2
        self.player.center_y = 100
        self.player.scale = CHARACTER_SCALE
        self.player_list.append(self.player)
        # Game values 
        self.score = 0
        self.timer = 0
        # dividing game into scenes to make it more attractive
        # there will be main menu - 0, game - 1, pause - 2
        self.scene = 0

        # obstacles spawning 
        for i in range(OBST_COUNT):
            obst = arcade.Sprite('images/obstacle.png', scale=OBST_SCALE)
            obst.center_x = random.randrange(50,SCREEN_WIDTH-50)
            obst.center_y = i*250 + SCREEN_HEIGHT
            self.obst_list.append(obst)

        # coins spawning
        for i in range(COIN_COUNT):
            coin = arcade.Sprite("images/coin.png", scale=COIN_SCALE)
            # Spawning coins in distance of 100px from player 
            coin.center_x = random.randrange(SCREEN_WIDTH)
            if coin.center_x - self.player.center_x < 100 and coin.center_x - self.player.center_x > 0:
                coin.center_x += 100
            elif coin.center_x - self.player.center_x > -100 and coin.center_x - self.player.center_x < 0:
                coin.center_x -= 100
            coin.center_y = 100
            coin.center_x = coin.center_x % SCREEN_WIDTH
            self.coin_list.append(coin)

    def on_draw(self):
        """
        Render the screen.
        """
        # TODO add ending screen and maybe reset of a game 
        # This command has to happen before we start drawing
        self.clear()
        # Draw the background texture
        arcade.draw_lrwh_rectangle_textured(0, 0,SCREEN_WIDTH, SCREEN_HEIGHT,self.background)
        if self.scene == 0:
            # Draw menu etc
            #print("Here is menu")
            self.start_button.draw()

        if self.scene == 1:
            # Draw all the sprites.
            self.player_list.draw()
            self.coin_list.draw()
            self.obst_list.draw()

            # Draw score on the screen
            score_text = f"Score: {self.score}"
            arcade.draw_text(score_text,
                            start_x=10,
                            start_y=10,
                            color=arcade.csscolor.WHITE,
                            font_size=18)

    def on_key_press(self, key, modifiers):
        """
        Called whenever a key is pressed.
        """
        if self.scene == 1:
            if key == arcade.key.LEFT:
                self.player.change_x = -MOVEMENT_SPEED
            if key == arcade.key.RIGHT:
                self.player.change_x = MOVEMENT_SPEED

    def on_key_release(self, key, modifiers):
        """
        Called when the user releases a key.
        """
        if self.scene == 1:
            if key == arcade.key.LEFT or key == arcade.key.RIGHT:
                self.player.change_x = 0

    def on_mouse_press(self, x, y, button, modifiers):
        """
        Called when the left mouse button is pressed.
        """
        vert = (x < self.start_button.center_x + self.start_button.width // 2 and  x > self.start_button.center_x - self.start_button.width // 2)
        hor = (y < self.start_button.center_y + self.start_button.height // 2 and  y > self.start_button.center_y - self.start_button.height // 2)
        if vert and hor and self.scene==0:
            self.scene = 1 
            self.background = arcade.load_texture('images/game_bg.png')

    def on_update(self, delta_time):
        """ Movement and game logic """
        # Move the obstacles
        if self.scene == 1:
            for obst in self.obst_list:
                obst.center_y -= OBST_SPEED

        # Move the player
        if self.player.center_x < 25:
            self.player.change_x = 0 
            self.player.center_x +=MOVEMENT_SPEED
        if self.player.center_x > SCREEN_WIDTH - 25:
            self.player.change_x = 0 
            self.player.center_x -=MOVEMENT_SPEED
        self.player_list.update()
        # Visual addition - moving start button 
        self.timer +=1
        if self.timer >= 12 and self.start_button.scale==BUTTON_SCALE:
            self.start_button.scale+=0.01
            self.timer=0
        elif self.timer >= 12 and self.start_button.scale!=BUTTON_SCALE:
            self.start_button.scale-=0.01
            self.timer=0
        # Update the players animation
        self.player_list.update_animation()

        # Generate a list of all sprites that collided with the player.
        hit_coins = arcade.check_for_collision_with_list(self.player, self.coin_list)
        hit_obst = arcade.check_for_collision_with_list(self.player,self.obst_list)
        
        # Loop through each colliding sprite, remove it, and add to the score.
        for coin in hit_coins:
            coin.remove_from_sprite_lists()
            self.score += 100
            # Adding coin after hitting one 
            coin = arcade.Sprite("images/coin.png", scale=COIN_SCALE)
            coin.center_x = random.randrange(SCREEN_WIDTH)
            if coin.center_x - self.player.center_x < 100 and coin.center_x - self.player.center_x > 0:
                coin.center_x += 100
            elif coin.center_x - self.player.center_x > -100 and coin.center_x - self.player.center_x < 0:
                coin.center_x -= 100
            coin.center_x = coin.center_x % SCREEN_WIDTH
            coin.center_y = 100
            self.coin_list.append(coin)

        for obst in hit_obst:
            obst.remove_from_sprite_lists()
            self.scene = 2

window = MyGameWindow(SCREEN_WIDTH,SCREEN_HEIGHT,title)
# Window setup 
window.setup()
arcade.run()