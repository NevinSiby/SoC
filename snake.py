import pygame,sys,random
from pygame.math import Vector2 as v2
pygame.font.init()
Game_over_font, BUTTON_FONT, Score_font = pygame.font.Font(None, 62), pygame.font.Font(None,48), pygame.font.Font(None, 30)
cell_no, cell_len = 30, 20
Blue, Dark_Green, White, Red, Black, Green= (0,0,255), (11,73,37), (255,255,255), (255,0,0), (0,0,0), (0,255,0)
WIDTH_SCREEN , HEIGHT_SCREEN = cell_no*cell_len, cell_no*cell_len
WIN = pygame.display.set_mode((WIDTH_SCREEN, HEIGHT_SCREEN))
pygame.display.set_caption("Snake Game")
SCREENUPDATE, GAMEOVER = pygame.USEREVENT, pygame.USEREVENT + 1
Clock = pygame.time.Clock()
class Fruit:
    def __init__ (self):
        self.pos = v2(random.randint(0,cell_no-1),random.randint(0,cell_no-1))
    def draw(self, snake):
        while self.pos in snake:
            self.pos = v2(random.randint(0,cell_no-1),random.randint(0,cell_no-1))           
        self.fruit = pygame.Rect(self.pos.x*cell_len,self.pos.y*cell_len,cell_len,cell_len)
        pygame.draw.rect(WIN, Red, self.fruit)        
    def fruit_acquire(self,snakeblock_vector):
        return  (self.pos == snakeblock_vector)
class Snake:
    def __init__(self):
        self.Snake_body = [v2(5,20),v2(6,20), v2(7,20)]
        self.movevector = v2(1,0)
    def draw(self):
        for pos in self.Snake_body:
            Block = pygame.Rect(pos.x*cell_len,pos.y*cell_len,cell_len,cell_len)
            pygame.draw.rect(WIN, Blue, Block)
    def Movement(self):
        for i in range((len(self.Snake_body) - 1), 0, -1):
            self.Snake_body[i].x,self.Snake_body[i].y = self.Snake_body[i-1].x,self.Snake_body[i-1].y
        self.Snake_body[0] += self.movevector
    def move(self, keys_pressed):
        if keys_pressed[pygame.K_LEFT] and self.movevector.x == 0:
            self.movevector = v2(-1,0)
        if keys_pressed[pygame.K_RIGHT] and self.movevector.x == 0:
            self.movevector = v2(1,0)
        if keys_pressed[pygame.K_UP] and self.movevector.y == 0:
            self.movevector = v2(0,-1)
        if keys_pressed[pygame.K_DOWN] and self.movevector.y == 0:
            self.movevector = v2(0,1)
    def add(self):
        tail = self.Snake_body[-1]
        new = v2(tail.x, tail.y)
        self.Snake_body.append(new)
    def handle_mechanic(self):
        head = self.Snake_body[0]
        if head.x > cell_no-1 or head.x<0 or head.y>cell_no-1 or head.y<0:
            pygame.event.post(pygame.event.Event(GAMEOVER))
        snakehead = pygame.Rect(head.x*cell_len, head.y*cell_len,cell_len,cell_len)
        if len(self.Snake_body) > 4: 
            for i in range(4, len(self.Snake_body)):
                snakebody = pygame.Rect(self.Snake_body[i].x*cell_len,self.Snake_body[i].y*cell_len,cell_len,cell_len)
                if snakebody.colliderect(snakehead):
                    pygame.event.post(pygame.event.Event(GAMEOVER))
def Draw(fruit,snake, text):
    WIN.fill(Dark_Green)
    snake.draw()
    fruit.draw(snake.Snake_body[1:])
    WIN.blit(text,(WIDTH_SCREEN - text.get_width() - 5, text.get_height()+5))
    pygame.display.update()
def draw_button(Text, Color, Text_color):
    Button_Text = BUTTON_FONT.render(Text,1,Text_color,Color)
    Button_Width, Button_height = Button_Text.get_width() + 40, Button_Text.get_height() + 20
    Button = pygame.Rect(WIDTH_SCREEN//2 - Button_Width//2, HEIGHT_SCREEN//2 - Button_height//2, Button_Width, Button_height)
    pygame.draw.rect(WIN, Color, Button)
    WIN.blit(Button_Text,(WIDTH_SCREEN//2 - Button_Width//2 + 20, HEIGHT_SCREEN//2 - Button_height//2 + 10))
    pygame.display.update()
    return Button
def start(Button_Text):
    WIN.fill(Black)
    Button = draw_button(Button_Text, Dark_Green, White)
    waiting = True
    while waiting:
        pos = pygame.mouse.get_pos()
        if Button.collidepoint(pos):
            Button = draw_button(Button_Text,Green, Black)
        else:
            Button = draw_button(Button_Text,Dark_Green, White)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == pygame.MOUSEBUTTONDOWN:
                mouse = pygame.mouse.get_pos()
                if Button.collidepoint(mouse):
                    main()
def main():
    snake = Snake()
    fruit = Fruit()
    score = 0
    pygame.time.set_timer(SCREENUPDATE, 150)
    while True:
        score_text = Score_font.render(f'Score : {score}', 1, White)
        Draw(fruit,snake,score_text)
        keys_pressed = pygame.key.get_pressed()
        snake.move(keys_pressed)        
        if fruit.fruit_acquire(snake.Snake_body[0]):
            score += 1
            score_text = Score_font.render(f'Score : {score}', 1, White)
            del fruit
            fruit = Fruit()            
            fruit.draw(snake.Snake_body[1:])
            snake.add()
            Draw(fruit, snake,score_text)
        snake.handle_mechanic()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == SCREENUPDATE:
                snake.Movement()
            if e.type == GAMEOVER:
                Game_Over_text = Game_over_font.render(f'Game Over! You Score : {score}',1, White)
                WIN.blit(Game_Over_text, (WIDTH_SCREEN//2 - Game_Over_text.get_width()//2 , HEIGHT_SCREEN//2 - Game_Over_text.get_height()//2))
                pygame.display.update()
                pygame.time.delay(3000)
                del snake
                del fruit
                start('RESTART')
        Clock.tick(60)
start('START')