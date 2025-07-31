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
    def __sub__(self, other):
        if isinstance(other, Snake):
            return self.pos - other.Snake_body[0]
class Snake:
    def __init__(self):
        self.Snake_body = [v2(5,20),v2(6,20), v2(7,20)]
        self.movevector = v2(1,0)
    def draw(self):
        for pos in self.Snake_body:
            Block = pygame.Rect(pos.x*cell_len,pos.y*cell_len,cell_len,cell_len)
            pygame.draw.rect(WIN, Blue, Block)
    def Movement(self, f):
        for i in range((len(self.Snake_body) - 1), 0, -1):
            self.Snake_body[i].x,self.Snake_body[i].y = self.Snake_body[i-1].x,self.Snake_body[i-1].y
        prev = self.Snake_body[0]
        self.Snake_body[0] += self.movevector
        curr = self.Snake_body[0]
        if f is not None:
            return prev.distance_to(f.pos), curr.distance_to(f.pos)
    def move(self, keys_pressed):
        if keys_pressed[pygame.K_LEFT] and self.movevector.x == 0:
            self.movevector = v2(-1, 0)
        if keys_pressed[pygame.K_RIGHT] and self.movevector.x == 0:
            self.movevector = v2(1, 0)
        if keys_pressed[pygame.K_UP] and self.movevector.y == 0:
            self.movevector = v2(0, -1)
        if keys_pressed[pygame.K_DOWN] and self.movevector.y == 0:
            self.movevector = v2(0, 1)
    def action(self, n):
        if n==0 and self.movevector.x == 0:
            self.movevector = v2(-1, 0)
        if n==1 and self.movevector.x == 0:
            self.movevector = v2(1, 0)
        if n==2 and self.movevector.y == 0:
            self.movevector = v2(0, 1)
        if n==3 and self.movevector.y == 0:
            self.movevector = v2(0, -1)
    def add(self):
        tail = self.Snake_body[-1]
        new = v2(tail.x, tail.y)
        self.Snake_body.append(new)
    def handle_mechanic(self, fruit, score):
        head = self.Snake_body[0]
        if head.x > cell_no-1 or head.x<0 or head.y>cell_no-1 or head.y<0:
            pygame.event.post(pygame.event.Event(GAMEOVER))
        snakehead = pygame.Rect(head.x*cell_len, head.y*cell_len,cell_len,cell_len)
        if len(self.Snake_body) > 4: 
            if self.Snake_body[0] in self.Snake_body[4:]:
                pygame.event.post(pygame.event.Event(GAMEOVER))
        if self.Snake_body[0] == fruit.pos:
            score += 1
            del fruit
            fruit = Fruit()
            fruit.draw(self.Snake_body[1:])
            self.add()
        return score, fruit

    def RETURNING_ALL(self, fruit, scr, s):
        head = self.Snake_body[0]
        REWARD = 0.01
        DONE = False
        if head.x > cell_no-1 or head.x<0 or head.y>cell_no-1 or head.y<0:
            REWARD = -1000
            DONE = True
        elif len(self.Snake_body) > 4 and head in self.Snake_body[4:]:
            REWARD = -800
            DONE = True

        elif head == fruit.pos:
            scr += 1
            REWARD = 200 + scr * 100
            fruit = Fruit()
            fruit.draw(self.Snake_body[1:])
            self.add()
            s = 0


        vec = fruit - self
        a = vec.x / abs(vec.x) if vec.x != 0 else 0
        b = vec.y / abs(vec.y) if vec.y != 0 else 0
#        move_dir = v2(a, b)
#        REWARD += (move_dir.dot(self.movevector) * 5)

        S = head + self.movevector
        R = head + v2(self.movevector.y, self.movevector.x)
        L = head + v2(-self.movevector.y, self.movevector.x)

        Dang_S = int(S.x >= cell_no or S.x < 0 or S.y >= cell_no or S.y < 0 or S in self.Snake_body[4:])
        Dang_R = int(R.x >= cell_no or R.x < 0 or R.y >= cell_no or R.y < 0 or R in self.Snake_body[4:])
        Dang_L = int(L.x >= cell_no or L.x < 0 or L.y >= cell_no or L.y < 0 or L in self.Snake_body[4:])
        if s > 500 + (15*scr): DONE = True

        STATE = [Dang_S, Dang_R, Dang_L, int(a + 1), int(b + 1)]
        return fruit, STATE, REWARD, DONE, scr, s

    

def Draw(fruit,snake, text):
    WIN.fill(Dark_Green)
    snake.draw()
    fruit.draw(snake.Snake_body[1:])
    if text is not None:
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
    pygame.time.set_timer(SCREENUPDATE, 100)
    while True:
        score_text = Score_font.render(f'Score : {score}', 1, White)
        Draw(fruit,snake,score_text)
        keys_pressed = pygame.key.get_pressed()
        snake.move(keys_pressed)        
        score, fruit = snake.handle_mechanic(fruit, score)
        score_text = Score_font.render(f'Score : {score}', 1, White)
        
        Draw(fruit, snake, score_text)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == SCREENUPDATE:
                snake.Movement(None)
            if e.type == GAMEOVER:
                Game_Over_text = Game_over_font.render(f'Game Over! You Score : {score}',1, White)
                WIN.blit(Game_Over_text, (WIDTH_SCREEN//2 - Game_Over_text.get_width()//2 , HEIGHT_SCREEN//2 - Game_Over_text.get_height()//2))
                pygame.display.update()
                pygame.time.delay(3000)
                del snake
                del fruit
                start('RESTART')
        Clock.tick(60)



if __name__ == "__main__":
    start('START')