import pygame
import sys
import time
import os
 
def display_grayscale_patch():
    pygame.init()
    num_displays = pygame.display.get_num_displays()
    display_sizes = pygame.display.get_desktop_sizes()
    monitor_index = 1 if num_displays > 1 else 0
    screen_size = display_sizes[monitor_index]
    monitor_x = sum(display_sizes[i][0] for i in range(monitor_index)) if monitor_index > 0 else 0
    monitor_y = 0
 
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{monitor_x},{monitor_y}"
 
    screen = pygame.display.set_mode(screen_size, pygame.NOFRAME)
    pygame.display.set_caption("Grayscale Patch")
 
    font = pygame.font.SysFont(None, 60)
 
    running = True
    gray_value = 0
    clock = pygame.time.Clock()
    last_update = time.time()
 
    # Rectangle size (smaller box)
    rect_width = 300
    rect_height = 300
 
    # Position: slightly right of center
    rect_x = 960 - 50
    print(rect_x)
    rect_y = 600 - 150
    print(rect_y)
 
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
 
        current_time = time.time()
 
        # Update every second
        if current_time - last_update >= 1:
            gray_value += 10
            if gray_value > 255:
                gray_value = 0
            last_update = current_time
 
        # White background
        screen.fill((255, 255, 255))
 
        # Draw grayscale rectangle
        pygame.draw.rect(
            screen,
            (gray_value, gray_value, gray_value),
            (rect_x, rect_y, rect_width, rect_height)
        )
 
        # Optional: display value inside rectangle
        #text_surface = font.render(f"{gray_value}", True, (255 - gray_value, 0, 0))
        #text_rect = text_surface.get_rect(center=(
        #    rect_x + rect_width // 2,
        #    rect_y + rect_height // 2
        #))
        #screen.blit(text_surface, text_rect)
 
        pygame.display.flip()
        clock.tick(60)

 
    pygame.quit()
    sys.exit()
 
 
# Run
display_grayscale_patch()
