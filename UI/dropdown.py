import pygame

class Dropdown:
    def __init__(self, x, y, w, h, font, main_color, hover_color, options, starting_index=0):
        self.rect = pygame.Rect(x, y, w, h)
        self.font = font
        self.main_color = main_color
        self.hover_color = hover_color
        self.options = options
        self.selected_index = starting_index
        self.expanded = False
        self.option_rects = [pygame.Rect(x, y + (i + 1) * h, w, h) for i in range(len(options))]

    def draw(self, surface):
        # Draw main box
        pygame.draw.rect(surface, self.main_color, self.rect)
        label = self.font.render(self.options[self.selected_index], True, (0, 0, 0))
        surface.blit(label, (self.rect.x + 5, self.rect.y + 5))

        # Draw arrow
        pygame.draw.polygon(surface, (0, 0, 0), [
            (self.rect.right - 15, self.rect.y + 10),
            (self.rect.right - 5, self.rect.y + 10),
            (self.rect.right - 10, self.rect.y + 20)
        ])

        # Draw options if expanded
        if self.expanded:
            for i, option_rect in enumerate(self.option_rects):
                pygame.draw.rect(surface, self.hover_color if option_rect.collidepoint(pygame.mouse.get_pos()) else self.main_color, option_rect)
                label = self.font.render(self.options[i], True, (0, 0, 0))
                surface.blit(label, (option_rect.x + 5, option_rect.y + 5))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.expanded = not self.expanded
            elif self.expanded:
                for i, option_rect in enumerate(self.option_rects):
                    if option_rect.collidepoint(event.pos):
                        self.selected_index = i
                        self.expanded = False
                        return True  # selection changed
                self.expanded = False
        return False  # no change

    def get_selected(self):
        return self.options[self.selected_index]