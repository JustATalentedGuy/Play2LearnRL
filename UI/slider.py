import pygame

class Slider:
    def __init__(self, x, y, w, min_val, max_val, start_val, label="", step=0.01):
        self.rect = pygame.Rect(x, y, w, 20)
        self.knob_radius = 10
        self.min = min_val
        self.max = max_val
        self.value = start_val
        self.dragging = False
        self.step = step
        self.label = label
        self.font = pygame.font.SysFont("Arial", 16)

    def draw(self, surface):
        pygame.draw.rect(surface, (180, 180, 180), self.rect, 2)
        t = (self.value - self.min) / (self.max - self.min)
        knob_x = self.rect.x + int(t * self.rect.width)
        pygame.draw.circle(surface, (200, 60, 60), (knob_x, self.rect.centery), self.knob_radius)

        # Text
        label_surface = self.font.render(f"{self.label}: {self.value:.4f}", True, (255, 255, 255))
        surface.blit(label_surface, (self.rect.x, self.rect.y - 25))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            rel_x = event.pos[0] - self.rect.x
            rel_x = max(0, min(self.rect.width, rel_x))
            t = rel_x / self.rect.width
            raw_val = self.min + t * (self.max - self.min)
            self.value = round(raw_val / self.step) * self.step