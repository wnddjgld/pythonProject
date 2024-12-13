class Car:
    def __init__(self, model, engine, color='black'):
        self.model=model
        self.engine=engine
        self.color=color
    def info(self):
        print(f'Model: {self.model} / {self.engine}cc')
        print(f'Color {self.color}')
    def run(self):
        print(f'{self.color} {self.model} 붕붕~~~')