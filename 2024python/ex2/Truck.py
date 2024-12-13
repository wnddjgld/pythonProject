from Car import Car

class Truck(Car):
    def __init__(self,model,engine,color='blue',cap=5000):
        #super(Truck, self) ._ init_(model, engine, color)
        Car.__init__(self,model,engine, color)
        self.capacity=cap
        self.loading=0
    def load(self, w):
        self.loading+=w
    def info(self):
        #super(Truck,self).info()
        Car.info(self)
        print(f'Capacity :{self.capacity/1000 : .1f} ton')
        print(f'Current loading: {self.loading} kg')